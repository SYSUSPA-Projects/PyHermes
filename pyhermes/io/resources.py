"""Resolve local and remote data resources for PyHermes readers."""

from __future__ import annotations

import hashlib
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from pyhermes.param.logbase import setup_logger


_REMOTE_SCHEMES = {"http", "https"}
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def is_remote_url(path) -> bool:
    """Return whether ``path`` is an HTTP(S) URL."""
    if not isinstance(path, (str, os.PathLike)):
        return False
    return urlparse(os.fspath(path)).scheme.lower() in _REMOTE_SCHEMES


def file_sha256(path, chunk_size=_DOWNLOAD_CHUNK_SIZE) -> str:
    """Return the hexadecimal SHA256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value):
    if value in (None, ""):
        return None
    value = str(value).lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("sha256 must be a 64-character hexadecimal digest.")
    return value


def _default_cache_dir() -> Path:
    configured = os.environ.get("PYHERMES_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "pyhermes"
    return Path.home() / ".cache" / "pyhermes"


def _remote_filename(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    return name or "download"


def remote_cache_path(url, cache_dir=None) -> Path:
    """Return the deterministic default cache path for ``url``."""
    cache_root = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    url_digest = hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"{url_digest}-{_remote_filename(str(url))}"


def _verify_checksum(path: Path, expected_sha256, source: str) -> None:
    if expected_sha256 is None:
        return
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {source}: expected {expected_sha256}, got {actual}."
        )


def resolve_data_path(
    path,
    *,
    cache_path=None,
    cache_dir=None,
    sha256=None,
    timeout=60,
) -> str:
    """Resolve a local path or download an HTTP(S) resource into a cache.

    Existing cache files are reused after optional SHA256 verification. New
    downloads are written to a unique partial file and atomically moved into
    place only after verification succeeds.
    """
    source = os.fspath(path)
    expected_sha256 = _normalize_sha256(sha256)
    if not is_remote_url(source):
        local_path = Path(source).expanduser()
        _verify_checksum(local_path, expected_sha256, str(local_path))
        return str(local_path)

    if cache_path and cache_dir:
        raise ValueError("Set only one of cache_path and cache_dir.")
    target = (
        Path(cache_path).expanduser()
        if cache_path
        else remote_cache_path(source, cache_dir=cache_dir)
    )
    logger = setup_logger(__name__, "resolve_data_path")

    if target.is_file():
        try:
            _verify_checksum(target, expected_sha256, str(target))
        except ValueError:
            logger.warning(
                f"Cached file failed checksum verification; downloading a fresh copy: {target}"
            )
        else:
            logger.info(f"Using cached particle data: {target}")
            return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    logger.info(f"Downloading particle data from {source}")
    try:
        with requests.get(
            source,
            stream=True,
            timeout=float(timeout),
            headers={"User-Agent": "PyHermes particle reader"},
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=32),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                disable=not sys.stderr.isatty(),
            ) as progress:
                task = progress.add_task(f"Downloading {target.name}", total=total)
                with partial.open("wb") as file_obj:
                    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        file_obj.write(chunk)
                        progress.update(task, advance=len(chunk))
        _verify_checksum(partial, expected_sha256, source)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)

    logger.info(f"Cached particle data at {target}")
    return str(target)

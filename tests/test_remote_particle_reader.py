import hashlib
import io

import numpy as np
import pytest

from pyhermes.io import read_particle_data, resolve_data_path


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start:start + chunk_size]


def _npz_payload():
    buffer = io.BytesIO()
    np.savez(
        buffer,
        pos=np.arange(12, dtype=np.float32).reshape(4, 3),
        mass=np.arange(4, dtype=np.float32),
    )
    return buffer.getvalue()


def _fof_payload():
    group_len = np.array([12, 34], dtype=np.int32)
    group_offset = np.array([0, 12], dtype=np.int32)
    group_mass = np.array([1.5, 2.5], dtype=np.float32)
    group_pos = np.array(
        [[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]],
        dtype=np.float32,
    )
    group_vel = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        dtype=np.float32,
    )
    zeros6 = np.zeros((2, 6), dtype=np.float32)
    arrays = [
        np.array([2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([46], dtype=np.int32),
        np.array([46], dtype=np.uint64),
        np.array([1], dtype=np.uint32),
        group_len,
        group_offset,
        group_mass,
        group_pos,
        group_vel,
        zeros6,
        zeros6,
    ]
    return b"".join(array.tobytes() for array in arrays)


def test_remote_npz_is_downloaded_verified_and_reused(tmp_path, monkeypatch):
    payload = _npz_payload()
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(payload)

    monkeypatch.setattr("pyhermes.io.resources.requests.get", fake_get)
    cache_path = tmp_path / "catalog.npz"
    download = {
        "cache_path": cache_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    first = read_particle_data(
        "https://example.invalid/catalog.npz?version=1",
        download=download,
        fields={},
    )
    second = read_particle_data(
        "https://example.invalid/catalog.npz?version=1",
        download=download,
    )

    assert first["pos"].shape == (4, 3)
    np.testing.assert_array_equal(first["mass"], np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(first["pos"], second["pos"])
    assert len(calls) == 1
    assert cache_path.read_bytes() == payload


def test_remote_single_file_fof_is_downloaded_and_read_directly(tmp_path, monkeypatch):
    payload = _fof_payload()
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(payload)

    monkeypatch.setattr("pyhermes.io.resources.requests.get", fake_get)
    cache_path = tmp_path / "group_tab_004.0"
    download = {
        "cache_path": cache_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    first = read_particle_data(
        "https://example.invalid/group_tab_004.0",
        download=download,
        redshift=0.5,
        fields={"mass": "mass", "vx": "vel_x", "npart": "npart"},
    )
    second = read_particle_data(
        "https://example.invalid/group_tab_004.0",
        download=download,
        fields={"mass": "mass"},
    )

    np.testing.assert_allclose(
        first["pos"],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    np.testing.assert_allclose(first["vx"], [15.0, 60.0])
    np.testing.assert_allclose(first["mass"], [1.5e10, 2.5e10])
    np.testing.assert_array_equal(first["npart"], [12, 34])
    np.testing.assert_array_equal(first["pos"], second["pos"])
    assert len(calls) == 1
    assert cache_path.read_bytes() == payload


def test_failed_checksum_does_not_replace_cache_target(tmp_path, monkeypatch):
    payload = _npz_payload()
    monkeypatch.setattr(
        "pyhermes.io.resources.requests.get",
        lambda *args, **kwargs: _FakeResponse(payload),
    )
    cache_path = tmp_path / "catalog.npz"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_data_path(
            "https://example.invalid/catalog.npz",
            cache_path=cache_path,
            sha256="0" * 64,
        )

    assert not cache_path.exists()
    assert not list(tmp_path.glob("*.part"))


def test_local_particle_path_is_unchanged(tmp_path):
    local_path = tmp_path / "catalog.npz"
    local_path.write_bytes(_npz_payload())

    resolved = resolve_data_path(local_path)

    assert resolved == str(local_path)

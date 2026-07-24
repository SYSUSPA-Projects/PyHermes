"""Compatibility helpers for PyHermes pickle-backed data products.

These helpers preserve the existing trusted-pickle file format while removing
``pathlib`` implementation details from newly written metadata. They also map
the private ``pathlib._local`` classes used by Python 3.13 back to public pure
path classes when older Python versions read an existing file.
"""

from __future__ import annotations

import os
import pickle
from io import BytesIO
from pathlib import PurePath, PurePosixPath, PureWindowsPath

import numpy as np

_PATHLIB_COMPAT_CLASSES = {
    "Path": PurePath,
    "PurePath": PurePath,
    "PosixPath": PurePosixPath,
    "PurePosixPath": PurePosixPath,
    "WindowsPath": PureWindowsPath,
    "PureWindowsPath": PureWindowsPath,
}


class _PathlibCompatUnpickler(pickle.Unpickler):
    """Resolve private pathlib classes without importing private modules."""

    def find_class(self, module, name):
        if module in {"pathlib._local", "pathlib._abc"}:
            replacement = _PATHLIB_COMPAT_CLASSES.get(name)
            if replacement is not None:
                return replacement
        return super().find_class(module, name)


def normalize_pathlike(value):
    """Recursively convert path-like metadata to portable strings."""
    if isinstance(value, os.PathLike):
        return os.fsdecode(os.fspath(value))
    if isinstance(value, dict):
        return {
            normalize_pathlike(key): normalize_pathlike(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_pathlike(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_pathlike(item) for item in value)
    if isinstance(value, set):
        return {normalize_pathlike(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(normalize_pathlike(item) for item in value)
    return value


def pickle_loads_compatible(payload):
    """Load one trusted pickle payload and normalize path-like metadata."""
    value = _PathlibCompatUnpickler(BytesIO(payload)).load()
    return normalize_pathlike(value)


def pickle_dumps_compatible(value, protocol=4):
    """Serialize one value after replacing path-like metadata with strings."""
    return pickle.dumps(normalize_pathlike(value), protocol=protocol)


def pickle_load_compatible(file_obj):
    """Load one trusted raw pickle stream with pathlib compatibility."""
    value = _PathlibCompatUnpickler(file_obj).load()
    return normalize_pathlike(value)


def pickle_dump_compatible(value, file_obj, protocol=4):
    """Write one portable trusted raw pickle stream."""
    pickle.dump(normalize_pathlike(value), file_obj, protocol=protocol)


def read_numpy_pickle(file_obj):
    """Read the NumPy-framed pickle format used by PyHermes data classes."""
    serialized = np.lib.format.read_array(file_obj, allow_pickle=False)
    return pickle_loads_compatible(serialized.tobytes())


def write_numpy_pickle(file_obj, value, protocol=4):
    """Write the NumPy-framed pickle format used by PyHermes data classes."""
    serialized = pickle_dumps_compatible(value, protocol=protocol)
    np.lib.format.write_array(file_obj, np.frombuffer(serialized, dtype=np.uint8))

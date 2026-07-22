from io import BytesIO
from pathlib import Path
import pickle
import sys
import types

import numpy as np

from pyhermes.io.pickle_compat import (
    pickle_loads_compatible,
    read_numpy_pickle,
    write_numpy_pickle,
)


def _python313_posix_path_pickle():
    module_name = "pathlib._local"
    module = types.ModuleType(module_name)
    legacy_class = type(
        "PosixPath",
        (object,),
        {
            "__module__": module_name,
            "__reduce__": lambda self: (legacy_class, ("/", "tmp", "field.pkl")),
        },
    )
    module.PosixPath = legacy_class
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        return pickle.dumps(legacy_class(), protocol=4)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def test_python313_private_pathlib_pickle_loads_on_older_python():
    payload = _python313_posix_path_pickle()

    assert pickle_loads_compatible(payload) == "/tmp/field.pkl"


def test_numpy_framed_pickle_normalizes_paths_before_writing():
    dataset = {
        "sfc_info": {"fin": {"path": Path("/tmp/catalog.npz")}},
        "epsilon": np.arange(4, dtype=np.float64),
    }
    buffer = BytesIO()

    write_numpy_pickle(buffer, dataset)
    buffer.seek(0)
    loaded = read_numpy_pickle(buffer)

    assert loaded["sfc_info"]["fin"]["path"] == "/tmp/catalog.npz"
    np.testing.assert_array_equal(loaded["epsilon"], dataset["epsilon"])

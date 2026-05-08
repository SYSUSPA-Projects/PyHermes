import os

import numpy as np

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info


# Shared reader utilities.
def _as_column_selector(value):
    """Normalize one or many column indices to int selectors."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    return int(value)


def _validate_reader_output(data, reader_name):
    """Validate and normalize the shared particle-reader output contract."""
    if not isinstance(data, dict):
        raise TypeError(f"{reader_name} must return a dict, got {type(data)}.")
    if "pos" not in data:
        raise ValueError(f"{reader_name} output must contain a 'pos' array.")
    pos = np.asarray(data["pos"])
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{reader_name} output 'pos' must have shape (N, 3), got {pos.shape}.")
    data["pos"] = np.ascontiguousarray(pos, dtype=np.float32)
    data["size"] = int(data.get("size", data["pos"].shape[0]))
    if data["size"] != data["pos"].shape[0]:
        raise ValueError(
            f"{reader_name} output 'size'={data['size']} does not match pos length {data['pos'].shape[0]}."
        )
    return data


# Generic table readers.
def read_bin(f_in, dtype="float32", ncols=3, pos_cols=(0, 1, 2), fields=None, **kwargs):
    """Read a raw binary table using configurable position and field columns."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {f_in} <---")
    ncols = int(ncols)
    if ncols <= 0:
        raise ValueError("read_bin requires ncols > 0.")
    raw = np.fromfile(f_in, dtype=np.dtype(dtype))
    if raw.size % ncols != 0:
        raise ValueError(
            f"Binary file '{f_in}' contains {raw.size} values, which cannot be reshaped into (-1, {ncols})."
        )
    table = raw.reshape(-1, ncols)
    pos_cols = _as_column_selector(pos_cols)
    if not isinstance(pos_cols, list) or len(pos_cols) != 3:
        raise ValueError("read_bin requires pos_cols to contain exactly three column indices.")
    data = {
        "pos": table[:, pos_cols],
        "size": table.shape[0],
    }
    for key, cols in (fields or {}).items():
        selector = _as_column_selector(cols)
        data[key] = table[:, selector]
    return _validate_reader_output(data, "read_bin")


def read_npz(f_in, pos_key="pos", fields=None, **kwargs):
    """Read a NumPy NPZ particle dataset with optional key remapping."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {f_in} <---")
    data = {}
    with np.load(f_in) as npz_data:
        if pos_key not in npz_data:
            raise ValueError(f"NPZ file '{f_in}' does not contain pos_key='{pos_key}'.")
        data["pos"] = npz_data[pos_key]
        if fields is None:
            for key in npz_data.files:
                if key != pos_key:
                    data[key] = npz_data[key]
        else:
            for out_key, in_key in fields.items():
                if in_key not in npz_data:
                    raise ValueError(f"NPZ file '{f_in}' does not contain key '{in_key}' for field '{out_key}'.")
                data[out_key] = npz_data[in_key]
    data["size"] = np.asarray(data["pos"]).shape[0]
    return _validate_reader_output(data, "read_npz")


# Gadget legacy binary helpers.
def _add_velocity_components(data):
    """Expose vel[:,0:3] as vel_x, vel_y, and vel_z when available."""
    if "vel" in data:
        vel = np.asarray(data["vel"])
        if vel.ndim == 2 and vel.shape[1] == 3:
            data["vel_x"] = vel[:, 0]
            data["vel_y"] = vel[:, 1]
            data["vel_z"] = vel[:, 2]
    return data


def _read_record_block(f, dtype, count, shape=None):
    """Read one Gadget-style record block with leading and trailing byte counts."""
    np.fromfile(f, dtype=np.int32, count=1)
    values = np.fromfile(f, dtype=dtype, count=count)
    np.fromfile(f, dtype=np.int32, count=1)
    if shape is not None:
        values = values.reshape(shape)
    return values


def _read_vec3_block(f, count):
    """Read a Gadget record block containing count three-vectors."""
    return _read_record_block(f, np.float32, count * 3, (-1, 3))


def _concat_particle_blocks(parts, keys=("pos", "vel", "mass")):
    """Concatenate split particle arrays and attach the shared size field."""
    out = {}
    for key in keys:
        out[key] = np.concatenate([part[key] for part in parts], axis=0)
    out["size"] = out["pos"].shape[0]
    return out


def _read_gadget_single(filename, ptype=1):
    """Read one legacy Gadget snapshot file for a single particle type."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    header_dtype = np.dtype([
        ("block1", np.int32),
        ("npart", np.uint32, 6),
        ("massarr", np.float64, 6),
        ("time", np.float64),
        ("redshift", np.float64),
        ("flag_sfr", np.int32),
        ("flag_feedback", np.int32),
        ("npartTotal", np.int32, (6,)),
        ("flag_cooling", np.int32),
        ("num_files", np.int32),
        ("BoxSize", np.float64),
        ("Omega0", np.float64),
        ("OmegaLambda", np.float64),
        ("HubbleParam", np.float64),
        ("fill", np.int32, 24),
        ("block2", np.int32),
    ])
    out = {}
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=header_dtype, count=1)[0]
        if header["npart"][0] != 0:
            logger.error(
                "Currently, only DM-only snapshots in Gadget1/2/3/4 legacy-format1 are supported, "
                "but it appears your file contains SPH particles."
            )
            func_util.safe_exit(1)
        npart = int(header["npart"][ptype])

        out["pos"] = _read_vec3_block(f, npart)
        out["vel"] = _read_vec3_block(f, npart)
        _read_record_block(f, np.int32, npart)

        if header["massarr"][ptype] != 0:
            out["mass"] = float(header["massarr"][ptype])
            masstab = True
        else:
            out["mass"] = _read_record_block(f, np.float32, npart)
            masstab = False
    return out, masstab


def _read_gadget_all(f_in, ptype=1):
    """Read and combine a possibly split legacy Gadget snapshot."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    files = func_util.find_subsplit_files(f_in)
    parts = []
    masstab_pre = None
    for filename in files:
        part, masstab = _read_gadget_single(filename, ptype=ptype)
        if masstab_pre is not None and masstab != masstab_pre:
            logger.error("Inconsistent masstab across subsnaps. Please make sure you are using one simulation.")
            func_util.safe_exit(1)
        masstab_pre = masstab
        parts.append(part)

    out = _concat_particle_blocks(parts, keys=("pos", "vel"))
    out["mass"] = parts[0]["mass"] if masstab_pre else np.concatenate([part["mass"] for part in parts], axis=0)
    return out


# Gadget legacy snapshot reader.
def read_gadget(f_in, **kwargs):
    """Read a legacy Gadget snapshot and return PyHermes particle fields."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {f_in} <---")
    data = _read_gadget_all(f_in, ptype=int(kwargs.get("ptype", 1)))
    data = _validate_reader_output(data, "read_gadget")
    if "mass" in data and np.isscalar(data["mass"]):
        data["mass"] = np.full(data["size"], data["mass"], dtype=np.float32)
    return _add_velocity_components(data)


# Gadget FoF binary helpers.
def _read_gadget_fof_single(filename):
    """Read one legacy Gadget FoF catalog file."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    header_dtype = np.dtype([
        ("block1", np.int32),
        ("Ngroups", np.int64),
        ("Nsubhalos", np.int64),
        ("Nids", np.int64),
        ("TotNgroups", np.int64),
        ("TotNsubhalos", np.int64),
        ("TotNids", np.int64),
        ("num_files", np.int32),
        ("dummy", np.int32),
        ("time", np.float64),
        ("redshift", np.float64),
        ("BoxSize", np.float64),
        ("block2", np.int32),
    ])
    header_fields = [
        "Ngroups",
        "Nsubhalos",
        "Nids",
        "TotNgroups",
        "TotNsubhalos",
        "TotNids",
        "num_files",
        "time",
        "redshift",
        "BoxSize",
    ]
    out = {}
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=header_dtype, count=1)[0]
        out.update({field: header[field] for field in header_fields})
        if out["Nsubhalos"] != 0:
            logger.error(
                "Currently, the 'gadget-fof' reader supports only FoF, i.e. Gadget compiled "
                "with FOF but without SUBFIND."
            )
            func_util.safe_exit(1)

        ngroups = int(out["Ngroups"])
        _read_record_block(f, np.int32, ngroups)
        out["mass"] = _read_record_block(f, np.float32, ngroups)
        out["pos"] = _read_vec3_block(f, ngroups)
        out["vel"] = _read_vec3_block(f, ngroups)
    return out


def _read_gadget_fof_all(f_in):
    """Read and combine a possibly split legacy Gadget FoF catalog."""
    files = func_util.find_subsplit_files(f_in)
    out = {
        "TotNgroups": 0,
        "TotNsubhalos": 0,
        "TotNids": 0,
        "num_files": 0,
        "time": 0.0,
        "redshift": 0.0,
        "BoxSize": 0.0,
    }
    parts = []
    for filename in files:
        part = _read_gadget_fof_single(filename)
        for key in ("TotNgroups", "TotNsubhalos", "TotNids", "num_files", "time", "redshift", "BoxSize"):
            out[key] = part[key]
        parts.append(part)
    out.update(_concat_particle_blocks(parts, keys=("pos", "vel", "mass")))
    return out


# Gadget FoF catalog reader.
def read_gadget_fof(f_in, **kwargs):
    """Read a legacy Gadget FoF catalog and return PyHermes particle fields."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {f_in} <---")
    data = _validate_reader_output(_read_gadget_fof_all(f_in), "read_gadget_fof")
    return _add_velocity_components(data)


# Quijote/Pylians FoF group_tab reader.
def read_fof(f_in, snapnum, redshift=0.0, **kwargs):
    """Read a Quijote/Pylians FoF group_tab catalog directory."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading Quijote FoF halo data from ---> {f_in} <---")
    ext = f"{int(snapnum):03d}"
    prefix = os.path.join(f_in, f"groups_{ext}", f"group_tab_{ext}.")
    vector3 = np.dtype((np.float32, 3))
    vector6 = np.dtype((np.float32, 6))

    group_len = None
    group_offset = None
    group_mass = None
    group_pos = None
    group_vel = None
    skip = 0
    file_index = 0
    nfiles = None
    while nfiles is None or file_index < nfiles:
        filename = f"{prefix}{file_index}"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing Quijote FoF tab file: {filename}")
        with open(filename, "rb") as f:
            ngroups = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            total_groups = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            _ = np.fromfile(f, dtype=np.int32, count=1)
            _ = np.fromfile(f, dtype=np.uint64, count=1)
            nfiles = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

            if file_index == 0:
                group_len = np.empty(total_groups, dtype=np.int32)
                group_offset = np.empty(total_groups, dtype=np.int32)
                group_mass = np.empty(total_groups, dtype=np.float32)
                group_pos = np.empty((total_groups, 3), dtype=np.float32)
                group_vel = np.empty((total_groups, 3), dtype=np.float32)

            loc = slice(skip, skip + ngroups)
            group_len[loc] = np.fromfile(f, dtype=np.int32, count=ngroups)
            group_offset[loc] = np.fromfile(f, dtype=np.int32, count=ngroups)
            group_mass[loc] = np.fromfile(f, dtype=np.float32, count=ngroups)
            group_pos[loc] = np.fromfile(f, dtype=vector3, count=ngroups)
            group_vel[loc] = np.fromfile(f, dtype=vector3, count=ngroups)
            np.fromfile(f, dtype=vector6, count=ngroups)
            np.fromfile(f, dtype=vector6, count=ngroups)
            if f.tell() != os.path.getsize(filename):
                raise ValueError(
                    f"Finished reading {filename} before EOF. read_fof supports standard Quijote "
                    "group_tab files without extra SFR blocks."
                )
        skip += ngroups
        file_index += 1

    vel = group_vel * (1.0 + float(redshift))
    data = {
        "pos": group_pos / 1e3,
        "vel": vel,
        "vel_x": vel[:, 0],
        "vel_y": vel[:, 1],
        "vel_z": vel[:, 2],
        "mass": group_mass * 1e10,
        "npart": group_len,
        "group_offset": group_offset,
        "size": group_pos.shape[0],
    }
    return _validate_reader_output(data, "read_fof")


# Reader dispatcher.
FORMAT_READERS = {
    "bin": read_bin,
    "npz": read_npz,
    "gadget": read_gadget,
    "gadget-fof": read_gadget_fof,
    "fof": read_fof,
}


def read_particle_data(f_in, f_format, reader_params=None):
    """Dispatch a local particle catalog path to one of the registered readers."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    if str(f_in).lower().startswith(("http://", "https://")):
        logger.error("Remote particle inputs are no longer supported. Download the data first and pass a local path.")
        func_util.safe_exit(1)
    if f_format not in FORMAT_READERS:
        supported_formats = ", ".join(FORMAT_READERS.keys())
        logger.error(f"Unsupported input particle format: {f_format}")
        logger.error(f"Supported formats: '{supported_formats}'")
        func_util.safe_exit(1)
    logger.info(f"Selected input particle format: {f_format}")
    params = {} if reader_params is None else dict(reader_params)
    return FORMAT_READERS[f_format](f_in, **params)


# Weight selection used by Convols and ConvolsData.
def resolve_particle_weight(particle_data, weight_key, logger=None):
    """Resolve a unit or named one-dimensional particle weight array."""
    size = int(particle_data["size"])
    if weight_key is None:
        return np.ones(size, dtype=np.float32), None
    if weight_key not in particle_data:
        available = list(particle_data.keys())
        message = f"Weight key '{weight_key}' not found in particle data. Available keys: {available}."
        if logger is not None:
            logger.error(message)
        raise KeyError(message)
    weight = np.asarray(particle_data[weight_key])
    if weight.ndim != 1 or weight.shape[0] != size:
        message = (
            f"Weight key '{weight_key}' must refer to a 1D array with length {size}, "
            f"but got shape {weight.shape}."
        )
        if logger is not None:
            logger.error(message)
        raise ValueError(message)
    return np.ascontiguousarray(weight, dtype=np.float32), weight_key

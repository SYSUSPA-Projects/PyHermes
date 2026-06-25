from pathlib import Path

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


def _infer_format_from_path(path):
    """Infer a reader format from the final file suffix."""
    suffix = Path(path).suffix.lower().lstrip(".")
    suffix_to_format = {
        "bin": "bin",
        "npz": "npz",
        "gadget": "gadget",
        "fof": "fof",
    }
    return suffix_to_format.get(suffix)


# Generic table readers.
def read_bin(path, dtype="float32", ncols=3, pos_cols=(0, 1, 2), fields=None, **kwargs):
    """Read a raw binary table using configurable position and field columns."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {path} <---")
    ncols = int(ncols)
    if ncols <= 0:
        raise ValueError("read_bin requires ncols > 0.")
    raw = np.fromfile(path, dtype=np.dtype(dtype))
    if raw.size % ncols != 0:
        raise ValueError(
            f"Binary file '{path}' contains {raw.size} values, which cannot be reshaped into (-1, {ncols})."
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


def read_npz(path, pos_key="pos", fields=None, **kwargs):
    """Read a NumPy NPZ particle dataset with optional key remapping."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {path} <---")
    data = {}
    with np.load(path) as npz_data:
        if pos_key not in npz_data:
            raise ValueError(f"NPZ file '{path}' does not contain pos_key='{pos_key}'.")
        data["pos"] = npz_data[pos_key]
        if fields is None:
            for key in npz_data.files:
                if key != pos_key:
                    data[key] = npz_data[key]
        else:
            for out_key, in_key in fields.items():
                if in_key not in npz_data:
                    raise ValueError(f"NPZ file '{path}' does not contain key '{in_key}' for field '{out_key}'.")
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


def _read_gadget_all(path, ptype=1):
    """Read and combine a possibly split legacy Gadget snapshot."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    files = func_util.find_subsplit_files(path)
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
def read_gadget(path, **kwargs):
    """Read a legacy Gadget snapshot and return PyHermes particle fields."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {path} <---")
    data = _read_gadget_all(path, ptype=int(kwargs.get("ptype", 1)))
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


def _read_gadget_fof_all(path):
    """Read and combine a possibly split legacy Gadget FoF catalog."""
    files = func_util.find_subsplit_files(path)
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
def read_gadget_fof(path, **kwargs):
    """Read a legacy Gadget FoF catalog and return PyHermes particle fields."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading particle data from ---> {path} <---")
    data = _validate_reader_output(_read_gadget_fof_all(path), "read_gadget_fof")
    return _add_velocity_components(data)


# Quijote/Pylians FoF group_tab reader.
def _select_particle_fields(data, fields, reader_name):
    """Select or rename optional reader fields while preserving pos and size."""
    if fields is None:
        return data
    selected = {
        "pos": data["pos"],
        "size": data["size"],
    }
    for out_key, in_key in fields.items():
        if in_key not in data:
            raise ValueError(f"{reader_name} output does not contain field '{in_key}' for output field '{out_key}'.")
        selected[out_key] = data[in_key]
    return selected


def _format_fof_catalog(group_pos, group_vel, group_mass, group_len, group_offset, redshift):
    """Convert raw FoF arrays into the shared PyHermes reader schema."""
    vel = group_vel * (1.0 + float(redshift))
    return {
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


def _read_fof_array(file_obj, dtype, count, swap=False):
    """Read a fixed-size FoF block and optionally byteswap it."""
    values = np.fromfile(file_obj, dtype=dtype, count=count)
    if len(values) != count:
        raise EOFError(f"FoF file ended early while reading {count} values of dtype {dtype}.")
    if swap:
        values = values.byteswap()
    return values


def _fof_tab_path(path, snapnum, file_index, prefix="/groups_"):
    """Return the path of one Quijote/Pylians-style group_tab split file."""
    snap_ext = f"{int(snapnum):03d}"
    group_dir = f"{str(prefix).strip('/')}{snap_ext}"
    return Path(path) / group_dir / f"group_tab_{snap_ext}.{file_index}"


def _read_fof_tab_file(filename, swap=False, sfr=False):
    """Read one Quijote/Pylians-style FoF group_tab file."""
    vec3 = np.dtype((np.float32, 3))
    vec6 = np.dtype((np.float32, 6))
    with open(filename, "rb") as file_obj:
        header = {
            "Ngroups": int(_read_fof_array(file_obj, np.int32, 1, swap=swap)[0]),
            "TotNgroups": int(_read_fof_array(file_obj, np.int32, 1, swap=swap)[0]),
            "Nids": int(_read_fof_array(file_obj, np.int32, 1, swap=swap)[0]),
            "TotNids": int(_read_fof_array(file_obj, np.uint64, 1, swap=swap)[0]),
            "Nfiles": int(_read_fof_array(file_obj, np.uint32, 1, swap=swap)[0]),
        }

        ngroups = header["Ngroups"]
        part = {
            "GroupLen": _read_fof_array(file_obj, np.int32, ngroups, swap=swap),
            "GroupOffset": _read_fof_array(file_obj, np.int32, ngroups, swap=swap),
            "GroupMass": _read_fof_array(file_obj, np.float32, ngroups, swap=swap),
            "GroupPos": _read_fof_array(file_obj, vec3, ngroups, swap=swap),
            "GroupVel": _read_fof_array(file_obj, vec3, ngroups, swap=swap),
        }

        _read_fof_array(file_obj, vec6, ngroups, swap=swap)
        _read_fof_array(file_obj, vec6, ngroups, swap=swap)
        if sfr:
            part["GroupSFR"] = _read_fof_array(file_obj, np.float32, ngroups, swap=swap)

        end_pos = file_obj.tell()
        file_obj.seek(0, 2)
        if end_pos != file_obj.tell():
            raise ValueError(f"Finished reading before EOF for FoF tab file: {filename}")

    return header, part


def _read_fof_catalog(path, snapnum, redshift=0.0, **kwargs):
    """Read a Quijote/Pylians-style FoF group_tab catalog without readfof."""
    swap = bool(kwargs.get("swap", False))
    sfr = bool(kwargs.get("SFR", False))
    prefix = kwargs.get("prefix", "/groups_")

    first_file = _fof_tab_path(path, snapnum, 0, prefix=prefix)
    if not first_file.exists():
        raise FileNotFoundError(f"FoF group_tab file not found: {first_file}")

    header0, part0 = _read_fof_tab_file(first_file, swap=swap, sfr=sfr)
    total_groups = int(header0["TotNgroups"])
    num_files = int(header0["Nfiles"])
    parts = [part0]

    for file_index in range(1, num_files):
        filename = _fof_tab_path(path, snapnum, file_index, prefix=prefix)
        if not filename.exists():
            raise FileNotFoundError(f"FoF group_tab file not found: {filename}")
        header, part = _read_fof_tab_file(filename, swap=swap, sfr=sfr)
        if header["TotNgroups"] != total_groups or header["Nfiles"] != num_files:
            raise ValueError(f"Inconsistent FoF header in split file: {filename}")
        parts.append(part)

    group_len = np.concatenate([part["GroupLen"] for part in parts])
    group_offset = np.concatenate([part["GroupOffset"] for part in parts])
    group_mass = np.concatenate([part["GroupMass"] for part in parts])
    group_pos = np.concatenate([part["GroupPos"] for part in parts], axis=0)
    group_vel = np.concatenate([part["GroupVel"] for part in parts], axis=0)
    if group_pos.shape[0] != total_groups:
        raise ValueError(
            f"FoF catalog expected {total_groups} groups, but read {group_pos.shape[0]} groups."
        )

    return _format_fof_catalog(
        group_pos,
        group_vel,
        group_mass,
        group_len,
        group_offset,
        redshift,
    )


def read_fof(path, snapnum, redshift=0.0, fields=None, **kwargs):
    """Read a Quijote/Pylians-style FoF group_tab catalog directory."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f"Reading Quijote FoF halo data from ---> {path} <---")
    data = _read_fof_catalog(path, snapnum, redshift=redshift, **kwargs)
    data = _select_particle_fields(data, fields, "read_fof")
    return _validate_reader_output(data, "read_fof")


# Reader dispatcher.
FORMAT_READERS = {
    "bin": read_bin,
    "npz": read_npz,
    "gadget": read_gadget,
    "gadget-fof": read_gadget_fof,
    "fof": read_fof,
}


def read_particle_data(path, data_format=None, **reader_params):
    """Dispatch a local particle catalog path to one of the registered readers."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    data_format = reader_params.pop("f_format", data_format)
    data_format = reader_params.pop("format", data_format)
    if data_format == "":
        data_format = None
    if data_format is None:
        data_format = _infer_format_from_path(path)
    if data_format is None:
        supported_formats = ", ".join(FORMAT_READERS.keys())
        logger.error(f"Could not infer input particle format from path: {path}")
        logger.error(f"Set fin.format explicitly. Supported formats: '{supported_formats}'")
        func_util.safe_exit(1)
    if data_format not in FORMAT_READERS:
        supported_formats = ", ".join(FORMAT_READERS.keys())
        logger.error(f"Unsupported input particle format: {data_format}")
        logger.error(f"Supported formats: '{supported_formats}'")
        func_util.safe_exit(1)
    logger.info(f"Selected input particle format: {data_format}")
    return FORMAT_READERS[data_format](path, **reader_params)


# Per-particle scalar selection used by SFCProjection and SFCField.
def resolve_particle_value(particle_data, value_key, label="Particle value", logger=None):
    """Resolve a unit or named one-dimensional particle scalar array."""
    size = int(particle_data["size"])
    if value_key is None:
        return np.ones(size, dtype=np.float32), None
    if value_key not in particle_data:
        available = list(particle_data.keys())
        message = f"{label} key '{value_key}' not found in particle data. Available keys: {available}."
        if logger is not None:
            logger.error(message)
        raise KeyError(message)
    value = np.asarray(particle_data[value_key])
    if value.ndim != 1 or value.shape[0] != size:
        message = (
            f"{label} key '{value_key}' must refer to a 1D array with length {size}, "
            f"but got shape {value.shape}."
        )
        if logger is not None:
            logger.error(message)
        raise ValueError(message)
    return np.ascontiguousarray(value, dtype=np.float32), value_key

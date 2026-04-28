import copy

import numpy as np

from pyhermes.io import ConvolsData
from pyhermes.io import WindowFunc
from pyhermes.utils.convolution import specialized_convolution_3d


PRODUCT_RULES = {
    "allowed": {"dd", "dr", "rd", "delta_dd", "rr", "xi"},
    "deps": {
        "xi": ["delta_dd", "rr"],
    },
}

PRODUCT_INPUT_FLAGS = {
    "dd": (True, False),
    "dr": (True, True),
    "rd": (True, True),
    "delta_dd": (True, True),
    "rr": (False, True),
    "xi": (True, True),
}


def los_to_vector(los):
    if isinstance(los, str):
        axis = los.strip().lower()
        if axis == "x":
            return (1.0, 0.0, 0.0)
        if axis == "y":
            return (0.0, 1.0, 0.0)
        if axis == "z":
            return (0.0, 0.0, 1.0)
        raise ValueError("Corr_2PCF 'los' must be one of 'x', 'y', 'z', or a length-3 array.")
    arr = np.asarray(los, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("Corr_2PCF 'los' must be one of 'x', 'y', 'z', or a length-3 array.")
    if np.linalg.norm(arr) == 0.0:
        raise ValueError("Corr_2PCF 'los' vector must be non-zero.")
    return tuple(float(v) for v in arr)


def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def default_pair_window(mode, los_vector):
    if mode == "smu":
        nx, ny, nz = los_vector
        return {
            "type": "ring",
            "len_args": {"R": None, "H": None},
            "other_args": {"nx": nx, "ny": ny, "nz": nz},
            "mapping": "smu_to_RH",
        }
    return {"type": "shell", "len_args": {"R": None}, "other_args": {}, "mapping": "s_to_R"}


def normalize_products(products):
    if isinstance(products, str):
        products = [products]
    elif products is None:
        products = ["xi"]
    elif not isinstance(products, (list, tuple, set)):
        raise TypeError(
            f"Unsupported products input: expected string or array of strings, got {type(products)}."
        )

    allowed = PRODUCT_RULES["allowed"]
    normalized = []
    for item in products:
        if not isinstance(item, str):
            raise TypeError("Each product name must be a string.")
        name = item.strip().lower()
        if name not in allowed:
            raise ValueError(f"Unsupported product '{item}'. Allowed values are {sorted(allowed)}.")
        if name not in normalized:
            normalized.append(name)
    return normalized


def expand_products(products):
    expanded = list(products)
    idx = 0
    while idx < len(expanded):
        for dep in PRODUCT_RULES["deps"].get(expanded[idx], []):
            if dep not in expanded:
                expanded.append(dep)
        idx += 1
    return expanded


def normalize_sampling_array(values, name, positive=True):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise TypeError(f"'{name}' must be a 1D array-like input.")
    if arr.size == 0:
        raise ValueError(f"'{name}' must contain at least one sampling point.")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"'{name}' values must be strictly positive.")
    diffs = np.diff(arr)
    if np.any(diffs < 0.0) and np.any(diffs > 0.0):
        raise ValueError(f"'{name}' values must be monotonic.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def describe_sampling(mode, n_s, s_min, s_max, n_mu, mu_min, mu_max, s_spec, mu_spec):
    base = f"mode={mode}, n_s={n_s}, s_min={s_min}, s_max={s_max}"
    if mode == "smu":
        base += f", n_mu={n_mu}, mu_min={mu_min}, mu_max={mu_max}"
    if isinstance(s_spec, dict) and (mode == "s" or isinstance(mu_spec, dict)):
        return base
    return f"{base}, source=explicit sampling array"


def format_product_list(products):
    return "[" + ", ".join(products) + "]"


def describe_products(products, expanded_products):
    computed = [product for product in expanded_products if product != "xi"]
    derived = ["xi"] if "xi" in expanded_products else []
    text = (
        f"Products: requested={format_product_list(products)} | "
        f"computed={format_product_list(computed)}"
    )
    if derived:
        text += f" | derived={format_product_list(derived)}"
    return text


def describe_task_distribution(total_tasks, n_ranks):
    min_tasks = total_tasks // n_ranks
    max_tasks = min_tasks + (1 if total_tasks % n_ranks else 0)
    return f"Sampling tasks: total={total_tasks}, ranks={n_ranks}, per_rank={min_tasks}-{max_tasks}"


def effective_pair_los(pair_window, mode, los_vector):
    if mode != "smu" or not isinstance(pair_window, dict):
        return None
    other_args = pair_window.get("other_args", {})
    if not all(axis in other_args for axis in ("nx", "ny", "nz")):
        return None
    los_values = tuple(other_args.get(axis) for axis in ("nx", "ny", "nz"))
    if all(value is not None for value in los_values):
        return los_values
    return los_vector


def describe_pair_window(pair_window, mode, los_vector):
    if isinstance(pair_window, dict):
        mapping = pair_window.get("mapping", "custom")
        mapping_name = mapping if isinstance(mapping, str) else getattr(mapping, "__name__", "custom callable")
        parts = [f"type={pair_window.get('type', 'custom')}", f"mapping={mapping_name}"]
        len_args = pair_window.get("len_args", {})
        other_args = pair_window.get("other_args", {})
        if len_args:
            parts.append(f"len_args={len_args}")
        if other_args:
            parts.append(f"other_args={other_args}")
        pair_los = effective_pair_los(pair_window, mode, los_vector)
        if pair_los is not None:
            parts.append(f"los={pair_los}")
        return "Pair window: " + " | ".join(parts)
    return "Pair window: default shell mapping"


def compact_window_desc(win):
    if win is None:
        return "window=none"
    if isinstance(win, dict):
        args = win.get("len_args", {})
        if args:
            return f"window={win.get('type', 'custom')} {args}"
        return f"window={win.get('type', 'custom')}"
    if isinstance(win, WindowFunc):
        args = getattr(win, "len_args", {})
        if args:
            return f"window={getattr(win, 'type', 'custom')} {args}"
        return f"window={getattr(win, 'type', 'custom')}"
    return "window=custom"


def describe_random_input(value):
    if value == "uniform":
        return "uniform random density"
    if isinstance(value, str):
        return f"path={value}"
    if isinstance(value, ConvolsData):
        return "provided random ConvolsData"
    if isinstance(value, (float, int, np.floating)):
        return f"density={float(value):.5e}"
    return "unset"


def serialize_convols_input(value):
    if isinstance(value, str):
        return value
    if value == "uniform":
        return "uniform"
    if isinstance(value, (float, int, np.floating)):
        return float(value)
    if isinstance(value, ConvolsData):
        return {
            "kind": "ConvolsData",
            "L": value.L,
            "box_size": value.box_size,
            "wavelet_mode": value.wavelet_mode,
            "wavelet_level": value.wavelet_level,
        }
    return value


def serialize_window_input(value):
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, WindowFunc):
        return {
            "kind": "WindowFunc",
            "type": getattr(value, "type", "custom"),
            "len_args": copy.deepcopy(getattr(value, "len_args", {})),
            "other_args": copy.deepcopy(getattr(value, "other_args", {})),
        }
    return value


def make_sampling_tasks(mode, s_arr, mu_arr):
    if mode == "smu":
        tasks = np.array(
            [(i, j, s_arr[i], mu_arr[j]) for i in range(len(s_arr)) for j in range(len(mu_arr))],
            dtype=np.float64,
        )
        return tasks, (len(s_arr), len(mu_arr))
    tasks = np.array([(i, -1, s_arr[i], np.nan) for i in range(len(s_arr))], dtype=np.float64)
    return tasks, (len(s_arr),)


def build_result_from_gathered(gathered_tasks, gathered_values, result_shape, mode):
    result = np.empty(result_shape, dtype=np.float64)
    flat_values = [item for sublist in gathered_values for item in sublist]
    flat_tasks = [item for sublist in gathered_tasks for item in sublist]
    for (s_idx, mu_idx), value in zip(flat_tasks, flat_values):
        if mode == "smu":
            result[s_idx, mu_idx] = value
        else:
            result[s_idx] = value
    return result


def mapping_s_to_R(s, mu, pair_window):
    params = copy.deepcopy(pair_window)
    params.setdefault("len_args", {})
    if params["len_args"].get("R") is None:
        params["len_args"]["R"] = s
    return params


def mapping_smu_to_RH(s, mu, pair_window, los_vector=None):
    if mu is None:
        raise ValueError("Corr_2PCF mapping='smu_to_RH' requires a mu value.")
    params = copy.deepcopy(pair_window)
    params.setdefault("len_args", {})
    params.setdefault("other_args", {})
    if params["len_args"].get("R") is None:
        params["len_args"]["R"] = s * np.sqrt(max(0.0, 1.0 - mu * mu))
    if params["len_args"].get("H") is None:
        params["len_args"]["H"] = s * mu
    if los_vector is not None:
        for key, value in zip(("nx", "ny", "nz"), los_vector):
            if key in params["other_args"] and params["other_args"][key] is None:
                params["other_args"][key] = value
    return params


PAIR_WINDOW_MAPPING_MODES = {
    "s_to_R": "s",
    "smu_to_RH": "smu",
}

PAIR_WINDOW_MAPPINGS = {
    "s_to_R": mapping_s_to_R,
    "smu_to_RH": mapping_smu_to_RH,
}


def field_density(field):
    if isinstance(field, (float, int, np.floating)):
        return float(field)
    if isinstance(field, ConvolsData):
        return 1.0 / field.V
    raise TypeError(f"Unsupported field type for density extraction: {type(field)}")


def pair_product_with_window(field1, field2, pair_window_obj, threads):
    if field2 is None:
        field2 = field1
    if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
        return field_density(field1) * field_density(field2)
    conv = specialized_convolution_3d(field1.epsilon, pair_window_obj.as_array(), threads=threads)
    return float(np.einsum("ijk,ijk->", conv, field2.epsilon, optimize=True) / conv.size)


def compute_pair_product_at_smu(s, mu, convols_data1, convols_data2=None, pair_window=None):
    if convols_data2 is None:
        convols_data2 = convols_data1
    if isinstance(convols_data1, (float, int, np.floating)) or isinstance(convols_data2, (float, int, np.floating)):
        return field_density(convols_data1) * field_density(convols_data2)
    if pair_window is None:
        pair_window = {"type": "shell", "len_args": {"R": None}, "other_args": {}, "mapping": "s_to_R"}
    mapping = pair_window.get("mapping")
    if mapping is None:
        mapping = "smu_to_RH" if pair_window.get("type") in ("ring", "cylinder") and mu is not None else "s_to_R"
    if isinstance(mapping, str):
        if mapping not in PAIR_WINDOW_MAPPINGS:
            raise ValueError(
                f"Unsupported pair_window mapping '{mapping}'. "
                f"Supported built-in mappings: {sorted(PAIR_WINDOW_MAPPINGS)}."
            )
        mapper = PAIR_WINDOW_MAPPINGS[mapping]
    elif callable(mapping):
        mapper = mapping
    else:
        raise TypeError("pair_window mapping must be a string or callable.")
    pair_window_params = mapper(s, mu, pair_window)
    if not isinstance(pair_window_params, dict):
        raise TypeError("pair_window mapping must return a pair_window dictionary.")
    pair_window_params = copy.deepcopy(pair_window_params)
    pair_window_params.pop("mapping", None)
    pair_window_params.setdefault("len_args", {})
    pair_window_params.setdefault("other_args", {})
    pair_window_obj = WindowFunc(pair_window_params, convols_data1.convols_info, threads=convols_data1.threads)
    return pair_product_with_window(convols_data1, convols_data2, pair_window_obj, convols_data1.threads)

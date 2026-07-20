import os
import inspect



def get_fname_info():
    # Get the frame object for the caller of this function
    frame = inspect.currentframe().f_back
    # Get the name of the function that called this function
    function_name = frame.f_code.co_name
    # Get the name of the module where the calling function is located
    module_name = frame.f_globals["__name__"]
    return module_name, function_name

def in_jupyter_notebook():
    """
    Check if the current environment is Jupyter Notebook.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    ipython = get_ipython()
    return ipython is not None and "IPKernelApp" in getattr(ipython, "config", {})

def safe_exit(exit_code=1):
    """
    A unified exit function that determines whether to use MPI_Abort or sys.exit
    based on the execution environment. In Jupyter Notebook, it avoids using
    MPI_Abort to prevent kernel crashes.
    Parameters:
    - exit_code (int): The exit code, default is 1.
    """
    if in_jupyter_notebook():
        # If running in Jupyter Notebook, use sys.exit to avoid kernel crash
        print("Detected Jupyter Notebook environment, using sys.exit()")
        import sys
        sys.exit(exit_code)
    else:
        from pyhermes.utils.mpi_util import MPI

        try:
            MPI.COMM_WORLD.Abort(exit_code)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error while trying to abort MPI: {e}")
            import sys
            sys.exit(exit_code)

def find_subsplit_files(file):
    """
    Finds and returns a list of subsplit files (e.g., 'filename.0', 'filename.1', etc.).
    If no subsplits are found, it returns the original file.
    """
    base_path, file_name = os.path.split(file)
    files = []
    i = 0
    if os.path.exists(file):
        base_name = file_name.split('.')[0]
        while True:
            potential_file = os.path.join(base_path, f"{base_name}.{i}")
            if os.path.exists(potential_file):
                files.append(potential_file)
                i += 1
            else:
                break
        if not files:
            files.append(file)
    else:
        base_name = file_name
        while True:
            potential_file = os.path.join(base_path, f"{base_name}.{i}")
            if os.path.exists(potential_file):
                files.append(potential_file)
                i += 1
            else:
                break
        if not files:
            files.append(file)
    return files


def describe_window_action(win_params):
    if win_params:
        return f"applying window type={win_params['type']} args={win_params.get('len_args', {})}"
    return "no window, reusing base field"


def validate_sfc_compatibility(sfc_list, required_keys, logger=None, label="SFCField inputs"):
    filtered = [c for c in sfc_list if c is not None]
    if len(filtered) < 2:
        if filtered:
            return {key: filtered[0].sfc_info.get(key) for key in required_keys}
        return {}
    reference = filtered[0]
    mismatches = []
    for idx, current in enumerate(filtered[1:], start=2):
        for key in required_keys:
            ref_val = reference.sfc_info.get(key)
            cur_val = current.sfc_info.get(key)
            if ref_val != cur_val:
                mismatches.append((idx, key, ref_val, cur_val))
    if mismatches:
        mismatch_text = ", ".join(
            [f"vertex{idx}.{key}={cur_val} (reference={ref_val})" for idx, key, ref_val, cur_val in mismatches]
        )
        if logger is not None:
            logger.error(f"{label} require matching required parameters. Found mismatches: {mismatch_text}")
        safe_exit(1)
    return {key: reference.sfc_info.get(key) for key in required_keys}

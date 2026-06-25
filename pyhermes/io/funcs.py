import os
import pickle
import datetime

from pyhermes.param.logbase import setup_logger 
from pyhermes.utils.func_util import get_fname_info

def timenow():
    return datetime.datetime.now().strftime('%Y%m%d%H%M')

def check_fout(instance, fout_path, overwrite=False):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    if fout_path is None or fout_path == "":
        logger.info("No <fout_path> specified, skipping disk output.")
        return False
    ext_dict = {
        'SFCField' : 'pkl',
        'WindowFunc'  : 'pkl',
        'CountingData': 'pkl',
        'Corr2PCFData': 'pkl',
        'Corr3PCFData': 'pkl',
    }
    isFolder = False
    if fout_path.endswith('/') or fout_path.endswith('\\'):
        isFolder = True
    if os.path.exists(fout_path):
        if os.path.isdir(fout_path):
            isFolder = True
        elif os.path.isfile(fout_path):
            if overwrite:
                logger.warning(
                    f"Output file '{fout_path}' already exists and will be overwritten (overwrite=True)."
                )
                # keep fout_path unchanged
            else:
                logger.warning(f"Output file '{fout_path}' already exists! Generating a new file name.")
                base, ext = os.path.splitext(fout_path)
                counter = 1
                new_fout_path = f"{base}_{counter}{ext}"
                while os.path.exists(new_fout_path):
                    counter += 1
                    new_fout_path = f"{base}_{counter}{ext}"
                fout_path = new_fout_path
    if not isFolder:
        parent_dir = os.path.dirname(fout_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    else:
        if not os.path.exists(fout_path):
            os.makedirs(fout_path)
        base_name = instance.__class__.__name__
        ext = ext_dict[base_name]
        fout_path = os.path.join(fout_path, f"Output_{base_name}_{timenow()}.{ext}")
    return fout_path


### ↓ Back previous ↓ ###

def write_tristan_plk(f_out, data):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f'Writing pickle data to ---> {f_out} <---')
    _dir = os.path.dirname(f_out)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    with open(f_out, 'wb') as f:
        pickle.dump(data, f)

def load_tristan_plk(f_in):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f'Reading pickle data from ---> {f_in} <---')
    with open(f_in, 'rb') as f:
        data = pickle.load(f)
    return data

def load_whatever(f_in):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f'Reading pickle data from ---> {f_in} <---')
    pass

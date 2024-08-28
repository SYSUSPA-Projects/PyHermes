import os
import re
import sys
import pickle
import requests

import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

from .my_gd_io import read_my_gd2
from pyhermes.utils import func_util
from pyhermes.param.logbase import setup_logger 
from pyhermes.utils.func_util import get_fname_info



### ↓ Particle reading functions ↓ ###

def read_generic(f_in):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f'Reading paricle data from ---> {f_in} <---')
    data = np.fromfile(f_in, dtype=np.float32).reshape(-1,3)
    data_size = data.shape[0]
    return data, data_size

def read_gadget(f_in):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    logger.info(f'Reading paricle data from ---> {f_in} <---')
    _data = read_my_gd2(f_in)
    # Here, _data also contains velocity and mass info
    #  may be useful when weight is considered in future
    #                                   dingdluan 20240828
    data = _data['pos']
    data_size = data.shape[0]
    return data, data_size

def read_somethingelse(f_in):
    pass

### ↑ Particle reading functions ↑ ###

def read_particle_data(f_in, f_format):
    format_to_function = {
        "generic": read_generic,
        "gadget": read_gadget,
        "somethingelse": read_somethingelse,
    }
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    if f_format in format_to_function:
        logger.info(f"Selected input particle format: {f_format}")
        f_in = handle_PATHorURL(f_in)
        read_function = format_to_function[f_format]
        return read_function(f_in)
    else:
        supported_formats = ", ".join(format_to_function.keys())
        logger.error(f"Unsupported input particle format: {f_format}")
        logger.error(f"Supported formats: '{supported_formats}'")
        logger.error(f"Please see the document for details")
        func_util.safe_exit(1)

def handle_PATHorURL(f_in):
    _check_fpath = check_fin(f_in)
    if _check_fpath == '_url_':
        _url = f_in
        f_in = dl_rich_pbar(_url)
    return f_in

def check_fin(fin_path):
    url_pattern = re.compile(r'^(http|https)://', re.IGNORECASE)
    if url_pattern.match(fin_path):
        return '_url_'
    else:
        return

def dl_rich_pbar(url, output_path=None):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    # Get the file size from the response headers
    response = requests.head(url, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))
    if not output_path:
        content_disposition = response.headers.get('content-disposition')
    if content_disposition:
        output_path = content_disposition.split("filename=")[-1].strip('"')
    else:
        output_path = os.path.basename(url)
    if os.path.exists(output_path):
        logger.info(f"File '{output_path}' already exists. Skipping download.")
        return output_path
    # Use Progress to customize the progress bar style
    logger.info(f"Downloading file from '{url}'")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),  # Set progress bar width to 40 characters
        DownloadColumn(),  # Display the amount of data downloaded
        TransferSpeedColumn(),  # Display the download speed
        TimeRemainingColumn(),  # Display the estimated remaining time
        refresh_per_second=10  # Set refresh rate to 10 times per second
    ) as progress:
        task = progress.add_task(f"Downloading {os.path.basename(output_path)}...", total=total_size)
        # Download the file and update the progress bar
        with requests.get(url, stream=True) as r, open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                # Update the progress bar by advancing the number of downloaded bytes
                progress.update(task, advance=len(chunk))
                progress.refresh()
                sys.stdout.flush()
    return output_path

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

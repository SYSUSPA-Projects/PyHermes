import os
from datetime import datetime

import numpy as np

import pyhermes
from .convols import ColvolsData



class Corr2PCFData(ColvolsData):

    def _load_single(self, f_in):
        with open(f_in, 'r') as f:
            lines = f.readlines()
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() == "" or line.startswith("#"):
                continue
            if line.startswith('---'):  
                data_start = i + 1
                break
        _data = np.loadtxt(f_in, delimiter=",", skiprows=data_start+1)
        self.r = _data[:, 0]
        self.xi = _data[:, 1]

    def _save_single(self, f_out):
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        self.r = np.asarray(self.r, dtype=np.float64)
        self.xi = np.asarray(self.xi, dtype=np.float64)
        version = pyhermes.__version__
        current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        header = (
            f"# Corr_2PCF output from PyHermes v{version}, TIME: {current_time}\n"
            "# Parameters from input :\n"
            f"#  R1            = {self.task_params['R1']}\n"
            f"#  R2            = {self.task_params['R2']}\n"
            f"#  xi_num        = {int(self.task_params['xi_num'])}\n"
            f"#  threads       = {int(self.task_params['threads'])}\n"
            f"#  fout_dir      = {self.task_params['fout_dir']}\n"
            f"#  deltac_in_pat = {self.task_params['deltac_in_path']}\n"
            "# Parameters from DeltaC:\n"
            f"#  J             = {self.task_params['J']}\n"
            f"#  SimBoxL       = {self.task_params['SimBoxL']}\n"
            f"#  SampRate      = {int(self.task_params['SampRate'])}\n"
            f"#  bandwidth     = {self.task_params['bandwidth']}\n"
            f"#  fin_path      = {self.task_params['fin_path']}\n"
            f"#  fin_size      = {self.task_params['orgDsize']}\n"
            f"#  fin_format    = {self.task_params['fin_format']}\n"
            f"#  wavelet_mode  = {self.task_params['wavelet_mode']}\n"
            f"#  wavelet_level = {self.task_params['wavelet_level']}\n"
            f"#  Window_Info   = {self.task_params['window']}\n"
            "\n"
            "---------------------------\n"
            "r[h-1 Mpc]  , xi"
        )
        data_to_save = np.column_stack((self.r, self.xi))
        fmt = '%.6e, %.6e'
        delimiter = ",   " 
        np.savetxt(f_out, data_to_save, delimiter=delimiter, header=header, comments='', fmt=fmt)


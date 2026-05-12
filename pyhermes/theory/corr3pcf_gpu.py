import os
import pickle
import time

import numpy as np
import pywt
from numba import cuda

from pyhermes.io import ConvolsData, Corr2PCFData, Corr3PCFData, read_particle_data
from pyhermes.pipeline import TaskBase
from pyhermes.utils import func_util, math_util


class Corr_3PCF_GPU(TaskBase):
    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params_input(self):
        # Parameters from json or input
        self.fout_path = self.task_params["fout_path"]
        self.deltac_in_path = self.task_params["deltac_in_path"]
        self.corr2pcf_in_path = self.task_params["corr2pcf_in_path"]
        self.fin_path = self.task_params["fin"]["path"]
        self.fin_format = self.task_params["fin"]["format"]
        self.NStheta = int(self.task_params["NStheta"])
        self.R1 = self.task_params["R1"]
        self.R2 = self.task_params["R2"]
        self.rot_num = int(self.task_params["rot_num"])

    def format_params_deltac(self):
        # Parameters inherited from DeltaC
        self.J = self.task_params["J"]
        self.SampRate = int(self.task_params["SampRate"])
        self.SimBoxL = self.task_params["SimBoxL"]
        self.wavelet_mode = self.task_params["wavelet_mode"]
        self.wavelet_level = self.task_params["wavelet_level"]
        self.window_type = self.task_params["window"]["type"]
        self.window_args = {key: float(value) for key, value in self.task_params["window"].items() if key != "type"}
        self.bandwidth = self.task_params["bandwidth"]
        self.orgDsize = self.task_params["orgDsize"]
        self.L = 1 << self.J
        self.ScaleFactor = self.L / self.SimBoxL

    def run(self, deltac=None, corr2pcf=None):
        self.Gamma = math_util.cal_gamma(self.phi_data, self.PhiSupport, self.SampRate)
        self.work_dir = os.path.dirname(self.fin_path)
        self.all_l_result = []
        self.Gamma_gpu = cuda.to_device(self.Gamma)
        self.result_gpu = cuda.device_array((L, L, L), dtype=np.complex128)
        self.corr_3pcf.load_deltac(f_in=self.deltac_in_path, single=True)
        self.data_gpu = cuda.to_device(self.corr_3pcf.deltac)
        for l in range(8):
            l_result = []
            for m in range(l + 1):
                if m == 0:
                    with open(
                        self.work_dir
                        + "/R"
                        + str(self.R1)
                        + "/deltac_"
                        + str(self.L)
                        + "_005_r"
                        + str(self.Radius)
                        + "_R"
                        + str(self.R1)
                        + "_l"
                        + str(l)
                        + "_m0_pywt.pk",
                        "rb",
                    ) as f:
                        self.corr_3pcf.data_R1 = pickle.load(f)
                    with open(
                        work_dir
                        + "/R"
                        + str(self.R2)
                        + "/deltac_"
                        + str(self.L)
                        + "_005_r"
                        + str(self.Radius)
                        + "_R"
                        + str(self.R2)
                        + "_l"
                        + str(l)
                        + "_m0_pywt.pk",
                        "rb",
                    ) as f:
                        self.corr_3pcf.data_R2 = pickle.load(f)
                elif m > 0:
                    with open(
                        self.work_dir
                        + "/R"
                        + str(self.R1)
                        + "/deltac_"
                        + str(self.L)
                        + "_005_r"
                        + str(self.Radius)
                        + "_R"
                        + str(self.R1)
                        + "_l"
                        + str(l)
                        + "_m"
                        + str(m)
                        + "_pywt.pk",
                        "rb",
                    ) as f:
                        self.corr_3pcf.data_R1 = pickle.load(f)
                    with open(
                        self.work_dir
                        + "/R"
                        + str(self.R2)
                        + "/deltac_"
                        + str(self.L)
                        + "_005_r"
                        + str(self.Radius)
                        + "_R"
                        + str(self.R2)
                        + "_l"
                        + str(l)
                        + "_m_minus"
                        + str(m)
                        + "_pywt.pk",
                        "rb",
                    ) as f:
                        self.corr_3pcf.data_R2 = pickle.load(f)
                self.data_gpu = cuda.to_device(self.corr_3pcf.deltac)
                self.data_R1_gpu = cuda.to_device(self.corr_3pcf.data_R1)
                self.data_R2_gpu = cuda.to_device(self.corr_3pcf.data_R2)
                threads_per_block = (8, 8, 8)
                blocks_per_grid = (
                    (self.L + threads_per_block[0] - 1) // threads_per_block[0],
                    (self.L + threads_per_block[1] - 1) // threads_per_block[1],
                    (self.L + threads_per_block[2] - 1) // threads_per_block[2],
                )
                math_util.compute_3d_result_gpu[blocks_per_grid, threads_per_block](
                    self.data_gpu,
                    self.data_R1_gpu,
                    self.data_R2_gpu,
                    self.Gamma_gpu,
                    self.result_gpu,
                    self.L,
                    self.PhiSupport,
                )
                cuda.synchronize()
                self.result = self.result_gpu.copy_to_host()

                rho = self.orgDsize / self.L**2
                self.result_sum = np.sum(self.result) / rho**3
                print("l = :", str(l), " m = :", str(m), " Result sum: ", self.result_sum * 4 * np.pi)
                l_result.append(self.result_sum * 4 * np.pi)
            coeffs = math_util.cal_coefficients(l_result, l)
            sign = (-1) ** l
            real_part = coeffs.real * sign
            imag_part = coeffs.imag * sign
            print(f"l = {l}, real part = {real_part}, imag part = {imag_part}")
            self.all_l_result.append(real_part)
        return self.corr_3pcf

import time
import pickle

import numpy as np

from pyhermes.io import WindowFunc, ConvolsData, Corr3PCFMultipoleData
from pyhermes.utils import func_util, math_util
from pyhermes.pipeline import TaskBase


class Corr_3PCF_Multipole(TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params(self):
        self.convols_data_path = self.task_params["convols_data_path"]
        self.fout_path = self.task_params["fout_path"]

        win_params = self.task_params.get("window", None)
        win_params = win_params if (win_params and win_params.get("type")) else None

        for i in range(1, 4):
            win_params_i = self.task_params.get(f"window{i}", None)
            win_params_i = win_params_i if (win_params_i and win_params_i.get("type")) else None
            if (not win_params_i) and win_params:
                win_params_i = dict(win_params)
            setattr(self, f"win_params{i}", win_params_i)

        self.r1 = float(self.task_params["r1"])
        self.r2 = float(self.task_params["r2"])
        self.l_max = int(self.task_params["l_max"])
        self.gpu_device_id = int(self.task_params["gpu_device_id"])
        self.conv_batch_mode = self.task_params["conv_batch_mode"]
        self.cache_multipole_fields = bool(self.task_params["cache_multipole_fields"])
        self.cache_dir = self.task_params["cache_dir"]
        self.threads = int(self.task_params["threads"])

    def run(self, convols_data1=None, convols_data2=None, convols_data3=None, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            if rank == 0:
                t0 = time.perf_counter()

            self.format_params()
            if self.conv_batch_mode != "by_l":
                self.logger.error(f"Unsupported conv_batch_mode='{self.conv_batch_mode}'. Only 'by_l' is implemented.")
                func_util.safe_exit(1)

            self.corr3pcf_multipole_data = Corr3PCFMultipoleData()
            convols_info1_serialized = None
            convols_info2_serialized = None
            convols_info3_serialized = None

            if rank == 0:
                if self.convols_data_path:
                    self.convols_data = ConvolsData(data_path=self.convols_data_path)
                elif not (convols_data1 and convols_data2 and convols_data3):
                    self.logger.error(
                        "No input 'convols_data' provided and 'convols_data_path' is not set. "
                        "Please either pass convols_data1/2/3 or set 'convols_data_path'."
                    )
                    func_util.safe_exit(1)

                for i, cdata in zip([1, 2, 3], [convols_data1, convols_data2, convols_data3]):
                    if cdata is not None:
                        if isinstance(cdata, ConvolsData):
                            setattr(self, f"convols_data{i}", cdata)
                        else:
                            self.logger.error(f"convols_data{i} is not ConvolsData.")
                            func_util.safe_exit(1)
                    else:
                        _win_params = getattr(self, f"win_params{i}", None)
                        if _win_params:
                            _window = WindowFunc(_win_params, self.convols_data.convols_info)
                            _convols_data = self.convols_data @ _window
                        else:
                            _convols_data = self.convols_data._spawn_like()
                            _convols_data.epsilon = self.convols_data.epsilon
                            _convols_data.format_convols_params()
                        setattr(self, f"convols_data{i}", _convols_data)

                    setattr(self.corr3pcf_multipole_data, f"convols_info{i}", getattr(self, f"convols_data{i}").convols_info)

                self.corr3pcf_multipole_data.corr3pcf_multipole_info = dict(self.task_params)
                convols_info1_serialized = pickle.dumps(self.convols_data1.convols_info)
                convols_info2_serialized = pickle.dumps(self.convols_data2.convols_info)
                convols_info3_serialized = pickle.dumps(self.convols_data3.convols_info)

            convols_info1_serialized = comm.bcast(convols_info1_serialized, root=0)
            convols_info2_serialized = comm.bcast(convols_info2_serialized, root=0)
            convols_info3_serialized = comm.bcast(convols_info3_serialized, root=0)

            if rank == 0:
                local_convols = [self.convols_data1, self.convols_data2, self.convols_data3]
            else:
                local_convols = []
                for serialized in [convols_info1_serialized, convols_info2_serialized, convols_info3_serialized]:
                    cdata = ConvolsData()
                    cdata.convols_info = pickle.loads(serialized)
                    cdata.format_convols_params()
                    local_convols.append(cdata)

            if rank == 0:
                self.logger.info("Start to calculate 3PCF multipole ...")
                R = 1.0 / self.convols_data1.V
                deltaD1 = self.convols_data1 - R
                deltaD2 = self.convols_data2 - R
                deltaD3 = self.convols_data3 - R
                l_arr, zeta_l = math_util.calc_DDD_multipole(
                    deltaD1, deltaD2, deltaD3,
                    self.r1, self.r2, self.l_max,
                    gpu_device_id=self.gpu_device_id,
                    cache_multipole_fields=self.cache_multipole_fields,
                    cache_dir=self.cache_dir,
                    threads=self.threads,
                )
                self.corr3pcf_multipole_data.r1 = self.r1
                self.corr3pcf_multipole_data.r2 = self.r2
                self.corr3pcf_multipole_data.l = l_arr
                self.corr3pcf_multipole_data.zeta_l = zeta_l

                if self.fout_path:
                    self.corr3pcf_multipole_data.saveflag = True
                    self.corr3pcf_multipole_data.save_corr3pcf_multipole(self.fout_path, overwrite=overwrite)

            comm.Barrier()
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)

        if self.rank == 0:
            t1 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {t1 - t0:.4f} sec")

        return self.corr3pcf_multipole_data

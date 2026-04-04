import os
import pickle

import numpy as np

from .base import HermesData
from pyhermes.utils import func_util


class Corr3PCFMultipoleData(HermesData):
    def __init__(self, *args, threads=None, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.corr3pcf_multipole_info = {}
        self.r1 = None
        self.r2 = None
        self.l = None
        self.zeta_l = None
        super().__init__(*args, threads=threads, **kwargs)
        if data_path:
            self.corr3pcf_multipole_info["corr3pcf_multipole_data_path"] = data_path
            self.load_corr3pcf_multipole(data_path)

    def format_corr3pcf_multipole_params(self):
        for key, value in self.corr3pcf_multipole_info.items():
            setattr(self, key, value)

    def load_corr3pcf_multipole(self, f_in, single=True):
        self.load(f_in, read_3pcf_multipole=True, single=single)

    def save_corr3pcf_multipole(self, f_out, single=True, overwrite=False):
        self.save(f_out, save_3pcf_multipole=True, single=single, overwrite=overwrite)

    def _load_corr3pcf_multipole(self, f_in):
        with open(f_in, "rb") as f:
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            dataset = pickle.loads(serialized_data.tobytes())
            for key in ["r1", "r2", "l", "zeta_l"]:
                if key not in dataset:
                    self.logger.error(f"Failed to load the dataset. The file is missing the '{key}' key.")
                    func_util.safe_exit(1)
                setattr(self, key, dataset[key])
            for i in range(1, 4):
                _convols_info = dataset.get(f"convols_info{i}")
                if _convols_info:
                    setattr(self, f"convols_info{i}", _convols_info)
            _info = dataset.get("corr3pcf_multipole_info")
            if _info:
                self.corr3pcf_multipole_info.update(_info)
            self.format_corr3pcf_multipole_params()

    def _save_corr3pcf_multipole(self, f_out):
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        if not self.corr3pcf_multipole_info:
            self.logger.error('The dictionary "corr3pcf_multipole_info" is empty.')
            self.logger.error("Please calculate or load the dataset before saving.")
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        dataset = {
            "convols_info1": self.convols_info1,
            "convols_info2": self.convols_info2,
            "convols_info3": self.convols_info3,
            "corr3pcf_multipole_info": self.corr3pcf_multipole_info,
            "r1": self.r1,
            "r2": self.r2,
            "l": self.l,
            "zeta_l": self.zeta_l,
        }
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, "wb") as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))

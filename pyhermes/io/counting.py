import os

import numpy as np

from .convols import ConvolsData



class CountingData(ConvolsData):

    def _load_single(self, f_in):
        with open(f_in, 'rb') as f:
            self.data_all = np.load(f)

    def _save_single(self, f_out):
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        with open(f_out, 'wb') as f:
            np.save(f, self.data)
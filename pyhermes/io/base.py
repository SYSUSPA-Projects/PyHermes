import pickle

import numpy as np

from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.param.logbase import setup_logger 
from pyhermes.io import handle_PATHorURL, check_fout



class HermesData(object):

    def __init__(self, *args, threads=1, **kwargs):
        self.comm                = MPI.COMM_WORLD
        self.rank                = self.comm.Get_rank()
        self.logger              = setup_logger(__name__, self.__class__.__name__)
        self.threads             = max(1, int(threads))
        self.data                = None
        self.deltac              = None
        self.dict_inht_vonDeltac = {}
        self.r                   = None
        self.xi                  = None
        self.theta               = None
        self.q                   = None
        self.saveflag            = False
        self.task_params         = None
        # Set numba threads
        math_util.configure(threads=self.threads)

    def load(self, f_in, read_deltac=False, read_2pcf=False, single=True):
        try:
            if single:
                if self.rank == 0:
                    f_in = handle_PATHorURL(f_in)
                    if read_deltac:
                        extra_str = 'DeltaC '
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_deltac(f_in)
                        self.logger.info(f'DeltaC: Shape{self.deltac.shape}, Min = {self.deltac.min():.4f}, Max = {self.deltac.max():.4f}, Mean = {self.deltac.mean():.4f}')
                    elif read_2pcf:
                        extra_str = '2PCF '
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_corr2pcf(f_in)
                    else:
                        extra_str = ''
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_single(f_in)
            else:
                # TODO, MPI multi read
                pass
        except Exception as e:
            self.logger.error(f"An error occurred while loading the file '{f_in}': {e}")
            func_util.safe_exit(1)

    def save(self, f_out, single=True):
        if single:
            if self.rank == 0:
                f_out = check_fout(self, f_out)
                if f_out:
                    self.logger.info(f'Writing data to ---> {f_out} <---')
                    self._save_single(f_out)
        else:
            # TODO, MPI multi save
            pass

    def load_deltac(self, f_in, single=True):
        self.load(f_in, read_deltac=True, single=single)

    def _load_deltac(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'data' key is present in the dataset
            if 'deltac' not in dataset:
                self.logger.error(f"Failed to load the dataset. The file is missing the 'data' key.")
                func_util.safe_exit(1)
            # Assign the dictionary from the file to self.dict_inht_vonDeltac
            self.dict_inht_vonDeltac = {key: value for key, value in dataset.items() if key != 'deltac'}
            self.deltac = dataset['deltac']

    def _load_single(self, f_in):
        pass

    def _save_single(self, f_out):
        pass

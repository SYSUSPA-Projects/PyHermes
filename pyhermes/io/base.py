from pyhermes.utils import func_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.io import handle_PATHorURL
from pyhermes.param.logbase import setup_logger 



class HermesData(object):

    def __init__(self):
        self.comm                 = MPI.COMM_WORLD
        self.rank                 = self.comm.Get_rank()
        self.logger               = setup_logger(__name__, self.__class__.__name__)
        self.data                 = None
        self.deltac               = None
        self.dict_inht_vonDeltac  = {}
        self.xi                   = None
        self.r                    = None
        self.saveflag             = False
        self.task_params          = None

    def load(self, f_in, read_deltac=False, single=True):
        try:
            if single:
                if self.rank == 0:
                    f_in = handle_PATHorURL(f_in)
                    if read_deltac:
                        deltac_str = 'DeltaC '
                        self.logger.info(f'Reading {deltac_str}data from ---> {f_in} <---')
                        self._load_deltac(f_in)
                        self.logger.info(f'DeltaC: Shape{self.deltac.shape}, Min = {self.deltac.min():.4f}, Max = {self.deltac.max():.4f}, Mean = {self.deltac.mean():.4f}')
                    else:
                        deltac_str = ''
                        self.logger.info(f'Reading {deltac_str}data from ---> {f_in} <---')
                        # self.logger.info(f'Data: Shape{self.data.shape}, Min = {self.data.min():.4f}, Max = {self.data.max():.4f}, Mean = {self.data.mean():.4f}')
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
                if self.data is None and not self.saveflag:  
                    self.logger.error('No data available to save!')
                    self.logger.error('Please ensure that the data has been loaded or calculated before attempting to save.')
                    func_util.safe_exit(1)
                else:
                    self.logger.info(f'Writing data to ---> {f_out} <---')
                    self._save_single(f_out)
        else:
            # TODO, MPI multi save
            pass

    def _load_single(self, f_in):
        # Here we need to return the loaded data, _data
        pass

    def _save_single(self, f_out):
        pass

from pyhermes.utils import func_util
from pyhermes.utils.runtime import configure
from pyhermes.utils.mpi_util import MPI
from pyhermes.param.logbase import setup_logger 
from pyhermes.io import check_fout



class HermesData(object):

    def __init__(self, *args, threads=None, **kwargs):
        self.comm                = MPI.COMM_WORLD
        self.rank                = self.comm.Get_rank()
        self.logger              = setup_logger(__name__, self.__class__.__name__)
        self.saveflag            = False
        self.task_params         = None
        # Set numba threads
        if threads:
            self.threads = max(1, int(threads))
            configure(threads=self.threads)
        else:
            self.threads = None

    def load(
        self, f_in, read_sfc_field=False, read_counting=False, read_2pcf=False,
        read_3pcf=False, read_3pcf_multipole=False, single=True
    ):
        try:
            if single:
                if self.rank == 0:
                    if read_sfc_field:
                        extra_str = 'SFCField '
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_sfc_field(f_in)
                        self.logger.info(f'epsilon: Shape{self.epsilon.shape}, Min = {self.epsilon.min():.4g}, Max = {self.epsilon.max():.4g}, Mean = {self.epsilon.mean():.4g}, Sum = {self.epsilon.sum():.4g}')
                    elif read_counting:
                        extra_str = 'Counting '
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_counting(f_in)
                        self.logger.info(f'nx: Shape{self.nx.shape}, Min = {self.nx.min():.4g}, Max = {self.nx.max():.4g}, Mean = {self.nx.mean():.4g}')
                    elif read_2pcf:
                        extra_str = '2PCF '
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_corr2pcf(f_in)
                        for name in getattr(self, "sampling_names", tuple(getattr(self, "sampling", {}).keys())):
                            values = self.sampling[name]
                            self.logger.info(
                                f'{name}: Num = {values.shape[0]}, Min = {values.min():.4g}, Max = {values.max():.4g}'
                            )
                        products = [key for key in ['dd', 'dr', 'rd', 'delta_dd', 'rr', 'xi'] if getattr(self, key) is not None]
                        self.logger.info(f'Products loaded: {products}')
                    elif read_3pcf:
                        extra_str = '3PCF '
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_corr3pcf(f_in)
                        self.logger.info(
                            f"theta: Num = {self.theta.shape[0]}, Min = {self.theta.min():.4g}, Max = {self.theta.max():.4g}"
                        )
                        self.logger.info(
                            f"mu: Num = {self.mu.shape[0]}, Min = {self.mu.min():.4g}, Max = {self.mu.max():.4g}"
                        )
                        products = [
                            key for key in ['ddd', 'rrr', 'd_delta_dd', 'r_delta_dd', 'delta_ddd', 'xi12', 'xi13', 'xi23', 'zeta', 'zeta_H', 'Q']
                            if getattr(self, key) is not None
                        ]
                        self.logger.info(f'Products loaded: {products}')
                    elif read_3pcf_multipole:
                        extra_str = '3PCF Multipole '
                        self.logger.info(f'Reading {extra_str}data from ---> {f_in} <---')
                        self._load_corr3pcf_multipole(f_in)
                        self.logger.info(
                            f"l: Num = {self.l.shape[0]}, Min = {self.l.min()}, Max = {self.l.max()}"
                        )
                        products = [
                            key for key in ['ddd_l', 'rrr_l', 'delta_ddd_l', 'zeta_l', 'zeta_condition']
                            if getattr(self, key) is not None
                        ]
                        self.logger.info(f'Products loaded: {products}')
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

    def save(
        self, f_out, save_sfc_field=False, save_counting=False, save_2pcf=False,
        save_3pcf=False, save_3pcf_multipole=False, single=True, overwrite=False
    ):
        if single:
            if self.rank == 0:
                f_out = check_fout(self, f_out, overwrite)
                if f_out:
                    if save_sfc_field:
                        extra_str = 'SFCField '
                        self.logger.info(f'Writing {extra_str}data to ---> {f_out} <---')
                        self._save_sfc_field(f_out)
                    elif save_counting:
                        extra_str = 'Counting '
                        self.logger.info(f'Writing {extra_str}data to ---> {f_out} <---')
                        self._save_counting(f_out)
                    elif save_2pcf:
                        extra_str = '2PCF '
                        self.logger.info(f'Writing {extra_str}data to ---> {f_out} <---')
                        self._save_corr2pcf(f_out)
                    elif save_3pcf:
                        extra_str = '3PCF '
                        self.logger.info(f'Writing {extra_str}data to ---> {f_out} <---')
                        self._save_corr3pcf(f_out)
                    elif save_3pcf_multipole:
                        extra_str = '3PCF Multipole '
                        self.logger.info(f'Writing {extra_str}data to ---> {f_out} <---')
                        self._save_corr3pcf_multipole(f_out)
                    else:
                        extra_str = ''
                        self.logger.info(f'Writing data to ---> {f_out} <---')
                        self._save_single(f_out)
        else:
            # TODO, MPI multi save
            pass

    def _load_single(self, f_in):
        pass

    def _save_single(self, f_out):
        pass

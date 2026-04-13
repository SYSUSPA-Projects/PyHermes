import pickle

from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.param.logbase import setup_logger
from pyhermes.param.parambase import ParamBase



class TaskBase(object):

    def __init__(self, param_task):
        # Set MPI related
        self.comm       = MPI.COMM_WORLD
        self.rank       = self.comm.Get_rank()
        self.size       = self.comm.Get_size()
        self.comm_local = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.size_local = self.comm_local.Get_size()
        # Set new style logging, added by dingding, 20231113
        self.logger = setup_logger(__name__, self.__class__.__name__)
        params_serialized = None
        # Here we handle the input & default params in rank 0
        if self.rank == 0:
            # Create an instance of ParamBase
            param_base_instance = ParamBase()
            # Read task default paramters
            param_base_instance.read_default(self.__class__)
            _params_task_default=param_base_instance.default_params
            _params_task_user = param_task
            _task_name_user = list(_params_task_user.keys())
            _task_name = self.task_name
            if _task_name not in _task_name_user:
                self.logger.error(f"Mismatch of task name, expected <{_task_name}>, but it was not found in the provided task(s) <{_task_name_user}>")
                func_util.safe_exit(1)
            # Update params to users input
            param_base_instance.recursive_update(
                default_dict=_params_task_default,
                new_dict=_params_task_user,
                section=_task_name
                )
            _params_task_user[_task_name]=_params_task_default[_task_name]
            _task_params = _params_task_user[_task_name]
            params_serialized = pickle.dumps(_task_params)
        # Setup parameters
        params_serialized = self.comm.bcast(params_serialized, root=0)
        self.task_params = pickle.loads(params_serialized)
        self.comm.Barrier()
        self.threads = self.task_params.get('threads', None)

    def sync_runtime_options(self, context=None, blank_line=False):
        threads = self.task_params.get('threads', self.threads)
        self.threads = threads
        if threads is not None:
            self.threads = max(1, int(threads))
            self.task_params['threads'] = self.threads
            math_util.configure(threads=self.threads)
        if self.rank == 0:
            if blank_line:
                print("")
            prefix = f"{context}: " if context else ""
            if self.threads is None:
                self.logger.info(f"{prefix}running on {self.size} MPI ranks")
            else:
                self.logger.info(
                    f"{prefix}running on {self.size} MPI ranks with {self.threads} threads per rank"
                )

    def run(self):
        try:
            pass
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)

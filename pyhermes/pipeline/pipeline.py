import pickle

from pyhermes.utils import func_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.param.logbase import setup_logger
from pyhermes.param.parambase import JsonBase



class TaskBase(object):

    def __init__(self, param_task):
        # Set MPI related
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        # Set new style logging, added by dingding, 20231113
        self.logger = setup_logger(__name__, self.__class__.__name__)
        params_serialized = None
        if self.rank == 0:
            # Create an instance of JsonBase
            json_base_instance = JsonBase()
            # Read task default paramters
            json_base_instance.read_default(self.__class__)
            _params_task_default=json_base_instance.default_params
            _params_task_user = param_task
            _task_name_user = list(_params_task_user.keys())
            _task_name_user_count = len(_task_name_user)
            # Now we only support single task pipeline :)
            if _task_name_user_count > 1:
                self.logger.error(f"The program only support 1 task name, but you have provided {_task_name_user_count}.")
                func_util.safe_exit(1)
            elif _task_name_user_count == 0:
                self.logger.error("Your input does not contain any task name.")
                func_util.safe_exit(1)
            _task_name_user = _task_name_user[0]
            _task_name = self.task_name
            if _task_name != _task_name_user:
                self.logger.error(f"Mismatch of task name, expected '{_task_name}', but received '{_task_name_user}'.")
                func_util.safe_exit(1)
            json_base_instance.recursive_update(
                default_dict=_params_task_default,
                new_dict=_params_task_user,
                section=_task_name
                )
            _params_task_user[_task_name]=_params_task_default[_task_name]
            _task_params = _params_task_user[_task_name]
            params_serialized = pickle.dumps(_task_params)
            print("")
            self.logger.info(f"The task will run on {self.size} MPI ranks")
        # Setup parameters
        params_serialized = self.comm.bcast(params_serialized, root=0)
        self.task_params = pickle.loads(params_serialized)
        self.comm.Barrier()

    def run(self):
        try:
            pass
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)

    def __del__(self):
        if self.rank == 0:
            self.logger.info('Bye.')
        self.comm.Barrier()


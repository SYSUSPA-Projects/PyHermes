from pyhermes.param.logbase import setup_logger



try:
    from mpi4py import MPI as MPI
except ImportError:
    # Fake MPI wrapper ↓ is used when mpi4py is not available
    # This implementation mimics basic MPI functionality and ensures compatibility
    # with mpi4py-like code. Users can write code that works seamlessly both with
    # and without mpi4py, with reduced functionality when mpi4py is not installed.
    class MPI:

        def __init__(self):
            self.logger = setup_logger(__name__, self.__class__.__name__)
            self.logger.warning("Package <mpi4py> not installed, using single-core fake MPI wrapper. For parallel functionality in pyhermes, please install <mpi4py>. Visit https://pypi.org/project/mpi4py/ for installation.")

        # Mimic MPI.COMM_WORLD by returning self
        @property
        def COMM_WORLD(self):
            return self

        def Get_size(self):
            return 1  # Single-core environment

        def Get_rank(self):
            return 0  # Single process

        def send(self, data, dest, tag=0):
            pass  # No-op for single-core

        def isend(self, data, dest, tag=0):
            pass  # No-op for single-core

        def recv(self, buf=None, source=0, tag=0, status=None):
            return buf  # Return the buffer unchanged

        def irecv(self, buf=None, source=0, tag=0, status=None):
            return buf  # Return the buffer unchanged

        def Barrier(self):
            pass  # No-op for single-core

        def bcast(self, data, root=0):
            return data  # No broadcast needed in single-core

        def reduce(self, sendbuf, recvbuf=None, op=None, root=0):
            return sendbuf  # No reduction needed in single-core

        def gather(self, sendbuf, recvbuf=None, root=0):
            if self.Get_rank() == root:
                return [sendbuf]  # Gather into a list
            else:
                return None  # No other ranks
        
        def SUM(self):
            return None

        # Ensure compatibility with mpi4py method naming conventions
        Send   = send
        Recv   = recv
        Bcast  = bcast
        Gather = gather

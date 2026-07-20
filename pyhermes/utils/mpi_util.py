import time

import numpy as np

from pyhermes.param.logbase import setup_logger


_FAKE_SUM = object()
_FAKE_MAX = object()
_FAKE_UNDEFINED = object()
_FAKE_COMM_TYPE_SHARED = object()


class FakeNullComm:

    def Free(self):
        pass


_FAKE_COMM_NULL = FakeNullComm()


try:
    from mpi4py import MPI as MPI
except ModuleNotFoundError as exc:
    if exc.name != "mpi4py":
        raise

    # The single-process fallback is used when mpi4py is not available.
    # This implementation mimics basic MPI functionality and ensures compatibility
    # with mpi4py-like code. Users can write code that works seamlessly both with
    # and without mpi4py, with reduced functionality when mpi4py is not installed.
    class FakeRequest:

        def __init__(self, comm, source, tag):
            self.comm = comm
            self.source = source
            self.tag = tag
            self.completed = False
            self.start_time = time.time()

        def test(self):
            if not self.completed and (time.time() - self.start_time) > 0.1:
                data = self.comm.buffers.get((self.source, self.tag), None)
                if data is not None:
                    self.completed = True
                    return (True, data)
            return (False, None)

        def wait(self):
            while not self.completed:
                self.completed = True
                return self.comm.buffers.get((self.source, self.tag), None)

        Test = test
        Wait = wait


    class FakeMPI:

        def __init__(self):
            self.logger = setup_logger(__name__, self.__class__.__name__)
            self.logger.warning("Package <mpi4py> not installed, using single-core fake MPI wrapper.")
            self.logger.warning("For parallel functionality in pyhermes, please install <mpi4py>.")
            self.logger.warning("Visit https://pypi.org/project/mpi4py/ for installation.")
            self.buffers = {}

        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def send(self, data, dest, tag=0):
            self.buffers[(dest, tag)] = data

        def isend(self, data, dest, tag=0):
            self.buffers[(dest, tag)] = data
            return FakeRequest(self, dest, tag)

        def recv(self, buf=None, source=0, tag=0, status=None):
            return self.buffers.pop((source, tag), buf)

        def irecv(self, buf=None, source=0, tag=0):
            return FakeRequest(self, source, tag)

        def Barrier(self):
            pass

        def bcast(self, data, root=0):
            return data

        def allgather(self, data):
            return [data]

        def reduce(self, sendbuf, recvbuf=None, op=None, root=0):
            return sendbuf

        def allreduce(self, sendbuf, op=None):
            return sendbuf

        @staticmethod
        def _buffer(value):
            if isinstance(value, (list, tuple)):
                return value[0]
            return value

        def Allreduce(self, sendbuf, recvbuf, op=None):
            source = np.asarray(self._buffer(sendbuf))
            target = np.asarray(self._buffer(recvbuf))
            np.copyto(target, source)

        def gather(self, sendbuf, recvbuf=None, root=0):
            if self.Get_rank() == root:
                if recvbuf is not None:
                    target = self._buffer(recvbuf)
                    if isinstance(target, np.ndarray):
                        target[...] = sendbuf
                    else:
                        target[:] = [sendbuf]
                return [sendbuf]
            return None

        def scatter(self, data, root=0):
            if self.Get_rank() == root:
                return data[0]
            return None

        def Scatterv(self, sendbuf, recvbuf, root=0):
            source = np.asarray(self._buffer(sendbuf)).reshape(-1)
            target = np.asarray(self._buffer(recvbuf))
            target[...] = source[:target.size].reshape(target.shape)

        def Bcast(self, buf, root=0):
            return buf

        def Split_type(self, split_type, key=0, info=None):
            return self

        def Split(self, color, key=0):
            if color is _FAKE_UNDEFINED:
                return _FAKE_COMM_NULL
            return self

        def Iprobe(self, source=0, tag=0, status=None):
            return (source, tag) in self.buffers

        def Abort(self, errorcode=1):
            raise SystemExit(errorcode)

        def Free(self):
            pass

        Send = send
        Recv = recv
        Gather = gather
        Scatter = scatter


    class MPI:

        COMM_NULL = _FAKE_COMM_NULL
        COMM_TYPE_SHARED = _FAKE_COMM_TYPE_SHARED
        COMM_WORLD = FakeMPI()
        COMPLEX = np.complex64
        COMPLEX16 = np.complex128
        DOUBLE = np.float64
        MAX = _FAKE_MAX
        SUM = _FAKE_SUM
        UNDEFINED = _FAKE_UNDEFINED

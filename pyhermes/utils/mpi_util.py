import time

import numpy as np

from pyhermes.param.logbase import setup_logger



try:
    from mpi4py import MPI as MPI
except ImportError:
    # Fake MPI wrapper ↓ is used when mpi4py is not available
    # This implementation mimics basic MPI functionality and ensures compatibility
    #  with mpi4py-like code. Users can write code that works seamlessly both with
    #  and without mpi4py, with reduced functionality when mpi4py is not installed.
    # Any question, feel free to get in touch with us :)
    #                                                   dingdluan@gmail, 2024.8.26
    class FakeRequest:

        def __init__(self, comm, source, tag):
            self.comm       = comm         # Reference to the FakeComm object
            self.source     = source
            self.tag        = tag
            self.completed  = False
            self.start_time = time.time()

        def test(self):
            # Simulate non-blocking behavior: assume data is available after a delay
            if not self.completed and (time.time() - self.start_time) > 0.1:  # Simulate delay
                data = self.comm.buffers.get((self.source, self.tag), None)  # Get the latest data
                if data is not None:
                    self.completed = True
                    return (True, data)
            return (False, None)

        def wait(self):
            # Block until the request is completed
            while not self.completed:
                self.completed = True
                return self.comm.buffers.get((self.source, self.tag), None)


    class FakeMPI:

        def __init__(self):
            self.logger = setup_logger(__name__, self.__class__.__name__)
            self.logger.warning("Package <mpi4py> not installed, using single-core fake MPI wrapper.")
            self.logger.warning("For parallel functionality in pyhermes, please install <mpi4py>.")
            self.logger.warning("Visit https://pypi.org/project/mpi4py/ for installation.")
            self.buffers = {}

        def Get_size(self):
            return 1  # Single-core environment

        def Get_rank(self):
            return 0  # Single process

        def send(self, data, dest, tag=0):
            # pass  # No-op for single-core
            self.buffers[(dest, tag)] = data

        def isend(self, data, dest, tag=0):
            # Store the data in the buffer for the destination rank
            self.buffers[(dest, tag)] = data
            #print(f"Debug: Stored {data} in buffer with key {(dest, tag)}")
            return FakeRequest(self, dest, tag)

        def recv(self, buf=None, source=0, tag=0, status=None):
            # return buf  # Return the buffer unchanged
            return self.buffers.pop((source, tag), buf)

        def irecv(self, buf=None, source=0, tag=0):
            # Return a FakeRequest that dynamically checks the buffer
            #print(f"Debug: irecv called for source={source}, tag={tag}")
            return FakeRequest(self, source, tag)

        def Barrier(self):
            pass  # No-op for single-core

        def bcast(self, data, root=0):
            return data  # No broadcast needed in single-core

        def reduce(self, sendbuf, recvbuf=None, op=None, root=0):
            return sendbuf  # No reduction needed in single-core
            
        def gather(self, sendbuf, recvbuf=None, root=0):
            # In a single-core environment, gather just assigns the sendbuf to the recvbuf if root
            if self.Get_rank() == root:
                if recvbuf is not None:
                    if isinstance(recvbuf, np.ndarray):  # If recvbuf is a NumPy array
                        recvbuf[:] = sendbuf  # Assign sendbuf directly to the first element
                    else:
                        recvbuf[:] = [sendbuf]  # If it's a list, wrap sendbuf in a list and assign
                return [sendbuf]  # Return gathered data as a list
            else:
                return None  # Other ranks would normally send their part
            
        def scatter(self, data, root=0):
            if self.Get_rank() == root:
                return data[0]  # Single process, return the first element of the data
            else:
                return None  # Other ranks would normally receive their portion of the data
        
        # Ensure compatibility with mpi4py method naming conventions
        Send    = send
        Recv    = recv
        Bcast   = bcast
        Gather  = gather
        Scatter = scatter


    class MPI:

        COMM_WORLD = FakeMPI()

        def SUM(self):
            return None

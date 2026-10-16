"""Bounded MPI progress messages with explicit request/buffer lifetimes."""

import numpy as np


class TaskProgress:
    """Track coarse progress and consume exactly the posted messages per run.

    Lowercase mpi4py isend owns a temporary pickle buffer. Keep every request
    alive until completion, and do not leave receives/messages for the next run.
    Each rank posts about ten small messages, independent of the sample count.
    """

    def __init__(self, comm, local_total):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.interval = max(1, local_total // 10)
        self.pending = []
        counts = comm.gather(local_total // self.interval, root=0)
        if self.rank == 0:
            self.expected = counts
            self.received = np.zeros(len(counts), dtype=int)
            self.completed = np.zeros(len(counts), dtype=int)
            self.requests = [None] + [
                comm.irecv(source=rank, tag=rank) if counts[rank] else None
                for rank in range(1, len(counts))
            ]

    def _accept(self, rank, value):
        self.completed[rank] = value
        self.received[rank] += 1
        self.requests[rank] = (
            self.comm.irecv(source=rank, tag=rank)
            if self.received[rank] < self.expected[rank]
            else None
        )

    def update(self, completed):
        if self.rank != 0:
            if completed % self.interval == 0:
                self.pending.append(self.comm.isend(completed, dest=0, tag=self.rank))
            return None
        self.completed[0] = completed
        for rank, request in enumerate(self.requests[1:], start=1):
            if request is not None:
                ready, value = request.test()
                if ready:
                    self._accept(rank, value)
        return int(np.sum(self.completed))

    def finish(self):
        if self.rank == 0:
            for rank in range(1, len(self.requests)):
                while self.requests[rank] is not None:
                    self._accept(rank, self.requests[rank].wait())
        else:
            for request in self.pending:
                request.wait()
        self.pending.clear()
        self.comm.Barrier()

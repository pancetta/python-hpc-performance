import time
import numpy as np
from mpi4py import MPI

from .benchmarks import Benchmarks


class MPI_Broadcast(Benchmarks):

    def __init__(self, params):
        super(MPI_Broadcast, self).__init__(name='mpi_broadcast', params=params)

    def run(self):
        comm = MPI.COMM_WORLD

        if comm.rank == 0:
            A = np.arange(self.params.n, dtype=self.params.dtype)  # rank 0 has proper data
        else:
            A = np.empty(self.params.n, dtype=self.params.dtype)  # all other just an empty array

        if self.params.dtype == 'float64':
            mpi_dtype = MPI.DOUBLE
        else:
            raise NotImplementedError()

        comm.Barrier()

        t0 = time.time()
        comm.Bcast([A, mpi_dtype])
        t1 = time.time()

        return t1 - t0


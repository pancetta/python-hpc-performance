import time
import numpy as np
from mpi4py import MPI

from .benchmarks import Benchmarks


class MPI_Broadcast(Benchmarks):

    def __init__(self, comm, params):

        if comm is None:
            raise ValueError('No valid MPI communicator given')

        super(MPI_Broadcast, self).__init__(name='mpi_broadcast', comm=comm, params=params)

        if comm.Get_rank() == 0:
            self.A = np.arange(self.params.n, dtype=self.params.dtype)  # rank 0 has proper data
        else:
            self.A = np.empty(self.params.n, dtype=self.params.dtype)  # all other just an empty array

        if self.params.dtype == 'float64':
            self.mpi_dtype = MPI.DOUBLE
        else:
            raise NotImplementedError()

    def reset(self):
        self.comm.Barrier()

    def run(self):

        t0 = time.time()
        self.comm.Bcast([self.A, self.mpi_dtype])
        t1 = time.time()

        return t1 - t0


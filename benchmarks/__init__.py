from tools.registry import register

from .matmul import MatMul
from.mpi_broadcast import MPI_Broadcast

register(cls=MatMul,
         bench_type={'partype': 'multithreaded', 'min_procs': 1, 'max_procs': None},
         bench_params={'n': [1000, 2000], 'm': [1000, 2000], 'dtype': ['float64', 'float32']}
         )

register(cls=MPI_Broadcast,
         bench_type={'partype': 'mpi', 'min_procs': 1, 'max_procs': None},
         bench_params={'n': [1000, 10000], 'dtype': ['float64']}
         )


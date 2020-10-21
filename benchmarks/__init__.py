from tools.registry import register

from .optimize import Optimize
from .matmul import MatMul
from .matmul import MatMulSparse
from .mpi_broadcast import MPI_Broadcast
from .imports import Import


register(cls=Optimize,
         bench_type={'name': 'optimize', 'partype': 'sequential', 'min_procs': 1, 'max_procs': None},
         bench_params={'n': [2, 5, 10], 'method': ['Nelder-Mead', 'Powell']}
         )

register(cls=MatMul,
         bench_type={'name': 'matmul', 'partype': 'multithreaded', 'min_procs': 1, 'max_procs': None},
         bench_params={'n': [1000, 2000], 'm': [1000, 2000], 'dtype': ['float64', 'float32']}
         )

register(cls=MatMulSparse,
         bench_type={'name': 'matmul_sp', 'partype': 'sequential', 'min_procs': 1, 'max_procs': None},
         bench_params={'n': [1000, 2000], 'm': [1000, 2000], 'density': [0.01, 0.05]}
         )

register(cls=MPI_Broadcast,
         bench_type={'name': 'mpi_broadcast', 'partype': 'mpi', 'min_procs': 1, 'max_procs': None},
         bench_params={'n': [10000, 100000], 'dtype': ['float64']}
         )

register(cls=Import,
         bench_type={'name': 'import', 'partype': 'sequential', 'min_procs': 1, 'max_procs': None},
         bench_params={'modules': ['numpy, scipy, mpi4py', 'numpy', 'scipy', 'mpi4py']}
         )


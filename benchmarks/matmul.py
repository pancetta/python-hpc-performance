import time
import numpy as np

from .benchmarks import Benchmarks


class MatMul(Benchmarks):

    def __init__(self, comm, params):
        super(MatMul, self).__init__(name='matmul', comm=comm, params=params)
        np.random.seed(0)
        self.matA = np.random.rand(self.params.n, self.params.m).astype(self.params.dtype)
        self.matB = np.random.rand(self.params.m, self.params.n).astype(self.params.dtype)

    def run(self):

        t0 = time.time()
        self.matA @ self.matB
        t1 = time.time()

        return t1 - t0

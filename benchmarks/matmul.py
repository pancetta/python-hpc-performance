import time
import numpy as np

from .benchmarks import Benchmarks


class MatMul(Benchmarks):

    def __init__(self, params):
        super(MatMul, self).__init__(name='matmul', params=params)

    @staticmethod
    def _matmul(matA, matB):
        matA @ matB

    def run(self):
        np.random.seed(0)
        matA = np.random.rand(self.params.n, self.params.m).astype(self.params.dtype)
        matB = np.random.rand(self.params.m, self.params.n).astype(self.params.dtype)

        t0 = time.time()
        self._matmul(matA, matB)
        t1 = time.time()

        return t1 - t0

import time
from numba import jit
import numpy as np
import random

from .benchmarks import Benchmarks


class MonteCarlo(Benchmarks):
    # Idea taken directly from https://numba.pydata.org/

    def __init__(self, name, params, num_threads, comm):
        super(MonteCarlo, self).__init__(name=name, params=params, num_threads=num_threads, comm=comm)
        np.random.seed(0)

        if self.params.type == 'numba':
            self.do_it = self.monte_carlo_numba
        elif self.params.type == 'naive':
            self.do_it = self.monte_carlo_naive
        elif self.params.type == 'numpy':
            self.do_it = self.monte_carlo_numpy
        else:
            raise NotImplementedError()

        _ = self.do_it(1)


    @staticmethod
    @jit(nopython=True)
    def monte_carlo_numba(nsamples):
        acc = 0
        for i in range(nsamples):
            x = random.random()
            y = random.random()
            if (x ** 2 + y ** 2) < 1.0:
                acc += 1
        return 4.0 * acc / nsamples

    @staticmethod
    def monte_carlo_naive(nsamples):
        acc = 0
        for i in range(nsamples):
            x = random.random()
            y = random.random()
            if (x ** 2 + y ** 2) < 1.0:
                acc += 1
        return 4.0 * acc / nsamples

    @staticmethod
    def monte_carlo_numpy(nsamples):
        x = np.random.rand(nsamples)
        y = np.random.rand(nsamples)
        r = x ** 2 + y ** 2
        return 4.0 * np.sum([r < 1.0]) / nsamples

    def run(self):

        t0 = time.time()
        self.do_it(self.params.nsamples)
        t1 = time.time()

        return t1 - t0




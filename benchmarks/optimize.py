import time
import numpy as np
import scipy.optimize as so

from .benchmarks import Benchmarks


class Optimize(Benchmarks):

    def __init__(self, name, params, comm):
        super(Optimize, self).__init__(name=name, params=params, comm=comm)
        np.random.seed(0)
        self.x0 = np.random.rand(self.params.n)

    def run(self):

        t0 = time.time()
        so.minimize(so.rosen, self.x0, method=self.params.method, tol=1E-10, options=dict(maxiter=100000))
        t1 = time.time()

        return t1 - t0




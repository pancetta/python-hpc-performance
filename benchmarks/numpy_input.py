import time
import numpy as np
from .benchmarks import Benchmarks


class NumpyInput(Benchmarks):

    def __init__(self, name, params, comm):
        super(NumpyInput, self).__init__(name=name, params=params, comm=comm)
        np.random.seed(0)
        x = np.linspace(0, 1, self.params.n3d)
        y = np.linspace(0, 1, self.params.n3d)
        X, Y = np.meshgrid(x, y)

        data = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

        if self.params.compressed:
            self.file = 'out.npz'
            np.savez_compressed(file=self.file, arr=data)
        else:
            self.file = 'out.npy'
            np.save(file=self.file, arr=data)

    def run(self):

        t0 = time.time()
        _ = np.load(file=self.file)
        t1 = time.time()

        return t1 - t0




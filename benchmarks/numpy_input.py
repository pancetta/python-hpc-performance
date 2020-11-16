import time
import numpy as np
from .benchmarks import Benchmarks


class NumpyInput(Benchmarks):

    def __init__(self, name, params, num_threads, comm):
        super(NumpyInput, self).__init__(name=name, params=params, num_threads=num_threads, comm=comm)
        np.random.seed(0)
        x = np.linspace(0, 1, self.params.n3d)
        y = np.linspace(0, 1, self.params.n3d)
        X, Y = np.meshgrid(x, y)

        data = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

        if self.params.compressed:
            self.file = f'out_{comm.Get_rank()}.npz'
            np.savez_compressed(file=self.file, arr=data)
        else:
            self.file = f'out_{comm.Get_rank()}.npy'
            np.save(file=self.file, arr=data)

    def run(self):

        t0 = time.time()
        _ = np.load(file=self.file)
        t1 = time.time()

        return t1 - t0




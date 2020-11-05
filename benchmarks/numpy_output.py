import time
import numpy as np
from .benchmarks import Benchmarks


class NumpyOutput(Benchmarks):

    def __init__(self, name, params, comm):
        super(NumpyOutput, self).__init__(name=name, params=params, comm=comm)
        np.random.seed(0)
        x = np.linspace(0, 1, self.params.n3d)
        y = np.linspace(0, 1, self.params.n3d)
        X, Y = np.meshgrid(x, y)

        self.data = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        noise = np.random.default_rng().uniform(low=-self.params.noise_level,
                                                high=self.params.noise_level,
                                                size=(self.params.n3d, self.params.n3d))
        self.data += noise
        if self.params.compressed:
            self.file = f'out_{comm.Get_rank()}.npz'
            self.write = np.savez_compressed
        else:
            self.file = f'out_{comm.Get_rank()}.npy'
            self.write = np.save

    def run(self):

        t0 = time.time()
        self.write(file=self.file, arr=self.data)
        t1 = time.time()

        return t1 - t0




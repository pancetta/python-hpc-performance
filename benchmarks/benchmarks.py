from tools.params import Params
from abc import abstractmethod


class Benchmarks(object):

    def __init__(self, name=None, params=None, num_threads=None, comm=None):
        self.name = name
        self.num_threads = num_threads
        self.comm = comm
        self.params = Params(params)

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def reset(self):
        pass

    def tear_down(self):
        pass

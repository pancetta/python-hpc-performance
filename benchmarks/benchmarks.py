from tools.params import Params
from abc import abstractmethod


class Benchmarks(object):

    def __init__(self, name=None, comm=None, params=None):
        self.name = name
        self.comm = comm
        self.params = Params(params)

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def reset(self):
        pass

    def tear_down(self):
        pass

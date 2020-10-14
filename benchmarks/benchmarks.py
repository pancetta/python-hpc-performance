from tools.params import Params
from abc import abstractmethod


class Benchmarks(object):

    def __init__(self, name=None, params=None):
        self.name = name
        self.params = Params(params)

    @abstractmethod
    def run(self):
        raise NotImplementedError()

import time
import subprocess

from .benchmarks import Benchmarks


class Import(Benchmarks):

    def run(self):

        t0 = time.time()
        subprocess.call(['python', '-c', 'import ' + self.params.modules])
        t1 = time.time()

        return t1 - t0




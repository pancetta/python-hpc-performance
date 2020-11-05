from numba import jit
import random
import time
import numpy as np

@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

def monte_carlo_pi_np(nsamples):
    x = np.random.rand(nsamples)
    y = np.random.rand(nsamples)
    r = x ** 2 + y ** 2
    return 4.0 * np.sum([r < 1.0]) / nsamples


monte_carlo_pi(1)
start = time.time()
r = monte_carlo_pi(1000000)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
print(r)

start = time.time()
r = monte_carlo_pi_np(1000000)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
print(r)
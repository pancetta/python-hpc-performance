import numpy as np
import pytest
from scipy.optimize import minimize, rosen


def minimizer(x0, method):
    res = minimize(rosen, x0, method=method, tol=1E-10, options={'maxiter': 100000})
    return res


@pytest.mark.parametrize("method", ['Nelder-Mead', 'Powell'])
@pytest.mark.parametrize("N", [2, 5, 10, 15])
def test_minimizer(benchmark, N, method):
    np.random.seed(0)
    x0 = np.random.rand(N)
    res = benchmark(minimizer, x0, method)
    assert abs(sum(res['x'])-N) < 3E-10

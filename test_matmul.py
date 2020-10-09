import pytest
import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits


def matmul(matA, matB):
    matA @ matB

# @pytest.mark.parametrize("limits", [1, 4])
# @pytest.mark.parametrize("dtype", ['float64', 'float32'])
# @pytest.mark.parametrize("n, m", [(1000, 1000), (2000, 2000), (1000, 2000), (2000, 1000)])
# def test_matmul_np(benchmark, dtype, limits, n, m):
#
#     np.random.seed(0)
#     A = np.random.rand(n, m).astype(dtype)
#     B = np.random.rand(m, n).astype(dtype)
#
#     with threadpool_limits(limits=limits, user_api='blas'):
#         benchmark(matmul, A, B)


@pytest.mark.parametrize("density", [0.01, 0.005, 0.001])
@pytest.mark.parametrize("n, m", [(2000, 2000), (4000, 4000), (2000, 4000), (4000, 2000)])
def test_matmul_sp(benchmark, density, n, m):

    np.random.seed(0)
    A = sp.random(n, m, density=density)
    B = sp.random(m, n, density=density)

    benchmark(matmul, A, B)

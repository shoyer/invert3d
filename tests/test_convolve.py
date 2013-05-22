import numpy as np
from numpy.testing import assert_allclose, assert_equal

from invert3des.convolve import R3_to_P3 as response_shift


def test_response_shift():
    A = np.arange(5 ** 3).reshape((5, 5, 5))
    B = response_shift(A)
    L = A.shape[0]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                assert_allclose(A[i, (i + j) % L, (i + j + k) % L],
                                B[i, j, k])

def test_response_shift_enlarge():
    A0 = np.arange(5 ** 3).reshape((5, 5, 5))
    A = response_enlarge(A0)
    B = response_shift_enlarge(A0)
    L = A.shape[0]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                assert_allclose(A[i, i + j, i + j + k], B[i, j, k])


def test_inverse_response_shift():
    A = np.arange(5 ** 3).reshape((5, 5, 5))
    assert_allclose(A, inverse_response_shift(response_shift(A)))


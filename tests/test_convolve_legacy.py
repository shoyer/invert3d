import numpy as np
from itertools import product
from numpy.testing import assert_equal

from invert3d import convolve_legacy as convolve


class TrivialPulseTests(object):
    def test_default_pulses(self):
        B = convolve.R3_to_P3(self.A, trim=True, include_margin=False)
        assert_equal(self.A, B)

    def test_same_pulses(self):
        B = convolve.R3_to_P3(self.A, E_all=((0, 1, 0), (0, 1, 0), (0, 1, 0)),
                              trim=True, include_margin=False)
        assert_equal(self.A, B)

    def test_different_pulses_asc(self):
        B = convolve.R3_to_P3(self.A, ((0, 1, 0), (0, 0, 1, 0, 0),
                                       (0, 0, 0, 0, 1, 0, 0, 0, 0)),
                              trim=True, include_margin=False)
        assert_equal(self.A, B)

    def test_different_pulses_desc(self):
        B = convolve.R3_to_P3(self.A, ((0, 0, 0, 0, 1, 0, 0, 0, 0),
                                       (0, 0, 1, 0, 0), (0, 1, 0)),
                              trim=True, include_margin=False)
        assert_equal(self.A, B)


class TestTrivialPulses1(TrivialPulseTests):
    def setUp(self):
        self.A = (1 + np.arange(5 ** 3)).reshape((5, 5, 5))


class TestTrivialPulses2(TrivialPulseTests):
    def setUp(self):
        self.A = (1 + np.arange(4 * 5 * 6)).reshape((4, 5, 6))


def test_shift_time_indices():
    A = (1 + np.arange(2 * 4 * 6)).reshape((2, 4, 6))
    B = convolve.shift_time_indices(A, trim=False)
    C = np.zeros_like(A)
    for i, j, k in product(*map(xrange, A.shape)):
        if (i + j) < A.shape[1] and (i + j + k) < A.shape[2]:
            C[i, j, k] = A[i, i + j, i + j + k]
    assert_equal(B, C)

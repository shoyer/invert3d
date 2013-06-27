import numpy as np
from numpy.testing import assert_equal

from invert3d import convolve_legacy as convolve, deconvolve


def test_shift_time_indices_undo():
    A = (1 + np.arange(2 * 4 * 6)).reshape((2, 4, 6))
    B = convolve.shift_time_indices(A, trim=True, include_margin=True)
    Aprime = deconvolve.shift_time_indices_undo(B)
    nz = Aprime > 0
    assert_equal(A[nz], Aprime[nz])
    for element in Aprime[~nz]:
        assert element == 0

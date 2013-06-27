import numpy as np
from numpy.testing import assert_equal

from invert3d import convolve


def test_S3_shape_to_R3_shape():
    R3_shape = (13, 12, 11)
    E_all = map(convolve.ControlField, ((0, 1, 0), (0, 1, 0), (0, 0, 1, 0, 0)))
    _, shapes, _  = convolve.convolution_parameters(R3_shape, E_all)
    assert_equal(R3_shape, convolve.S3_shape_to_R3_shape(shapes[-1], E_all))


def test_shift_time_indices_undo():
    E_all = map(convolve.ControlField, ((0, 1, 0), (0, 1, 0), (0, 0, 1, 0, 0)))
    A = (1 + np.arange(2 * 4 * 6)).reshape((2, 4, 6))
    B = convolve.shift_time_indices(A, E_all)
    Aprime = convolve.unshift_time_indices(B, E_all)
    assert_equal(A, Aprime)


def test_resize_ticks():
    ticks = (np.arange(100), np.arange(200), np.arange(300))
    E = convolve.ControlField([1, 2, 3, 2, 1])
    E_all = (E, E, E)
    assert_equal([len(t) for t in convolve.resize_ticks(ticks, E_all)],
                 convolve.resized_shape((len(t) for t in ticks), E_all))

from numpy.testing import assert_equal

from invert3des import convolve


def test_S3_shape_to_R3_shape():
    R3_shape = (13, 12, 11)
    E_all = ((0, 1, 0), (0, 1, 0), (0, 0, 1, 0, 0))
    _, shapes, _  = convolve.convolution_parameters(R3_shape, E_all)
    assert_equal(R3_shape, convolve.S3_shape_to_R3_shape(shapes[-1], E_all))

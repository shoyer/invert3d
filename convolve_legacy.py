import numpy as np

from convolve import convolution_operator, loop_matvec
from utils import MetaArray


def R3_to_P3(R, E_all=((1,), (1,), (1,)), trim=True, include_margin=True):
    E3, E2, E1 = E_all
    P3 = shift_time_indices(
        convolve_third(convolve_second(convolve_first(R, E1), E2), E3),
        E_all, trim, include_margin)

    try:
        return MetaArray(P3, ticks=expand_ticks(R, E_all), rw_freq=R.rw_freq)
    except AttributeError:
        return P3


def convolve_first(R, E1=(1,)):
    dL = len(E1) - 1
    L3, L2, L1 = R.shape
    S1 = np.zeros((L3, L2, L1 + L2 + L3 + dL), dtype=R.dtype)
    A = convolution_operator(E1, L1, False, False)
    for t3 in xrange(L3):
        for t2 in xrange(L2):
            S1[t3, t2, (t3 + t2):(t3 + t2 + L1 + dL)] = A.dot(R[t3, t2, :])
    return S1


def convolve_second(S1, E2=(1,)):
    dL = len(E2) - 1
    L3, L2, L1 = S1.shape
    S2 = np.zeros((L3, L2 + L3 + dL, L1), dtype=S1.dtype)
    A = convolution_operator(E2, L2, False, False)
    for t3 in xrange(L3):
        for sum12 in xrange(L1):
            S2[t3, t3:(t3 + L2 + dL), sum12] = A.dot(S1[t3, :, sum12])
    return S2


def convolve_third(S2, E3=(1,)):
    dL = len(E3) - 1
    L3, L2, L1 = S2.shape
    S3 = np.zeros((L3 + dL, L2, L1), dtype=S2.dtype)
    A = convolution_operator(E3, L3, False, False)
    for sum23 in xrange(L2):
        for T1 in xrange(L1):
            S3[:, sum23, T1] = A.dot(S2[:, sum23, T1])
    return S3


def shift_time_indices(S3, E_all=((1,), (1,), (1,)), trim=True,
                       include_margin=True):
    """
    Shift indices from (T3, T3 + T2, T3 + T2 + T1) to (T3, T2, T1)
    """
    P3 = S3.copy()

    E3, E2, E1 = E_all
    L3 = P3.shape[0] - len(E3) + 1
    L2 = P3.shape[1] - L3 - len(E2) + 1
    L1 = P3.shape[2] - L3 - L2 - len(E1) + 1

    tau3, tau2, tau1 = (int((len(E) - 1) / 2.0) for E in E_all)

    for T3 in xrange(S3.shape[0]):
        P3[T3, :, :] = np.roll(P3[T3, :, :], -T3 + tau3, axis=0)
        P3[T3, :, :] = np.roll(P3[T3, :, :], -T3 + tau3, axis=1)
        if T3 > tau3:
            P3[T3, (-T3 + tau3):, :] = 0
            P3[T3, :, (-T3 + tau3):] = 0

    for T2 in xrange(S3.shape[1]):
        P3[:, T2, :] = np.roll(P3[:, T2, :], -T2 + tau2, axis=1)
        if T2 > tau2:
            P3[:, T2, (-T2 + tau2):] = 0

    if trim:
        if include_margin:
            P3 = P3[:(L3 + len(E3) - 1),
                    :(L2 + len(E2) - 1),
                    :(L1 + len(E1) - 1)]
        else:
            P3 = P3[tau3:(L3 + tau3),
                    tau2:(L2 + tau2),
                    tau1:(L1 + tau1)]
    return P3


def expand_ticks(R, E_all=((1,), (1,), (1,))):
    ticks_new = []
    for t, Ei in zip(R.ticks, E_all):
        dt = t[1] - t[0]
        dL = int((len(Ei) - 1) / 2.0)
        t_before = t[0] - dt * np.arange(1, dL + 1)[::-1]
        t_after = t[-1] + dt * np.arange(1, dL + 1)
        ticks_new.append(np.concatenate((t_before, t, t_after)))
    return tuple(ticks_new)


def R3_to_P3_alt(R, E_all=((1,), (1,), (1,)), trim=True, include_margin=True):
    conv_mat, shapes, slice_maps = convolution_parameters(R.shape, E_all)
    P3 = shift_time_indices(
        reduce(loop_matvec, zip(conv_mat, shapes[1:], slice_maps), R),
        E_all, trim, include_margin)

    try:
        return MetaArray(P3, ticks=expand_ticks(R, E_all), rw_freq=R.rw_freq)
    except AttributeError:
        return P3


def convolution_parameters((L3, L2, L1), E_all=((1,), (1,), (1,))):
    E3, E2, E1 = E_all
    conv_matrices = [convolution_operator(E1, L1, False, False),
                     convolution_operator(E2, L2, False, False),
                     convolution_operator(E3, L3, False, False)]
    d1, d2, d3 = (len(conv_matrices[i]) for i in xrange(3))
    M1 = L3 + L2 + d1
    M2 = L3 + d2
    M3 = d3
    shapes = [(L3, L2, L1), (L3, L2, M1), (L3, M2, M1), (M3, M2, M1)]
    slice_maps = [[((t3, t2, slice(t3 + t2, t3 + t2 + d1)), (t3, t2))
                   for t3 in xrange(L3) for t2 in xrange(L2)],
                  [((t3, slice(t3, t3 + d2), sum12), (t3, slice(None), sum12))
                   for t3 in xrange(L3) for sum12 in xrange(M1)],
                  [((slice(None), sum23, T1), (slice(None), sum23, T1))
                   for sum23 in xrange(M2) for T1 in xrange(M1)]]
    return conv_matrices, shapes, slice_maps

def R3_add_margin(R, E_all=((1,), (1,), (1,))):
    tau = np.array(2[int((len(E) - 1) / 2.0) for E in E_all])
    tau3, tau2, tau1 = tau
    L3, L2, L1 = R.shape

    R_new = np.zeros(tuple(np.array(R.shape) + tau), dtype=R.dtype)
    R_new[tau3:(L3 + tau3), tau2:(L2 + tau2), tau1:(L1 + tau1)] = R

    try:
        return MetaArray(R_new, ticks=expand_ticks(R, E_all), rw_freq=R.rw_freq)
    except AttributeError:
        return R_new

def trim_pulse_overlap(R, E_all=((1,), (1,), (1,))):
    # Not quite right?
    start = len(E_all[0]) + len(E_all[1])
    end = None

    R_new = R[:, start:end, :]
    ticks_new = (R.ticks[0], R.ticks[1][start:end], R.ticks[2])
    return MetaArray(R_new, ticks=ticks_new, rw_freq=R.rw_freq)


class DotProduct(object):
    def __init__(self, matrix):
        self.matrix = matrix
    def __call__(self, other):
        return self.matrix.dot(other)


def convolve_loop_map(R, convolution_matrix, to_shape, slice_map, map_=None):
    S = np.zeros(to_shape, dtype=R.dtype)
    if map_ is None:
        map_ = lambda *args: Pool(processes=4).map(*args, chunksize=1000)
    # put the dot product into an object so it can be pickled
    results = map_(DotProduct(convolution_matrix),
                   (R[slice_from] for _, slice_from in slice_map))
    for (slice_to, _), result in zip(slice_map, results):
        S[slice_to] = result
    return S

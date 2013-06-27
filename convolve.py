import numpy as np

from utils import MetaArray, end


def convolution_operator(pulse_vec, signal_len, trim_left_boundary=True,
                         trim_right_boundary=True):
    pulse_vec = np.asanyarray(pulse_vec[:])
    C = np.zeros((len(pulse_vec) + signal_len - 1, signal_len),
                 dtype=pulse_vec.dtype)
    for i in xrange(signal_len):
        C[i:(len(pulse_vec) + i), i] = pulse_vec[:]
    if trim_left_boundary:
        C = C[(len(pulse_vec) - 1):, :]
    if trim_right_boundary:
        C = C[:(1 - len(pulse_vec) if len(pulse_vec) > 1 else None), :]
    return C


# def loop_matvec(R, (matrix, to_shape, slice_map)):
#     """Apply matrix.dot to many independent slices of R"""
#     S = np.zeros(to_shape, dtype=R.dtype)
#     for slice_to, slice_from in slice_map:
#         S[slice_to] = matrix.dot(R[slice_from])
#     return S


def loop_matvec(R, (to_shape, slice_map)):
    """Apply matrix.dot to many independent slices of R"""
    S = np.zeros(to_shape, dtype=R.dtype)
    for matrix, slice_to, slice_from in slice_map:
        S[slice_to] = matrix.dot(R[slice_from])
    return S


# def convolution_parameters((L3, L2, L1), (E3, E2, E1)):
#     conv_matrices = [convolution_operator(E1, L1),
#                      convolution_operator(E2, L2),
#                      convolution_operator(E3, L3, trim_left_boundary=False)]
#     d1, d2, d3 = (len(conv_matrices[i]) for i in xrange(3))
#     M1 = L3 + L2 + d1
#     M2 = L3 + d2
#     M3 = d3
#     shapes = [(L3, L2, L1), (L3, L2, M1), (L3, M2, M1), (M3, M2, M1)]
#     slice_maps = [[((t3, t2, slice(t3 + t2, t3 + t2 + d1)), (t3, t2))
#                    for t3 in xrange(L3) for t2 in xrange(L2)],
#                   [((t3, slice(t3, t3 + d2), sum12), (t3, slice(None), sum12))
#                    for t3 in xrange(L3) for sum12 in xrange(M1)],
#                   [((slice(None), sum23, sum123), (slice(None), sum23, sum123))
#                    for sum23 in xrange(M2) for sum123 in xrange(M1)]]
#     return conv_matrices, shapes, slice_maps


def convolution_parameters((L3, L2, L1), (E3, E2, E1)):
    conv_matrices = [convolution_operator(E1, L1),
                     convolution_operator(E2, L2),
                     # convolution_operator(E2, L2)[:end(E1.center), :],
                     convolution_operator(E3, L3, trim_left_boundary=False)]
    d1, d2, d3 = (len(conv_matrices[i]) for i in xrange(3))
    M1 = L3 + L2 + d1
    M2 = L3 + d2
    M3 = d3
    shapes = [(L3, L2, L1), (L3, L2, M1), (L3, M2, M1), (M3, M2, M1)]
    slice_maps = [[], [], []]

    for t3 in xrange(L3):
        for t2 in xrange(L2):
            slice_maps[0].append((conv_matrices[0][E1.center:end(E1.center)],
                                  (t3, t2, slice(t3 + t2 + E1.center,
                                                 t3 + t2 + d1 - E1.center)),
                                  (t3, t2)))
            # slice_maps[0].append((conv_matrices[0],
            #                       (t3, t2, slice(t3 + t2, t3 + t2 + d1)),
            #                       (t3, t2)))

    for t3 in xrange(L3):
        for sum12 in xrange(M1):
            slice_maps[1].append((conv_matrices[1],
                                  (t3, slice(t3, t3 + d2), sum12),
                                  (t3, slice(None), sum12)))

    for sum23 in xrange(M2):
        for sum123 in xrange(M1):
            sl = slice(max(0, sum23 + len(E3) + E2.center - L3),
                       min(M3, sum23 + 1))
            slice_maps[2].append((conv_matrices[2][sl, :],
                                  (sl, sum23, sum123),
                                  (slice(None), sum23, sum123)))
    return shapes, slice_maps


# def R3_to_P3(R3, E_all, shortcut=False):
#     conv_mat, shapes, slice_maps = convolution_parameters(R3.shape, E_all)
#     S3 = reduce(loop_matvec, zip(conv_mat, shapes[1:], slice_maps), R3)
#     if shortcut:
#         return S3
#     P3 = P3_resize(shift_time_indices(S3, E_all), E_all)

#     try:
#         return MetaArray(P3, ticks=resize_ticks(R3.ticks, E_all),
#                          rw_freq=R3.rw_freq)
#     except AttributeError:
#         return P3


def R3_to_P3(R3, E_all, shortcut=False):
    shapes, slice_maps = convolution_parameters(R3.shape, E_all)
    S3 = reduce(loop_matvec, zip(shapes[1:], slice_maps), R3)
    if shortcut:
        return S3
    P3 = P3_resize(shift_time_indices(S3, E_all), E_all)

    try:
        return MetaArray(P3, ticks=resize_ticks(R3.ticks, E_all),
                         rw_freq=R3.rw_freq)
    except AttributeError:
        return P3



def R3_resize(R3, E_all):
    E3, E2, E1 = E_all
    R3_new = np.zeros(resized_shape(R3.shape, E_all), dtype=R3.dtype)
    R3_new[-(R3.shape[0] - E3.center):, :, :] = \
        R3[:end(E3.center),
           (E2.center + E3.center):end(E2.center),
           E1.center:end(E1.center)]
    try:
        return MetaArray(R3_new, ticks=resize_ticks(R3.ticks, E_all),
                         rw_freq=R3.rw_freq)
    except AttributeError:
        return R3_new


def P3_resize(P3, E_all):
    E3, E2, E1 = E_all
    R3_shape = S3_shape_to_R3_shape(P3.shape, E_all)
    P3_new = np.zeros(resized_shape(R3_shape, E_all), dtype=P3.dtype)
    P3_new[-P3.shape[0]:, :, :] = P3[
        :, E3.center:(P3_new.shape[1] + E2.center),
        :P3_new.shape[2]]
    return P3_new


def P3_unsize(P3, R3_shape, E_all, fill=0):
    E3, E2, E1 = E_all
    shapes, _ = convolution_parameters(R3_shape, E_all)
    P3_old = np.empty(shapes[-1], dtype=P3.dtype)
    P3_old[:] = fill
    P3_old[:, E3.center:(P3.shape[1] + E2.center),
           :P3.shape[2]] = P3[-P3_old.shape[0]:, :, :]
    return P3_old


def shift_time_indices(S3, (E3, E2, _)):
    """
    Shift indices from (T3, T3 + T2, T3 + T2 + T1) to (T3, T2, T1)
    """
    P3 = S3.copy()
    for T3 in xrange(S3.shape[0]):
        P3[T3, :, :] = np.roll(P3[T3, :, :], -T3 + E3.center, axis=0)
        P3[T3, :, :] = np.roll(P3[T3, :, :], -T3 + E3.center, axis=1)
    for T2 in xrange(S3.shape[1]):
        P3[:, T2, :] = np.roll(P3[:, T2, :], -T2 - E2.center, axis=1)
    return P3



def unshift_time_indices(P3, (E3, E2, _)):
    """
    Shift from indices (T3, T2, T1) to (T3, T3 + T2, T3 + T2 + T1)
    """
    S3 = P3.copy()
    for T2 in xrange(S3.shape[1]):
        S3[:, T2, :] = np.roll(S3[:, T2, :], T2 + E2.center, axis=1)
    for T3 in xrange(S3.shape[0]):
        S3[T3, :, :] = np.roll(S3[T3, :, :], T3 - E3.center, axis=0)
        S3[T3, :, :] = np.roll(S3[T3, :, :], T3 - E2.center, axis=1)
    return S3


def resize_ticks((ticks3, ticks2, ticks1), (E3, E2, E1)):
    return (np.append(-ticks3[:end(E3.center)][:0:-1], ticks3[:end(E3.center)]),
            ticks2[(E2.center + E3.center):end(E2.center + E1.center)],
            ticks1[E1.center:end(E1.center)])


def S3_shape_to_R3_shape((M3, M2, M1), (E3, E2, E1)):
    L3 = M3
    L2 = M2 - L3 + 2 * E2.center  # + E1.center
    L1 = M1 - L3 - L2 + 2 * E1.center
    return (L3, L2, L1)


def resized_shape((L3, L2, L1), (E3, E2, E1)):
    return (2 * L3 - 2 * E3.center - 1,
            L2 - 2 * E2.center - E1.center,
            L1 - 2 * E1.center)


class ControlField(object):
    def __init__(self, pulse_vec, center=None):
        self.vec = np.asanyarray(pulse_vec)
        self.center = (int((len(self) - 1) / 2.0)
                       if center is None else center)
        if (len(self) % 2) == 0:
            raise ValueError('pulse should have odd length (since the center '
                             'of each pulse is assumed to be in the middle)')

    def __getitem__(self, item):
        return self.vec[item]

    def __len__(self):
        return len(self.vec)

    @property
    def dtype(self):
        return self.vec.dtype

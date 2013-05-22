import numpy as np

from utils import MetaArray


def dlen(x):
    "Returns (length(x)-1)/2"
    return int((len(x) - 1) / 2)


def convolve_matrix(pulse_vec, signal_len):
    pulse_vec = np.array(pulse_vec if pulse_vec is not None else [1])
    if (len(pulse_vec) % 2) == 0:
        raise ValueError('Pulse_vec should have odd length')

    C = np.zeros((len(pulse_vec)+signal_len-1, signal_len),
                 dtype=pulse_vec.dtype)
    for i in xrange(signal_len):
        C[i:(len(pulse_vec) + i), i] = pulse_vec
    return C


def convolve_first(R, E1=None):
    dL = dlen(E1)
    L3, L2, L1 = R.shape
    S1 = np.zeros((L3, L2, L3 + L2 + L1 + 2 * dL), dtype=R.dtype)
    # convolution loop
    A = convolve_matrix(E1, L1)
    for t3 in xrange(L3):
        for t2 in xrange(L2):
            # integrate over t1
            x = R[t3, t2, :]
            start = L3 - t3 + L2 - t2
            end = start + L1 + 2 * dL
            S1[t3, t2, start:end] = A.dot(x)
    return S1


def convolve_second(S1, E2=None):
    dL = dlen(E2)
    L3, L2, L1 = S1.shape
    S2 = np.zeros((L3, L3 + L2 + 2 * dL, L1), dtype=S1.dtype)
    # convolution loop
    A = convolve_matrix(E2, L2)
    for t3 in xrange(L3):
        for Sig12 in xrange(L1):
            # integrate over t2
            x = S1[t3, :, Sig12]
            start = L3 - t3
            end = start + L2 + 2 * dL
            S2[t3, start:end, Sig12] = A.dot(x)
    return S2


def convolve_third(S2, E3=None):
    dL = dlen(E3)
    L3, L2, L1 = S2.shape
    S3 = np.zeros((L3 + 2 * dL, L2, L1), dtype=S2.dtype)
    # integration loopR
    A = convolve_matrix(E3, L3)
    for Sum23 in xrange(L2):
        for T1 in xrange(L1):
            # integrate over t3
            x = S2[:, Sum23, T1]
            S3[:, Sum23, T1] = A.dot(x)
    return S3


def shift_all(S3, E_all=None, truncate=False):
    "Shift from indices (T3, T3 + T2, T3 + T2 + T1) to (T3, T2, T1)"
    P3 = S3.copy()
    if np.array(E_all).ndim == 1:
        E_all = [E_all, E_all, E_all]

    dL3, dL2, _ = [dlen(Ei) for Ei in E_all]
    L3 = P3.shape[0] - 2 * dL3
    L2 = P3.shape[1] - L3 - 2 * dL2
    # L1 = P3.shape[2] - L3 - L2 - 2 * dL1
    for T3 in range(S3.shape[0]):
        P3[T3, :, :] = np.roll(P3[T3, :, :], T3 - dL3, axis=0)
        P3[T3, :, :] = np.roll(P3[T3, :, :], T3 + L2 - L3 - dL3, axis=1)
    for T2 in range(S3.shape[1]):
        P3[:, T2, :] = np.roll(P3[:, T2, :], T2 - L2 - dL2, axis=1)
    if truncate:
        P3 = P3[:, L3:, (L3 + L2):]
    return P3


def R3_to_P3(R, E_all=None, truncate=False):
    if np.array(E_all).ndim == 1:
        E_all = [E_all, E_all, E_all]

    ticks_new = []
    for t, Ei in zip(R.ticks, E_all):
        dt = t[1] - t[0]
        dL = dlen(Ei)
        t_before = t[0] - dt * np.arange(1, dL + 1)[::-1]
        t_after = t[-1] + dt * np.arange(1, dL + 1)
        ticks_new.append(np.concatenate((t_before, t, t_after)))

    P3 = shift_all(convolve_third(convolve_second(convolve_first(R, E_all[2]),
                                                  E_all[1]),
                                  E_all[1]),
                   E_all, truncate)

    return MetaArray(P3, ticks=tuple(ticks_new), rw_freq=R.rw_freq)


def R3_add_margin(R, E_all=None):
    if np.array(E_all).ndim == 1:
        E_all = [E_all, E_all, E_all]

    ticks_new = []
    for t, Ei in zip(R.ticks, E_all):
        dt = t[1] - t[0]
        dL = dlen(Ei)
        t_before = t[0] - dt * np.arange(1, dL + 1)[::-1]
        t_after = t[-1] + dt * np.arange(1, dL + 1)
        ticks_new.append(np.concatenate((t_before, t, t_after)))

    R_new = np.zeros(tuple(len(t) for t in ticks_new), dtype=R.dtype)

    dL_vec = [dlen(Ei) for Ei in E_all]
    R_new[dL_vec[0]:-dL_vec[0], dL_vec[1]:-dL_vec[1], dL_vec[2]:-dL_vec[2]] = R

    return MetaArray(R_new, ticks=tuple(ticks_new), rw_freq=R.rw_freq)


def trim_pulse_overlap(R, E_all=None):
    if np.array(E_all).ndim == 1:
        E_all = [E_all, E_all, E_all]

    # Not quite right?
    start = 2 * max(dlen(E_all[0]), dlen(E_all[1]))
    end = -(2 * dlen(E_all[2]) + start)

    R_new = R[:, start:end, :]
    R_new.update(ticks=(R.ticks[0], R.ticks[1][start:end], R.ticks[2]))

    return R_new


def select_NR(R_old):
    R = R_old.copy()
    x = (R.shape[2] - 1) / 2
    R[:, :, :x] = 0
    R[:, :, x] /= 2
    return R


def select_PE(R_old):
    R = R_old.copy()
    x = (R.shape[2] - 1) / 2
    R[:, :, -x:] = 0
    R[:, :, x] /= 2
    return R


def reweight_R3(R, freq_weights):
    if np.array(freq_weights).ndim == 1:
        freq_weights = [freq_weights, freq_weights, freq_weights]
    return (R * freq_weights[0].reshape(-1, 1, 1)
            * (freq_weights[1] * freq_weights[2]).reshape(1, 1, -1) ** 2)

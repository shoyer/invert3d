import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, fftfreq

from utils import MetaArray


def convolve_matrix(pulse_vec, signal_len):
    pulse_vec = np.asanyarray(pulse_vec)
    if (len(pulse_vec) % 2) == 0:
        raise ValueError('pulse_vec should have odd length (since the center '
                         'of each pulse is assumed to be in the middle)')

    C = np.zeros((len(pulse_vec) + signal_len - 1, signal_len),
                 dtype=pulse_vec.dtype)
    for i in xrange(signal_len):
        C[i:(len(pulse_vec) + i), i] = pulse_vec
    return C


def convolve_first(R, E1=(1,)):
    """
    Integrate over t1
    """
    dL = len(E1) - 1
    L3, L2, L1 = R.shape
    S1 = np.zeros((L3, L2, L3 + L2 + L1 + dL), dtype=R.dtype)
    A = convolve_matrix(E1, L1)
    for t3 in xrange(L3):
        for t2 in xrange(L2):
            x = R[t3, t2, :]
            start = t3 + t2
            end = start + L1 + dL
            S1[t3, t2, start:end] = A.dot(x)
    return S1


def convolve_second(S1, E2=(1,)):
    """
    Integrate over t2
    """
    dL = len(E2) - 1
    L3, L2, L1 = S1.shape
    S2 = np.zeros((L3, L3 + L2 + dL, L1), dtype=S1.dtype)
    A = convolve_matrix(E2, L2)
    for t3 in xrange(L3):
        for sum12 in xrange(L1):
            # integrate over t2
            x = S1[t3, :, sum12]
            start = t3
            end = start + L2 + dL
            S2[t3, start:end, sum12] = A.dot(x)
    return S2


def convolve_third(S2, E3=(1,)):
    """
    Integrate over t3
    """
    dL = len(E3) - 1
    L3, L2, L1 = S2.shape
    S3 = np.zeros((L3 + dL, L2, L1), dtype=S2.dtype)
    A = convolve_matrix(E3, L3)
    for sum23 in xrange(L2):
        for T1 in xrange(L1):
            x = S2[:, sum23, T1]
            S3[:, sum23, T1] = A.dot(x)
    return S3


def shift_all(S3, E_all=((1,), (1,), (1,)), trim=True, include_margin=True):
    "Shift from indices (T3, T3 + T2, T3 + T2 + T1) to (T3, T2, T1)"
    P3 = S3.copy()

    E3, E2, E1 = E_all
    L3 = P3.shape[0] - len(E3) + 1
    L2 = P3.shape[1] - L3 - len(E2) + 1
    L1 = P3.shape[2] - L3 - L2 - len(E1) + 1

    tau3, tau2, tau1 = (int((len(E) - 1) / 2.0) for E in E_all)

    for T3 in np.arange(S3.shape[0]):
        P3[T3, :, :] = np.roll(P3[T3, :, :], -T3 + tau3, axis=0)
        P3[T3, :, :] = np.roll(P3[T3, :, :], -T3 + tau3, axis=1)
        if T3 > tau3:
            P3[T3, (-T3 + tau3):, :] = 0
            P3[T3, :, (-T3 + tau3):] = 0

    for T2 in np.arange(S3.shape[1]):
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



def R3_to_P3(R, E_all=((1,), (1,), (1,)), trim=True, include_margin=True):
    E3, E2, E1 = E_all
    P3 = shift_all(
        convolve_third(convolve_second(convolve_first(R, E1), E2), E3),
        E_all, trim, include_margin)
    try:
        return MetaArray(P3, ticks=expand_ticks(R, E_all), rw_freq=R.rw_freq)
    except AttributeError:
        return P3


def R3_add_margin(R, E_all=((1,), (1,), (1,))):
    tau = (int((len(E) - 1) / 2.0) for E in E_all)
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


def combine_NR_PE(R_NR, R_PE, add_t3_negative=True):
    if add_t3_negative:
        pad_t3 = R_NR.shape[0] - 1
    else:
        pad_t3 = 0

    x3, x2, x1 = R_NR.shape
    y3, y2, y1 = R_PE.shape
    if (x3 != y3) or (x2 != y2):
        raise ValueError('Mismatched t3 or t2')

    R_c = np.zeros((x3 + pad_t3, x2, x1 + y1 - 1), dtype=R_NR.dtype)
    R_c[pad_t3:, :, :(x1-1)] = R_NR[:, :, :0:-1]
    R_c[pad_t3:, :, x1:] = R_PE[:, :, 1:]
    R_c[pad_t3:, :, (x1-1)] = (R_PE[:, :, 0] + R_NR[:, :, 0])/2.0

    t3 = np.concatenate((-R_NR.ticks[0][pad_t3:0:-1], R_NR.ticks[0]))
    t2 = R_NR.ticks[1]
    t1 = np.concatenate((-R_NR.ticks[2][:0:-1], R_PE.ticks[2]))

    return MetaArray(R_c, ticks=(t3, t2, t1), rw_freq=R_NR.rw_freq)


def reweight_R3(R, freq_weights):
    if np.array(freq_weights).ndim == 1:
        freq_weights = [freq_weights, freq_weights, freq_weights]
    return (R * freq_weights[0].reshape(-1, 1, 1)
            * (freq_weights[1] * freq_weights[2]).reshape(1, 1, -1))


def fft_2D(R, convert=3e-5, freq_bounds=None):
    """
    Perform a 2D FFT to transform a 3rd order response function defined in the
    rotating frame into a series of 2D spectra.

    First argument must be a MetaArray object wth ticks and rw_freq defined.

    Returns the FFT in another MetaArray object with ticks updated to the
    calculated frequencies (converted using the convert argument which defaults
    to cm to fs).
    """

    # reverses frequency order to keep convention e^{+i \omega t}
    R_2D = fftshift(fft2(ifftshift(R, axes=(0, 2)), axes=(0, 2)),
                    axes=(0, 2))[::-1, :, ::-1]

    dt = [t[1] - t[0] for t in R.ticks]

    freqs = [fftshift(fftfreq(R.shape[axis], dt[axis] * convert))
             + R.rw_freq for axis in (0, 2)]

    if freq_bounds is not None:
        bounds = [bound_indices(ticks, freq_bounds) for ticks in freqs]
        freqs = [freq[bound[0]:bound[1]] for freq, bound in zip(freqs, bounds)]
        R_2D = R_2D[bounds[0][0]:bounds[0][1], :, bounds[1][0]:bounds[1][1]]

    return MetaArray(R_2D, ticks=(freqs[0], R.ticks[1], freqs[1]))


def bound_indices(ticks, bounds):
    i0 = np.argmin(np.abs(ticks - bounds[0]))
    i1 = np.argmin(np.abs(ticks - bounds[1]))
    return (min(i0, i1), max(i0, i1) + 1)


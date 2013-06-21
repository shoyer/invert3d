import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, fftfreq

from utils import MetaArray


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


def combine_NR_PE(R_NR, R_PE):
    x3, x2, x1 = R_NR.shape
    y3, y2, y1 = R_PE.shape
    if (x3 != y3) or (x2 != y2):
        raise ValueError('Mismatched t3 or t2')

    R_c = np.zeros((x3, x2, x1 + y1 - 1), dtype=R_NR.dtype)
    R_c[:, :, :(x1-1)] = R_NR[:, :, :0:-1]
    R_c[:, :, x1:] = R_PE[:, :, 1:]
    R_c[:, :, (x1-1)] = (R_PE[:, :, 0] + R_NR[:, :, 0]) / 2.0

    new_ticks = (R_NR.ticks[0], R_NR.ticks[1],
                 np.append(-R_NR.ticks[2][:0:-1], R_PE.ticks[2]))
    return MetaArray(R_c, ticks=new_ticks, rw_freq=R_NR.rw_freq)


def reweight_R3(R, freq_weights):
    if np.array(freq_weights).ndim == 1:
        freq_weights = [freq_weights, freq_weights, freq_weights]
    return (R * freq_weights[0].reshape(-1, 1, 1)
            * (freq_weights[1] * freq_weights[2]).reshape(1, 1, -1))


def unweight_R3(R, freq_weights):
    if np.array(freq_weights).ndim == 1:
        freq_weights = [freq_weights, freq_weights, freq_weights]
    return (R / freq_weights[0].reshape(-1, 1, 1)
            / (freq_weights[1] * freq_weights[2]).reshape(1, 1, -1))


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


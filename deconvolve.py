import numpy as np
import scipy.optimize
from scipy.linalg import norm, svd

from convolve import convolution_parameters, loop_matvec
from utils import MetaArray


def invert_loop_matvec(minimizer, **kwargs):
    def func(S, (matrix, to_shape, slice_map, damp0)):
        def solve(log_damp):
            mat_inv, gcv_k = generalized_tikhonov_inverse(matrix,
                                                          10 ** log_damp)
            R = loop_matvec(S, (mat_inv, to_shape, slice_map))
            return R, gcv_k

        def error(log_damp):
            R, gcv_k = solve(log_damp)
            return sum(gcv_error(R[slice_to] - matrix.dot(S[slice_from]),
                                 matrix.shape[1], gcv_k)
                       for slice_to, slice_from in slice_map)

        result = minimizer(error, np.log10(damp0), **kwargs)
        R, _ = solve(result.x)
        return R
    return func


def P3_to_R3_fixed_damp(P3, E_all=((1,), (1,), (1,)), damp=None, trim=True,
                        include_margin=True):
    conv_mat, shapes, slice_maps = convolution_parameters(P3.shape, E_all)
    deconv_mat = [Ainv for (Ainv, _) in generalized_tikhonov_inverse(A, d) for
                  A, d in zip(conv_mat, damp)]
    P3 = reduce(loop_matvec, zip(deconv_mat[::-1], shapes[-2::-1],
                                 [[(y, x) for (x, y) in slice_map]
                                  for slice_map in slice_maps[::-1]]),
                shift_time_indices_undo(P3, E_all, trim, include_margin))

    try:
        return MetaArray(P3, ticks=unexpand_ticks(P3, E_all),
                         rw_freq=P3.rw_freq)
    except AttributeError:
        return P3


def P3_to_R3_opt_damp(P3, E_all=((1,), (1,), (1,)), damp0=None, trim=True,
                      include_margin=True, minimizer=scipy.optimize.minimize,
                      **min_kwargs):
    conv_mat, shapes, slice_maps = convolution_parameters(P3.shape, E_all)
    P3 = reduce(invert_loop_matvec(minimizer, **min_kwargs),
                zip(conv_mat[::-1], shapes[-2::-1],
                    [[(y, x) for (x, y) in sm] for sm in slice_maps[::-1]],
                    damp0),
                shift_time_indices_undo(P3, E_all, trim, include_margin))

    try:
        return MetaArray(P3, ticks=unexpand_ticks(P3, E_all),
                         rw_freq=P3.rw_freq)
    except AttributeError:
        return P3


def shift_time_indices_undo(P3, E_all=((1,), (1,), (1,)), trim=True,
                            include_margin=True):
    """
    Shift from indices (T3, T2, T1) to (T3, T3 + T2, T3 + T2 + T1)

    Assumes input from shift_time_indices(trim=True, include_margin=True)
    """
    E3, E2, E1 = E_all
    L3, L2, L1 = (d - len(E) + 1 for E, d in zip(E_all, P3.shape))

    S3 = np.zeros((L3 + len(E3) - 1,
                   L3 + L2 + len(E3) - 1,
                   L3 + L1 + L1 + len(E1) - 1),
                  dtype=P3.dtype)
    S3[:(L3 + len(E3) - 1),
       :(L2 + len(E2) - 1),
       :(L1 + len(E1) - 1)] = P3

    tau3, tau2, _ = (int((len(E) - 1) / 2.0) for E in E_all)

    for T2 in xrange(S3.shape[1]):
        S3[:, T2, :] = np.roll(S3[:, T2, :], T2 - tau2, axis=1)
    for T3 in xrange(S3.shape[0]):
        S3[T3, :, :] = np.roll(S3[T3, :, :], T3 - tau3, axis=0)
        S3[T3, :, :] = np.roll(S3[T3, :, :], T3 - tau3, axis=1)
    return S3


def generalized_tikhonov_inverse(A, damp, deriv_penalty_order=2):
    (m, n) = A.shape
    L = damp * derivative_matrix(n, deriv_penalty_order)
    Ax = np.concatenate((A, L))

    U, s, Vh = svd(Ax, full_matrices=False)
    Ainv = (Vh.conj().T, 1 / s * np.dot(U.conj().T))[:, :m]

    H = A.dot(Vh.conj().T.dot((1 / s).reshape(-1, 1) * U.conj().T))
    gcv_k = np.trace(H).real

    return (Ainv, gcv_k)


def gcv_error(residual, len_x, gcv_k, gcv_w=1):
    return (norm(residual) / (1 - gcv_w * gcv_k / len_x)) ** 2


def ncp_error(residual):
    error = 0
    n = len(residual)
    for f in (np.real, np.imag):
        ncp = np.add.accumulate(
            np.abs(np.fft.fft(f(residual))[:int(n / 2)]) ** 2)
        ncp /= ncp[-1]
        error += np.sum(ncp - np.linspace(1. / len(ncp), 1, len(ncp))) ** 2
    return error

# def solve_inverse_problem(A, b, damp, deriv_penalty_order=1, gcv_w=1):
#     n = A.shape[1]
#     L = damp * derivative_matrix(n, deriv_penalty_order)

#     Ax = np.concatenate((A, L))
#     bx = np.append(b, np.zeros(L.shape[0]))

#     U, s, Vh = svd(Ax, full_matrices=False)
#     x = np.dot(Vh.conj().T, 1 / s * np.dot(U.conj().T, bx))

#     residual = A.dot(x) - b
#     MSPE = norm(residual) ** 2
#     x_norm = norm(x) ** 2
#     Lx_norm = norm(1 / damp * L.dot(x)) ** 2

#     H = A.dot(Vh.conj().T.dot((1 / s).reshape(-1, 1) * U.conj().T))
#     k = np.trace(H).real
#     # don't calculate CV error, since sometimes it is numerically unstable
#     # cv_error = norm(residual / (1 - np.diag(H))) ** 2
#     gcv_error = MSPE / (1 - gcv_w * k / n) ** 2

#     ncp_error = 0
#     for f in (np.real, np.imag):
#         ncp = np.add.accumulate(
#             np.abs(np.fft.fft(f(residual))[:int(n / 2)]) ** 2)
#         ncp /= ncp[-1]
#         ncp_error += np.sum(ncp - np.linspace(1. / len(ncp), 1, len(ncp))) ** 2

#     return MetaArray(x, damp=damp, MSPE=MSPE, residual=residual,
#                      Lx_norm=Lx_norm, x_norm=x_norm, ncp_error=ncp_error,
#                      gcv_error=gcv_error)


def derivative_matrix(n, deriv_order, center_discontinuous=False):
    if deriv_order == 0:
        L = np.identity(n)
    elif deriv_order == 1:
        L = -np.eye(n - 1, n) + np.eye(n - 1, n, 1)
        if center_discontinuous:
            if n % 2:
                L = np.concatenate([L[:((n - 1) / 2 - 1)], L[((n - 1) / 2 + 1):]])
            else:
                L = np.concatenate([L[:(n / 2 - 1)], L[(n / 2):]])
    elif deriv_order == 2:
        L = np.eye(n - 2, n) - 2 * np.eye(n - 2, n, 1) + np.eye(n - 2, n, 2)
        if center_discontinuous:
            if n % 2:
                L = np.concatenate([L[:((n - 1) / 2 - 2)],
                                    L[((n - 1) / 2 + 1):]])
            else:
                L = np.concatenate([L[:(n / 2 - 2)], L[(n / 2):]])
    else:
        ValueError('Deriv order not set')
    return L


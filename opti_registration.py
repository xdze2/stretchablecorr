# ===========================
#  Phase image registration
# ===========================

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.fft import fftshift, fftfreq
from scipy.signal.windows import blackman
from scipy.optimize import minimize
from numba import jit

nopython = False

@jit(nopython=nopython)
def custom_fftfreq(n):
    """
    same as numpy fftfreq function, but working with jit
    https://github.com/numpy/numpy/blob/92ebe1e9a6aeb47a881a1226b08218175776f9ea/numpy/fft/helper.py#L124-L170
    """
    val = 1.0 / n
    results = np.empty(n, dtype=np.int64)
    N = (n-1)//2 + 1
    p1 = np.arange(0, N, dtype=np.int64)
    results[:N] = p1
    p2 = np.arange(-(n//2), 0, dtype=np.int64)
    results[N:] = p2
    return results * val

@jit(nopython=nopython)
def dft_dot(A, yx):
    im2pi = 1j * 2 * np.pi
    y, x = yx
    yky = np.exp( im2pi * y * custom_fftfreq(A.shape[0]) )
    xkx = np.exp( im2pi * x * custom_fftfreq(A.shape[1]) )

    a = np.dot(A, xkx)
    a = np.dot(a, yky)
    return a / A.size


@jit(nopython=nopython)
def grad_dft(data, yx):
    im2pi = 1j * 2 * np.pi
    y, x = yx
    kx = im2pi * custom_fftfreq(data.shape[1])
    ky = im2pi * custom_fftfreq(data.shape[0])

    exp_kx = np.exp(x * kx)
    exp_ky = np.exp(y * ky)

    gradx = np.dot(data, exp_kx * kx)
    gradx = np.dot(gradx, exp_ky)

    grady = np.dot(data.T, exp_ky * ky)
    grady = np.dot(grady, exp_kx)

    return np.array([grady, gradx]) / data.size


def phase_registration_optim(A, B, phase=True, verbose=False):
    """Find translation between images A and B
    as the argmax of the phase cross corelation
    use iterative optimization

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    upsamplefactor = 1

    u = blackman(A.shape[0])
    v = blackman(A.shape[1])
    window = u[:, np.newaxis] * v[np.newaxis, :]
    if not phase:
        window = 1

    a = fftn(A * window)
    b = fftn(B * window)

    ab = a * b.conj()
    if phase:
        phase = ab / np.abs(ab)
    else:
        phase = ab
    phase_corr = ifftn(fftshift(phase),
                       s=upsamplefactor*np.array(ab.shape))
    phase_corr = np.abs( fftshift(phase_corr) )

    dx_span = fftshift( fftfreq(phase_corr.shape[1]) )*A.shape[1]
    dy_span = fftshift( fftfreq(phase_corr.shape[0]) )*A.shape[0]

    # argmax
    argmax_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
    argmax = dy_span[argmax_idx[0]], dx_span[argmax_idx[1]]

    def cost(xy, ab):
        return -np.abs(dft_dot(ab, xy))

    def jac(xy, ab):
        return -np.real(grad_dft(ab, xy))

    res = minimize(cost, argmax,
                   args=(phase, ),
                   method='BFGS',
                   tol=1e-3,
                   jac=jac)
    if verbose:
        print(res)
    return -res.x, np.trace(res.hess_inv)*np.std(A-B)*1.6


def output_cross_correlation(A, B, upsamplefactor=1, phase=True):
    """Output the cross correlation (or phase)

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]
    upsamplefactor : int, optional
        [description], by default 1
    phase : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    if phase:
        u = blackman(A.shape[0])
        v = blackman(A.shape[1])
        window = u[:, np.newaxis] * v[np.newaxis, :]
    else:
        window = 1

    a, b = fftn(A * window), fftn(B * window)
    ab = a * b.conj()
    if phase:
        ab = ab / np.abs(ab)
    phase_corr = ifftn(fftshift(ab),
                       s=upsamplefactor*np.array(ab.shape))
    phase_corr = np.abs( fftshift(phase_corr) )

    dx_span = fftshift( fftfreq(phase_corr.shape[1]) )*A.shape[1]
    dy_span = fftshift( fftfreq(phase_corr.shape[0]) )*A.shape[0]

    return dx_span, dy_span, phase_corr
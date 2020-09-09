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
    """Return the Discrete Fourier Transform sample frequencies.
    same as numpy's `fftfreq` function but working with JIT (numba)
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
    """2D Discrete Fourier Transform of `A` at position `xy`

    Parameters
    ----------
    A : 2D array
    yx : tuple of floats (y, x)

    Returns
    -------
    complex
        value DFT of `A` at position `xy`
    """
    im2pi = 1j * 2 * np.pi
    y, x = yx
    yky = np.exp(im2pi * y * custom_fftfreq(A.shape[0]))
    xkx = np.exp(im2pi * x * custom_fftfreq(A.shape[1]))

    a = np.dot(A, xkx)
    a = np.dot(a, yky)
    return a / A.size


@jit(nopython=nopython)
def grad_dft(data, yx):
    """2D Discrete Fourier Transform of `grad(TF(A))` at position `xy`

    Parameters
    ----------
    A : 2D array
    yx : tuple of floats (y, x)

    Returns
    -------
    (2, 1) array of complex numbers
        value `grad(TF(A))` at position xy
    """
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



def phase_registration_optim(A, B, phase=False, verbose=False, no_optim=False):
    """Find translation between images A and B
    as the argmax of the (phase) cross correlation
    use iterative optimization

    Parameters
    ----------
    A, B : 2D arrays
        source and targer images
    phase : bool, optional
        if True use only the phase angle, by default False
    verbose : bool, optional
        if true print debug information, by default False

    Returns
    -------
    (2, 1) nd-array
        displacement vector (u_y, u_x)   (note the order Y, X)
    tuple of floats
        error estimations
    """
    upsamplefactor = 1

    if phase:
        u = blackman(A.shape[0])
        v = blackman(A.shape[1])
        window = u[:, np.newaxis] * v[np.newaxis, :]
    else:
        window = 1

    a = fftn(A * window)
    b = fftn(B * window)

    ab = a * b.conj()
    if phase:
        ab = ab / np.abs(ab)

    phase_corr = ifftn(fftshift(ab),
                       s=upsamplefactor*np.array(ab.shape))
    phase_corr = np.abs(fftshift(phase_corr))

    dx_span = fftshift(fftfreq(phase_corr.shape[1])) * A.shape[1]
    dy_span = fftshift(fftfreq(phase_corr.shape[0])) * A.shape[0]

    # argmax
    argmax_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
    argmax = dy_span[argmax_idx[0]], dx_span[argmax_idx[1]]

    # shortcut for coarse search
    if no_optim:
        return -np.array(argmax), 0, 0, 0

    def cost(xy, ab):
        return -np.abs(dft_dot(ab, xy))

    def jac(xy, ab):
        return -np.real(grad_dft(ab, xy))

    res = minimize(cost, argmax,
                   args=(ab, ),
                   method='BFGS',
                   tol=1e-3,
                   jac=jac)
    if verbose:
        print(res)

    # if argmax_idx[0]>1 and argmax_idx[0] < A.shape[0]-1 and \
    #     argmax_idx[1]>1 and argmax_idx[1] < A.shape[1]-1:

    #     ky = phase_corr[argmax_idx[0]+1, argmax_idx[1]] +\
    #         phase_corr[argmax_idx[0]-1, argmax_idx[1]] -\
    #         2*phase_corr[argmax_idx[0], argmax_idx[1]] 

    #     kx = phase_corr[argmax_idx[0], argmax_idx[1]+1] +\
    #         phase_corr[argmax_idx[0], argmax_idx[1]-1] -\
    #         2*phase_corr[argmax_idx[0], argmax_idx[1]] 
    #     invH = -1/kx + -1/ky
    # else:
    #     invH = np.NaN
    

    #  FRAE - error estimation
    sigma_J = np.mean(A**2 + B**2) + 2*res.fun/A.size
    # np.std(phase_corr)  #  + np.sqrt(A.size)*4
    # lmbda = 1.68
    C_theta = np.trace(res.hess_inv) * sigma_J
    FRAE = np.sqrt(C_theta)
    #FRAE_px = np.sqrt( invH * sigma_J )
    cc_mean = np.mean(phase_corr)
    cc_std = np.std(phase_corr)
    z_score = (-res.fun - cc_mean)/cc_std
    z_score_px = np.abs(phase_corr - cc_mean)/cc_std
    c = np.count_nonzero( z_score_px > 0.5*z_score)
    #L = phase_corr
    #integral_breadth = np.sqrt( (np.sum(L) - np.mean(L))/np.ptp(L) )
    #  c = np.sum(phase_corr > np.min(phase_corr) + np.ptp(phase_corr)*0.7)
    return -res.x, (-res.fun, C_theta)


def output_cross_correlation(A, B, upsamplefactor=1, phase=True):
    """Output the cross correlation image (or phase)
    for verification and debug

    Parameters
    ----------
    A, B : 2D array
        source and target images
    upsamplefactor : int, optional
        use zero-padding to interpolated the CC on a finer grid, by default 1
    phase : bool, optional
        if True norm the CC by its amplitude, by default True

    Returns
    -------
    1D array
        shift X value
    1D array
        shift Y value
    2D array
        phase corr
    tuple
        argmax from the optimization
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
    phase_corr = np.abs(fftshift(phase_corr))

    dx_span = fftshift(fftfreq(phase_corr.shape[1])) * A.shape[1]
    dy_span = fftshift(fftfreq(phase_corr.shape[0])) * A.shape[0]

    # argmax
    argmax_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
    argmax = dy_span[argmax_idx[0]], dx_span[argmax_idx[1]]

    def cost(xy, ab):
        return -np.abs(dft_dot(ab, xy))

    def jac(xy, ab):
        return -np.real(grad_dft(ab, xy))

    res = minimize(cost, argmax,
                   args=(ab, ),
                   method='BFGS',
                   tol=1e-3,
                   jac=jac)

    return -dx_span, -dy_span, phase_corr, res.x

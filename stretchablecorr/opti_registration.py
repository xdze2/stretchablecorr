# ===========================
#  Phase image registration
# ===========================

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.fft import fftshift, fftfreq
from scipy.signal.windows import blackman
from scipy.optimize import minimize

try:
    from numba import jit
except ModuleNotFoundError:
    print('Warning: numba is not installed (no compilation optimization)')

    def jit(*args, **kwargs):
        def do_nothing(f):
            return f
        return do_nothing

nopython = True

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


from skimage.feature import peak_local_max

def phase_registration_optim(A, B, phase=False, verbose=False):
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

    # Error estimation
    # from Inv. Hessian :
    a_moins_b_2 = (np.mean(A) - np.mean(B))**2
    sigma2 = np.mean(A**2 + B**2) - a_moins_b_2 + 2*res.fun/A.size
    C_theta = np.trace(res.hess_inv) * sigma2

    # CRBD :
    #ux = np.diff(A, axis=1).flatten()
    #uy = np.diff(A, axis=0).flatten()
    #ux2 = np.dot(ux, ux)
    #uy2 = np.dot(uy, uy)
    #uxy2 = np.dot(ux, uy)**2
    #CRBD = sigma2 * (ux2 + uy2)/(ux2*uy2 - uxy2)

    # nbr of peaks
    peaks = peak_local_max(phase_corr-np.min(phase_corr),
                           min_distance=2, threshold_rel=0.7)
    return -res.x, (peaks.shape[0], C_theta)


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

    return -dx_span, -dy_span, phase_corr, res



import matplotlib.pylab as plt

def plot_cross_correlation(A0, B0, zoom_factor=1, upsamplefactor=1, phase=True):

    dx_span, dy_span, cross_corr, res = output_cross_correlation(A0, B0,
                                                                 upsamplefactor=upsamplefactor,
                                                                 phase=phase)
    x_opt = -res.x

    argmax_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    argmax = dy_span[argmax_idx[0]], dx_span[argmax_idx[1]]
    argmax_idx_cc = argmax_idx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.pcolor(dx_span, dy_span, np.log(cross_corr))
    ax1.plot(*x_opt[::-1], 'xr'); 
    ax1.axis('equal')
    print('x_opt:', x_opt[::-1])
    zoom_lim = int( zoom_factor*A0.shape[0]/2 )
    title = 'phase corr.' if phase else 'cross corr.'
    ax1.set_title('cross corr')
    ax1.set_xlim(np.array([-zoom_lim, zoom_lim]) + x_opt[::-1])
    ax1.set_ylim(np.array([-zoom_lim, zoom_lim]) + x_opt[::-1])

    ax2.set_title('profiles at max')
    ax2.plot(dy_span, cross_corr[argmax_idx[0], :], label='cut along x')
    ax2.plot(dx_span, cross_corr[:, argmax_idx[1]], label='cut along y'); ax2.legend()
    ax2.set_xlim(np.array([-zoom_lim, zoom_lim]) + x_opt[::-1])


    ky = -(cross_corr[argmax_idx[0]+1, argmax_idx[1]] +\
        cross_corr[argmax_idx[0]-1, argmax_idx[1]] -\
        2*cross_corr[argmax_idx[0], argmax_idx[1]] )/np.diff(dy_span).mean()**2
    x_peak = np.linspace(-1.5, 1.5, 55)
    y = np.max(cross_corr) - 0.5*ky*x_peak**2
    ax2.plot(x_peak, y, label='k FD'); ax2.legend();

    H = np.linalg.inv(res.hess_inv)
    x_peak = np.linspace(-1.7, 1.7, 55)
    y = np.max(cross_corr) - 0.5*H[0, 0]*x_peak**2
    ax2.plot(x_peak, y, label='k Hessian'); ax2.legend();

    print('ky', ky)
    print('H0', H[0, 0])

    peaks = peak_local_max(cross_corr-np.min(cross_corr), min_distance=4, threshold_rel=0.7)
    peaks_ampl = cross_corr[peaks[:, 0], peaks[:, 1]] - np.min(cross_corr)
    arg_peaks = np.argsort(peaks_ampl)

    ax1.plot(dx_span[peaks[arg_peaks, 1]],
            dy_span[peaks[arg_peaks, 0]], '-sr',
            markersize=3, linewidth=1, alpha=0.2);

    
    return -dx_span, -dy_span, cross_corr, res
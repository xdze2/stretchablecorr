# ===========================
#  Phase image registration
# ===========================
import numpy as np
from scipy.fft import fftn, ifftn
from scipy.fft import fftshift, fftfreq
from scipy.signal.windows import blackman
from scipy.optimize import minimize


def dft_tensordot(A, yx):
    im2pi = 1j * 2 * np.pi
    y, x = yx
    yky = np.exp( im2pi * y * fftfreq(A.shape[0]) )
    xkx = np.exp( im2pi * x * fftfreq(A.shape[1]) )

    a = np.tensordot(xkx, A, axes=(0, -1))
    a = np.tensordot(yky, a, axes=(0, -1))
    return a


def grad_dft(data, yx):
    im2pi = 1j * 2 * np.pi
    y, x = yx
    kx = im2pi * fftfreq(data.shape[1])
    ky = im2pi * fftfreq(data.shape[0])

    exp_kx = np.exp(x * kx)
    exp_ky = np.exp(y * ky)

    gradx = np.tensordot(exp_kx * kx, data, axes=(0, -1))
    gradx = np.tensordot(exp_ky, gradx, axes=(0, -1))

    grady = np.tensordot(exp_kx, data, axes=(0, -1))
    grady = np.tensordot(exp_ky * ky, grady, axes=(0, -1))

    return np.array([grady, gradx])


def phase_registration_optim(A, B, phase=True):
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

    a, b = fftn(A * window), fftn(B * window)

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
        return -np.abs(dft_tensordot(ab, xy))

    #def jac(xy, ab):
    #    return -np.real(grad_dft(ab, xy))

    res = minimize(cost, argmax, args=(phase, ), method='BFGS', tol=1e-3)#, jac=jac)
    return res.x, res.hess_inv, res




# FFT shift interpolation
def fft_translate(A, dx, dy):
    a = fftn(A)
    kx = fftfreq(A.shape[1])
    ky = fftfreq(A.shape[0])
    k_xy = np.meshgrid(kx, ky, indexing='ij')

    b = a*np.exp(-1j*2*np.pi*(dx*k_xy[0] + dy*k_xy[1]))

    B = np.abs( ifftn(b) ) 
    return B
# ===========================
#  Phase image registration
# ===========================
import numpy as np
from scipy.fft import fftn, ifftn
from scipy.fft import fftshift, fftfreq
from scipy.signal.windows import blackman
from scipy.optimize import minimize




# FFT shift interpolation
def fft_translate(A, dx, dy):
    a = fftn(A)
    kx = fftfreq(A.shape[1])
    ky = fftfreq(A.shape[0])
    k_xy = np.meshgrid(kx, ky, indexing='ij')

    b = a*np.exp(-1j*2*np.pi*(dx*k_xy[0] + dy*k_xy[1]))

    B = np.abs( ifftn(b) ) 
    return B
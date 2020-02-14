import numpy as np
import matplotlib.pylab as plt

from skimage import io
from skimage.feature import register_translation


def crop(I, ij_center, window_half_size):
    """Return the centered square at the position"""
    
    ij_center = np.around(ij_center).astype(np.int)
    i, j = ij_center
    i_slicing = np.s_[i - window_half_size:i + window_half_size + 1]
    j_slicing = np.s_[j - window_half_size:j + window_half_size + 1]
    
    return I[i_slicing, j_slicing]


def get_shifts(I, J, x, y,
               window_half_size=15,
               upsample_factor=20):
    """
    Cross-correlation between images I and J,
    at the position (x, y) using a windows of size 2*window_half_size + 1
    
    see `register_translation` from skimage
    # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation
    """
    print(x, y, ' '*12, end='\r')
    source = crop(I, (x, y), window_half_size)
    target = crop(J, (x, y), window_half_size)

    shifts, error, _ = register_translation(source, target,
                                            upsample_factor=upsample_factor)
    shifts = -shifts # displacement = -registration = dst - src
    return shifts[1], shifts[0], error


def build_grid(img_shape, margin, grid_spacing):
    """
    Build a regular grid
        img_shape from I.shape i.e. (Int, Int)
        margin in px
        grid_spacing in px
    """
    x_span = np.arange(margin, img_shape[1]-margin, grid_spacing)
    y_span = np.arange(margin, img_shape[0]-margin, grid_spacing)
    x_grid, y_grid = np.meshgrid(x_span, y_span)
    
    print("grid size:", "%ix%i"%(len(x_span), len(y_span)))
    print(" i.e.", len(x_span)*len(y_span), "points")
    
    return x_grid, y_grid


# Compute shifts
def compute_shifts(I, J, grid, **kargs):
    """
    Compute the shift for each point of the grid
    
    returns shift_x, shift_y, corr_errors (flatten)
    """
    x_grid, y_grid = grid
    shifts = [get_shifts(I, J, yi, xi, **kargs)
              for (xi, yi) in zip(x_grid.flatten(), y_grid.flatten())]
    print('done', ' '*20, end='\r')
    # unzip
    shift_x = np.array([dx[0] for dx in shifts])
    shift_y = np.array([dx[1] for dx in shifts])
    corr_errors = np.array([dx[2] for dx in shifts])
    return shift_x, shift_y, corr_errors


def bilinear_fit(x, y, shift_x, shift_y):
    """ Least square bilinear fit (a*x + b*y + c) on entire grid
    returns strains (eps_x, eps_y, eps_xy) and local residuals
    
    x, y               1D arrays
    shift_x, shift_y   1D arrays
    """  
    # Least Square
    ones = np.ones_like(shift_x)
    M = np.vstack([ones, x, y]).T

    p_ux, residual_x, rank, s = np.linalg.lstsq(M, shift_x, rcond=None)
    p_uy, residual_y, rank, s = np.linalg.lstsq(M, shift_y, rcond=None)
    
    # Strain
    eps_x = p_ux[1]
    eps_y = p_uy[2]
    eps_xy = p_ux[2] + p_uy[1] 
    
    # Residuals
    ux_fit = np.matmul(M, p_ux)
    uy_fit = np.matmul(M, p_uy)
    residuals = np.sqrt( (ux_fit-shift_x)**2 + (uy_fit-shift_y)**2 )
    
    return (eps_x, eps_y, eps_xy), residuals
import numpy as np
import matplotlib.pylab as plt

from skimage import io
from skimage.feature import register_translation


def colorize_image(image, intensity_low, intensity_high, cmap='viridis'):
    """ Convert intensity values to color using a colormap
        rescale values between (intensity_low, intensity_high)
    """
    image_normalized = (image.astype(np.float) - intensity_low)/(intensity_high - intensity_low)

    cm = plt.get_cmap(cmap)
    colored_image = cm(image_normalized)
    colored_image[image_normalized > 0.999] = np.array([1, 0, 0, 1])
    colored_image[image_normalized < 0.001] = np.array([0, 0, 0, 1])
    return colored_image


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
    source = crop(I, (x, y), window_half_size)
    target = crop(J, (x, y), window_half_size)

    shifts, error, _ = register_translation(source, target,
                                            upsample_factor=upsample_factor)
    shifts = -shifts  # displacement = -registration = dst - src
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
  
    print("grid size:", "%ix%i" % (len(x_span), len(y_span)))
    print(" i.e.", len(x_span)*len(y_span), "points")

    return x_grid, y_grid


# Compute shifts
def compute_shifts(I, J, grid, **kargs):
    """
    Compute the shift for each point of the grid
    
    returns shift_x, shift_y, corr_errors (flatten)
    """
    x_grid, y_grid = grid
    
    shift_x, shift_y, corr_errors = [], [], []
    
    for (xi, yi) in zip(x_grid.flatten(), y_grid.flatten()):
        sx, sy, er = get_shifts(I, J, yi, xi, **kargs)
        shift_x.append(sx)
        shift_y.append(sy)
        corr_errors.append(er)
        print(f"{xi: 5d}, {yi: 5d}: error {er}", end='\r')

    print('done', ' '*40, end='\r')

    shift_x = np.array(shift_x).reshape(grid[0].shape)
    shift_y = np.array(shift_y).reshape(grid[0].shape)
    corr_errors = np.array(corr_errors).reshape(grid[0].shape)
    return shift_x, shift_y, corr_errors


def bilinear_fit(x, y, shift_x, shift_y):
    """ Least square bilinear fit (a*x + b*y + c) on entire grid
    returns strains (eps_x, eps_y, eps_xy) and local residuals
    
    x, y               2D arrays
    shift_x, shift_y   2D arrays
    """  
    x_flat, y_flat, shift_x_flat, shift_y_flat = [u.flatten()
                                                  for u in (x, y, shift_x, shift_y)]
    
    # Least Square
    ones = np.ones_like(shift_x_flat)
    M = np.vstack([ones, x_flat, y_flat]).T

    p_ux, residual_x, rank, s = np.linalg.lstsq(M, shift_x_flat, rcond=None)
    p_uy, residual_y, rank, s = np.linalg.lstsq(M, shift_y_flat, rcond=None)
    
    # Strain
    eps_x = p_ux[1]
    eps_y = p_uy[2]
    eps_xy = p_ux[2] + p_uy[1] 
    
    # Residuals
    ux_fit = np.matmul(M, p_ux)
    uy_fit = np.matmul(M, p_uy)
    residuals_x = ux_fit.reshape(shift_x.shape) - shift_x
    residuals_y = uy_fit.reshape(shift_y.shape) - shift_y
    
    return (eps_x, eps_y, eps_xy), residuals_x, residuals_y
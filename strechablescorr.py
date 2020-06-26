# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

from skimage import io
from skimage.color import rgb2gray

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    print('Warning: scikit-image not up-to-date')
    from skimage.feature import register_translation as phase_cross_correlation


def load_image(path, verbose=True):
    """Load the image at the given path
         returns 2d array (float)
         convert to grayscale if needed
    """
    try:
        image = io.imread(path)
        # Convert to grayscale if needed:
        image = rgb2gray(image) if image.ndim == 3 else image
        image = image.astype(np.float)
        if verbose:
            print(f'load "{path}"', f"size={image.shape}")
    except FileNotFoundError:
        print("File %s Not Found" % path)
        image = None

    return image


def colorize_image(image, intensity_low, intensity_high, cmap='viridis'):
    """Convert intensity values to color using a colormap
       rescale values between (intensity_low, intensity_high)
    """
    image_normalized = (image.astype(np.float) - intensity_low)/(intensity_high - intensity_low)

    cm = plt.get_cmap(cmap)
    colored_image = cm(image_normalized)
    colored_image[image_normalized > 0.999] = np.array([1, 0, 0, 1])
    colored_image[image_normalized < 0.001] = np.array([0, 0, 0, 1])
    return colored_image


def crop(I, xy_center, half_size):
    """Return the centered square at the position xy

    Args:
        I: input image (2D array)
        xy_center: tuple of float
        half_size: half of the size of the crop

    Returns:
        cropped image array
        indices of the center


    >>> from skimage.data import rocket
    >>> x, y = (322, 150)
    >>> plt.imshow(rocket());
    >>> print(rocket().shape)
    >>> plt.plot(x, y, 'sr');
    >>> plt.imshow(crop(rocket(), (x, y), 50)[0]);
    
    # todo: unit test using hash for image
    # https://github.com/opencv/opencv/blob/e6171d17f8b22163997487b16762d09671a68597/modules/python/test/tests_common.py#L55
    """

    j, i = np.around(xy_center).astype(np.int)
    i_slicing = np.s_[i - half_size:i + half_size + 1]
    j_slicing = np.s_[j - half_size:j + half_size + 1]

    return I[i_slicing, j_slicing], (i, j)


def get_shifts(I, J, x, y,
               offset=(0.0, 0.0),
               window_half_size=15,
               upsample_factor=20):
    """Cross-correlation between images I and J,
        at the position (x, y) using a windows of size 2*window_half_size + 1

    see `phase_cross_correlation` from skimage
    # https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation
    
    Args:
      I, J: input images
      x, y: point coordinates arround which shift is evaluated
      offset: tuple (dx, dy) pre-computed displacement of J relative to I
    
    Returns:
        dx, dy: displacements
        error: scalar correlation error
        
    >>> from skimage.data import camera
    >>> dx, dy = 10, 15
    >>> I = camera()[dy:, dx:]
    >>> J = camera()[:-dy, :-dx]
    >>> plt.imshow(I+J);
    >>> print( get_shifts(I, J, 250, 250, window_half_size=150, upsample_factor=1) )
    >>> print( get_shifts(I, J, 250, 250,
                      window_half_size=150,
                      upsample_factor=1,
                      offset=(4.5, 14.2)) )
    """
    dx, dy = offset

    source, ij_src = crop(I, (x, y), window_half_size)
    target, ij_tgt = crop(J, (x+dx, y+dy), window_half_size)
  
    shifts, error, _ = phase_cross_correlation(source, target,
                                               upsample_factor=upsample_factor)
    shifts = -shifts  # displacement = -registration = dst - src

    dx = shifts[1] + (ij_tgt[1] - ij_src[1])
    dy = shifts[0] + (ij_tgt[0] - ij_src[0])
    return dx, dy, error


def build_grid(img_shape, margin, spacing):
    """Build a regular grid
    
    Args:
        img_shape: tuple shape of the image (Int, Int)
        margin: size of the margin in px
        spacing: spacing in px between points
        
    Returns:
        x_grid, y_grid: 2d arrays of coordinates
    """
    margin = int(np.ceil(margin))
    spacing = int(np.ceil(spacing))
    x_span = np.arange(0, img_shape[1]-2*margin, spacing)
    y_span = np.arange(0, img_shape[0]-2*margin, spacing)
    
    x_offset = int( (img_shape[1] - x_span[-1])/2 )
    y_offset = int( (img_shape[0] - y_span[-1])/2 )    
    
    x_grid, y_grid = np.meshgrid(x_span + x_offset, y_span + y_offset)
    
    print("grid size:", "%ix%i" % (len(x_span), len(y_span)))
    print(" i.e.", len(x_span)*len(y_span), "points")

    return x_grid, y_grid


def compute_shifts(I, J, points, **kargs):
    """Compute shifts for each point
    
    Args:
        I, J: input images (2D arrays) 
        points: tuple of (x_coord, y_coord)
        **kargs: passed to get_shifts(), window_half_size, upsample_factor
         
    Returns:
        shift_x, shift_y, corr_errors (flatten)
    """
    x_grid, y_grid = points
    
    shift_x, shift_y, corr_errors = [], [], []
    
    for k, (xi, yi) in enumerate(zip(x_grid.flatten(), y_grid.flatten())):
        sx, sy, er = get_shifts(I, J, xi, yi, **kargs)
        shift_x.append(sx)
        shift_y.append(sy)
        corr_errors.append(er)
        print(f" {k: 4d}/{len(x_grid.flatten())}:  {sx:.2f} {sy:.2f}  error {er}", end='\r')

    print('done', ' '*40, end='\r')

    shift_x = np.array(shift_x).reshape(points[0].shape)
    shift_y = np.array(shift_y).reshape(points[1].shape)
    corr_errors = np.array(corr_errors).reshape(points[0].shape)
    return shift_x, shift_y, corr_errors


def get_displacement_from_ref(cube, x, y, reference_image,
                            window_half_size, upsample_factor,
                            verbose=True):
    """Find displacement for each images relative to the reference frame
        starting from the point (x, y) in the reference frame
        
        (Lagrangian)
    """
    I_ref = cube[:, :, reference_image]
    disp_to_ref = np.zeros((cube.shape[2], 2))
    
    # use the previous estimated position as offset
    # Forward, image_by_image
    dx_ref, dy_ref = 0, 0
    for k in range(reference_image+1, cube.shape[2]):
        J = cube[:, :, k]
        try:
            dx_ref, dy_ref, error = get_shifts(I_ref, J, x, y,
                                       offset=(dx_ref, dy_ref),
                                       window_half_size=window_half_size,
                                       upsample_factor=upsample_factor)
            disp_to_ref[k] = [dx_ref, dy_ref]

            #previous = cube[:, :, k-1]
            #dx1, dy1, error = get_shifts(previous, J, x+dx_ref, y+dy_ref,
            #                               offset=(0, 0),
            #                               window_half_size=window_half_size,
            #                               upsample_factor=upsample_factor)
            #step1[k] = [dx1, dy1]
        except ValueError:
            if verbose:
                print('out of limits for image', k)
            disp_to_ref[k] = [np.NaN, np.NaN]

    # Backward, image_by_image
    dx_ref, dy_ref = 0, 0
    for k in range(0, reference_image)[::-1]:
        J = cube[:, :, k]
        try:
            dx_ref, dy_ref, error = get_shifts(I_ref, J, x, y,
                                               offset=(dx_ref, dy_ref),
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor)
            disp_to_ref[k] = [dx_ref, dy_ref]
        except ValueError:
            print('out of limits for image', k)
            step1[k] = [np.NaN, np.NaN]
            
    return disp_to_ref


def get_displacement_from_previous(cube, x, y,
                                   window_half_size, upsample_factor,
                                   offsets = None,
                                   verbose=True):
    """Find displacement for each images relative to the previous frame
        at the point (x, y) in the camera reference frame
        
        (Eulerian)
    """
    I_ref = cube[:, :, 0]
    disp_to_previous = np.zeros((cube.shape[2], 2))

    if offsets is not None:
        dx_ref, dy_ref = offsets[0, :]
    else:
        dx_ref, dy_ref = 0, 0

    for k in range(1, cube.shape[2]):
        J = cube[:, :, k]
        try:
            if offsets is not None:
                dx_guess = dx_ref - offsets[k-1, 0] + offsets[k, 0]
                dy_guess = dy_ref - offsets[k-1, 1] + offsets[k, 1]
            else:
                dx_guess, dy_guess = dx_ref, dy_ref

            dx_ref, dy_ref, error = get_shifts(I_ref, J, x, y,
                                               offset=(dx_guess, dy_guess),
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor)
            disp_to_previous[k] = [dx_ref, dy_ref]

        except ValueError:
            if verbose:
                print('out of limits for image', k)
            disp_to_previous[k] = [np.NaN, np.NaN]

        I_ref = J

    return disp_to_previous


def bilinear_fit_old(x, y, shift_x, shift_y):
    """Least square bilinear fit (a*x + b*y + c) on entire grid
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


# ===============
#  Bilinear Fit
# ===============

def bilinear_fit(points, displacements):
    # Least Square
    u, v = displacements.T
    mask = ~np.isnan(u) & ~np.isnan(v) 
    u, v = u[mask], v[mask]

    x, y = points[mask, :].T

    ones = np.ones_like(x)
    M = np.vstack([x, y, ones]).T

    p_ux, residual_x, rank, s = np.linalg.lstsq(M, u, rcond=None)
    p_uy, residual_y, rank, s = np.linalg.lstsq(M, v, rcond=None)

    p = np.vstack([p_ux, p_uy])
    # np.linalg.inv(np.matmul(M.T, M))

    # unbiased estimator variance (see p47 T. Hastie)
    sigma_hat_x = np.sqrt(residual_x/(M.shape[0]-M.shape[1]-1))
    sigma_hat_y = np.sqrt(residual_y/(M.shape[0]-M.shape[1]-1))

    # Residus 
    u_linear = np.matmul( M, p_ux )
    v_linear = np.matmul( M, p_uy )

    residus_x = u - u_linear
    residus_y = v - v_linear

    residus_xy = np.vstack([residus_x, residus_y]).T
    
    a = np.full(displacements.shape, np.nan)
    a[mask, :] = residus_xy
    return p, a  #(sigma_hat_x, sigma_hat_y)



if __name__ == "__main__":
    import doctest
    doctest.testmod()



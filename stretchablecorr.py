# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt


try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    print('Warning: scikit-image not up-to-date')
    from skimage.feature import register_translation as phase_cross_correlation

from opti_registration import phase_registration_optim


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
    """Returns the centered square at the position xy
    rounds xy_center to nearest integers first

    Parameters
    ----------
    I : 2D array
        input image
    xy_center : tuple of floats
        central coordinates (will be rounded)
    half_size : integer
        half size of the cropped region.
        The actuak size is `(2*half_size + 1)`

    Returns
    -------
    2D array
        cropped image array
    tuple of integers
        indices of the actual center

    Examples
    --------
    >>> from skimage.data import rocket
    >>> x, y = (322, 150)
    >>> plt.imshow(rocket());
    >>> print(rocket().shape)
    >>> plt.plot(x, y, 'sr');
    >>> plt.imshow(crop(rocket(), (x, y), 50)[0]);


    .. todo:: unit test using hash for image https://github.com/opencv/opencv/blob/e6171d17f8b22163997487b16762d09671a68597/modules/python/test/tests_common.py#L55

    """

    j, i = np.around(xy_center).astype(np.int)
    i_slicing = np.s_[i - half_size:i + half_size + 1]
    j_slicing = np.s_[j - half_size:j + half_size + 1]

    I_crop = I[i_slicing, j_slicing]

    if I_crop.shape[:2] != (2*half_size+1, 2*half_size+1):
        raise ValueError("crop out of image bounds", I_crop.shape)

    return I_crop, (i, j)


def get_shifts(I, J, x, y,
               window_half_size,
               offset=(0.0, 0.0),
               method='skimage',
               coarse_search=True, **params):
    """Interface to perform local shifts estimation between images I and J,
       at position (x, y) using a window of size `2*window_half_size + 1`

    Available methods:  
    - 'skimage': see `phase_cross_correlation` from skimage
    https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation

    - 'opti': use iterative optimization to find the maximum

    Parameters
    ----------
    I, J : 2D arrays
        input images
    x, y : tuple
        point coordinates arround which shift is evaluated
    offset : tuple (dx, dy)
        pre-computed displacement of J relative to I

    Returns
    -------
    dx, dy
        displacements
    error
        scalar correlation error

    Examples
    --------
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

    if coarse_search:
        coarse_window_half_size = 100 # 3*window_half_size
        x_margin = int(min(x, I.shape[1]-x))
        y_margin = int(min(y, I.shape[0]-y))
        coarse_window_half_size = min(coarse_window_half_size, x_margin, y_margin)
        source, ij_src = crop(I, (x, y), coarse_window_half_size)
        target, ij_tgt = crop(J, (x+dx, y+dy), coarse_window_half_size)
        shifts = phase_cross_correlation(source, target,
                                         upsample_factor=1,
                                         return_error=False)
        shifts = -shifts  # displacement = -registration = dst - src
        dx += shifts[1]
        dy += shifts[0]

    source, ij_src = crop(I, (x, y), window_half_size)
    target, ij_tgt = crop(J, (x+dx, y+dy), window_half_size)

    if method == 'skimage':
        shifts, error, _ = phase_cross_correlation(source, target,
                                                   **params)
        shifts = -shifts  # displacement = -registration = dst - src
    elif method == 'opti':
        shifts, error = phase_registration_optim(source, target,
                                                 **params)
    else:
        raise TypeError("method must be 'skimage' or 'opti'")

    dx = shifts[1] + (ij_tgt[1] - ij_src[1])
    dy = shifts[0] + (ij_tgt[0] - ij_src[0])

    return dx, dy, error


def build_grid(img_shape, margin, spacing):
    """Build a centered regular grid

    note: as given by `np.meshgrid`

    Parameters
    ----------
    img_shape : tuple (height, width)
        size of the image for which the grid will be used
    margin : Int or Float
        minimal distance to image edges without points
    spacing : Int or Float
        distance in pixel between points

    Returns
    -------
    3D nd-array of floats, shape (2, nbr pts height, width)
       grid[0]: X coordinates of grid points
       grid[1]: Y coordinates of grid points
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

    return np.stack((x_grid, y_grid))


# ==========================
#  Loops for displacements
# ==========================

def displacements_img_to_img(images, points,
                             window_half_size, upsample_factor,
                             offsets=None,
                             verbose=True):
    """Eulerian image-to-image correlation
    i.e. at position (points) fixed relative to the camera frame

    Parameters
    ----------
    images : iterable
        sequence of images
    points : iterable of point coordinates [[x1, y1], ...]
        positions where displacement is computed
    window_half_size : integer
        size in pixel of the square area used for correlation
        actual size is (2w + 1)
    upsample_factor : integer
        wanted accuracy of the correlation
        see doc. of scikit-image phase-cross-correlation
    offsets : float array, optional
        could be an 2D array (nbr_images-1, 2)
        of a 3D array (nbr_images-1, nbr_points, 2)
        by default zeros (None)
    verbose : bool, optional
        print information

    Returns
    -------
    3d array of shape (nbr_images-1, nbr_points, 2)
        Displacement vector.
        NaN if an error occured (often because ROI out of image)
    """

    params = {'window_half_size':window_half_size,
              'upsample_factor':upsample_factor}

    if offsets is None:
        offsets = np.zeros((len(images)-1, len(points), 2))
    elif len(offsets.shape)==2:
        offsets = np.tile(offsets[:, np.newaxis, :], (1, len(points), 1))
        print(offsets.shape)

    displ = np.empty((len(images)-1,
                      len(points),
                      2))
    displ[:] = np.NaN

    N = (len(images) - 1)*len(points)
    for k, (A, B) in enumerate(zip(images, images[1:])):
        for i, xyi in enumerate(points):
            try:
                sx, sy, _err = get_shifts(A, B, *xyi,
                                        offset=offsets[k, i, :],
                                        **params)

                displ[k, i, :] = sx, sy
            except ValueError:
                pass

            if verbose:
                print(f'{int(100*(k*len(points)+i))//N: 3d}%'+
                      f'  images:{k:02d}→{k+1:02d}'+
                      f'  point:{i: 4d} ...',
                      end='\r')

    print('done', ' '*30)
    return displ


def track_displ_img_to_img(images, start_points,
                           offsets=None,
                           verbose=True, **params):
    #params = {'window_half_size':window_half_size,
    #          'upsample_factor':upsample_factor}

    if offsets is None:
        offsets = np.zeros((len(images)-1, len(start_points), 2))
    elif len(offsets.shape)==2:
        offsets = np.tile(offsets[:, np.newaxis, :], (1, len(start_points), 1))
        print(offsets.shape)

    displ = np.empty((len(images)-1,
                      len(start_points),
                      2))
    displ[:] = np.NaN

    errors = np.empty((len(images)-1,
                      len(start_points)))
    errors[:] = np.NaN

    N = (len(images) - 1)*len(start_points)
    for i, (x0, y0) in enumerate(start_points):
        xi, yi = x0, y0
        for k, (A, B) in enumerate(zip(images, images[1:])):

            if verbose:
                print(f'{int(100*(i*(len(images)-1)+k))//N: 3d}%'+
                      f'  images:{k:02d}→{k+1:02d}'+
                      f'  point:{i: 4d} ...',
                      end='\r')

            try:
                sx, sy, _err = get_shifts(A, B, xi, yi,
                                          offset=offsets[k, i, :],
                                          **params)

                displ[k, i, :] = sx, sy
                errors[k, i] = _err
                xi += sx
                yi += sy
            except ValueError:
                #if verbose:
                #    print('out of limits for image', k)
                break

    print('done', ' '*30)
    return displ, errors


# broken:
def track_displ_img_to_ref(images, start_points,
                           offsets=None,
                           verbose=True, **params):
    #params = {'window_half_size':window_half_size,
    #          'upsample_factor':upsample_factor,
    #          'method':method}

    if offsets is None:
        offsets = np.zeros((len(images)-1, len(start_points), 2))
    elif len(offsets.shape)==2:
        offsets = np.tile(offsets[:, np.newaxis, :], (1, len(start_points), 1))
        print(offsets.shape)

    displ = np.empty((len(images)-1,
                      len(start_points),
                      2))
    displ[:] = np.NaN

    errors = np.empty((len(images)-1,
                      len(start_points)))
    errors[:] = np.NaN
    A = images[0]
    N = (len(images) - 1)*len(start_points)
    for i, (x0, y0) in enumerate(start_points):
        xi, yi = x0, y0
        for k, B in enumerate(images[1:]):

            if verbose:
                print(f'{int(100*(i*(len(images)-1)+k))//N: 3d}%'+
                      f'  images:{k:02d}→{k+1:02d}'+
                      f'  point:{i: 4d} ...',
                      end='\r')

            try:
                sx, sy, _err = get_shifts(A, B, xi, yi,
                                          offset=offsets[k, i, :],
                                          **params)

                displ[k, i, :] = sx, sy
                errors[k, i] = _err
                xi += sx
                yi += sy
            except ValueError:
                #if verbose:
                #    print('out of limits for image', k)
                break

    print('done', ' '*30)
    return displ, errors


# ===============
#  Bilinear Fit
# ===============

def bilinear_fit(points, displacements):
    """Performs a bilinear fit on the displacements field

    Solve the equation u = A*x + t

    Parameters
    ----------
    points : nd-array (nbr_points, 2)
        coordinates of points (x, y)
    displacements : nd-array (nbr_points, 2)
        displacement for each point (u, v)
        could include NaN

    Returns
    -------
    nd-array (2, 3)
        coefficients matrix (affine transformation + translation)
    nd-array (nbr_points, 2)
        residuals for each points
    """
    u, v = displacements.T
    mask = np.logical_not(np.logical_or(np.isnan(u), np.isnan(v)))
    u, v = u[mask], v[mask]
    x, y = points[mask, :].T

    ones = np.ones_like(x)
    M = np.vstack([x, y, ones]).T

    p_uy, _residual_y, _rank, _s = np.linalg.lstsq(M, v, rcond=None)
    p_ux, _residual_x, _rank, _s = np.linalg.lstsq(M, u, rcond=None)

    coefficients = np.vstack([p_ux, p_uy])

    ## Unbiased estimator variance (see p47 T. Hastie)
    #sigma_hat_x = np.sqrt(residual_x/(M.shape[0]-M.shape[1]-1))
    #sigma_hat_y = np.sqrt(residual_y/(M.shape[0]-M.shape[1]-1))

    # Residuals:
    u_linear = np.matmul( M, p_ux )
    v_linear = np.matmul( M, p_uy )

    residuals_x = u - u_linear
    residuals_y = v - v_linear

    residuals_xy = np.vstack([residuals_x, residuals_y]).T

    # Merge with ignored NaN values:
    residuals_NaN = np.full(displacements.shape, np.nan)
    residuals_NaN[mask, :] = residuals_xy

    return coefficients, residuals_NaN



if __name__ == "__main__":
    import doctest
    doctest.testmod()



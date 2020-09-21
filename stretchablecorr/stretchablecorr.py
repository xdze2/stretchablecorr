# -*- coding: utf-8 -*-
import numpy as np

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    print('Warning: scikit-image not up-to-date')
    from skimage.feature import register_translation as phase_cross_correlation

from skimage.transform.pyramids import pyramid_reduce

from .opti_registration import phase_registration_optim
from .opti_registration import coarse_peak_search

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
    """

    j, i = np.around(xy_center).astype(np.int)
    i_slicing = np.s_[i - half_size:i + half_size + 1]
    j_slicing = np.s_[j - half_size:j + half_size + 1]

    I_crop = I[i_slicing, j_slicing]

    if I_crop.shape[:2] != (2*half_size+1, 2*half_size+1):
        raise ValueError("crop out of image bounds", I.shape, xy_center)

    return I_crop, (i, j)


def get_shifts(I, J, x, y,
               window_half_size,
               offset=(0.0, 0.0),
               method='skimage',
               coarse_search=False, **params):
    """Interface to registration methods

    Available methods:
        - 'skimage': see [`phase_cross_correlation`](https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation) from skimage
        - 'opti': use iterative optimization to find the maximum (function `phase_registration_optim`)

    Parameters
    ----------
    I, J : 2D arrays
        input images
    x, y : tuple
        point coordinates arround which shift is evaluated
    window_half_size: int
        half-size of the square region centered on (x, y) used for registration
    offset : tuple (dx, dy), default (0, 0)
        pre-computed displacement of J relative to I
    method : string {'skimage', 'opti'}
        name of method used
    coarse_search : Bool, default False
        if True perform a first registration
        on a larger region (100px) to find offset
    params : other paramters
        passed to the registration method

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
    >>> print(get_shifts(I, J, 250, 250, window_half_size=150, upsample_factor=1))
    >>> print(get_shifts(I, J, 250, 250,
                         window_half_size=150,
                         upsample_factor=1,
                         offset=(4.5, 14.2)) )
    """
    dx, dy = offset

    if coarse_search:
        coarse_window_half_size = 70  #  3*window_half_size
        x_margin = int(min(x, I.shape[1]-x))
        y_margin = int(min(y, I.shape[0]-y))
        coarse_window_half_size = min(coarse_window_half_size,
                                      x_margin,
                                      y_margin)
        A0, ij0_src = crop(I, (x, y), coarse_window_half_size)
        B0, ij0_tgt = crop(J, (x+dx, y+dy), coarse_window_half_size)

        offsets, _percent = coarse_peak_search(A0, B0,
                                               phase=False,
                                               threshold_rel=0.9)
        if  len(offsets)>1:
            print('len(offsets)', len(offsets), '---', end='\r')

        A1, ij_src = crop(I, (x, y), window_half_size)
        res = []
        for offset in offsets:
            offset = offset + np.array(ij0_tgt) - np.array(ij0_src)
            xy_with_offset = (x+dx+offset[1], y+dy+offset[0])
            B1, ij_tgt = crop(J, xy_with_offset, window_half_size)
            shift, err = phase_registration_optim(A1, B1, phase=False)
            res.append((shift + ij_tgt - ij_src, err))
            
        shifts, errors = min(res, key=lambda x:x[1][0])

        #shifts, *_ = phase_registration_optim(source, target, phase=True)

        dx = shifts[1]
        dy = shifts[0]
        return np.array((dx, dy)), errors

    source, ij_src = crop(I, (x, y), window_half_size)
    target, ij_tgt = crop(J, (x+dx, y+dy), window_half_size)

    if method == 'skimage':
        shifts, *errors = phase_cross_correlation(source, target,
                                                  **params)
        shifts = -shifts  # displ. = -registration = dst - src
    elif method == 'opti':
        shifts, errors = phase_registration_optim(source, target,
                                                  **params)
    else:
        raise TypeError("method must be 'skimage' or 'opti'")

    dx = shifts[1] + (ij_tgt[1] - ij_src[1])
    dy = shifts[0] + (ij_tgt[0] - ij_src[0])

    return np.array((dx, dy)), errors


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

    x_offset = int((img_shape[1] - x_span[-1])/2)
    y_offset = int((img_shape[0] - y_span[-1])/2)

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

    .. error:: broken

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
        could be an 2D array (nbr_images-1, 2),
        or a 3D array (nbr_images-1, nbr_points, 2),
        by default zeros (None)
    verbose : bool, optional
        print information

    Returns
    -------
    3D array of shape (nbr_images-1, nbr_points, 2)
        Displacement vector.
        NaN if an error occured (often because ROI out of image)
    """

    params = {'window_half_size': window_half_size,
              'upsample_factor':  upsample_factor}

    if offsets is None:
        offsets = np.zeros((len(images)-1, len(points), 2))
    elif len(offsets.shape) == 2:
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
    """Lagrangian image-to-image correlation
    i.e. track points on the sample surface

    Parameters
    ----------
    images : iterable
        sequence of images
    start_points : iterable of point coordinates [[x1, y1], ...]
         starting positions of trajectories
    offsets : float array, optional
        could be an 2D array (nbr_images-1, 2), 
        or a 3D array (nbr_images-1, nbr_points, 2), 
        by default zeros (None)
    verbose : bool, optional
        print information if True (default)
    **params : other parameters
        passed to the `get_shifts` function and then to the registration method

    Returns
    -------
    3D array of shape (nbr_images-1, nbr_points, 2)
        Displacement vector.
        NaN if an error occured (often because ROI out of image)
    """

    if verbose:
        print('Compute image-to-image Lagrangian displacement field:')

    if offsets is None:
        offsets = np.zeros((len(images)-1, len(start_points), 2))
    elif len(offsets.shape) == 2:
        offsets = np.tile(offsets[:, np.newaxis, :], (1, len(start_points), 1))
        print(offsets.shape)

    displ = np.empty((len(images)-1,
                      len(start_points),
                      2))
    displ[:] = np.NaN

    errors = np.empty((len(images)-1,
                      len(start_points),
                      2))
    errors[:] = np.NaN

    nbr_steps = (len(images) - 1)*len(start_points)
    for i, (x0, y0) in enumerate(start_points):
        xi, yi = x0, y0
        for k, (A, B) in enumerate(zip(images, images[1:])):

            if verbose:
                print(f'{int(100*(i*(len(images)-1)+k))//nbr_steps: 3d}%' +
                      f'  images:{k:02d}→{k+1:02d}' +
                      f'  point:{i: 4d} ...',
                      end='\r')

            if np.isnan(xi) or np.isnan(yi) or np.any(np.isnan(offsets[k, i, :])):
                # pass -> return NaN value
                break

            try:
                u, err = get_shifts(A, B, xi, yi,
                                    offset=offsets[k, i, :],
                                    **params)

                displ[k, i, :] = u
                errors[k, i] = err
                xi += u[0]
                yi += u[1]
            except ValueError:
                # pass -> return NaN value
                break

    if verbose:
        print('done', ' '*30)

    return displ, errors


def track_displ_2steps(cube, points, offsets=None, **params):
    displ1, _err1 = track_displ_img_to_img(cube, points,
                                           offsets=offsets,
                                           **params)

    if offsets is not None:
        offsets = displ1
        offsets_2steps = offsets[0:-1] + offsets[1:]
        offsets_2a = offsets_2steps[0::2]
        offsets_2b = offsets_2steps[1::2]
    else:
        offsets_2a = None
        offsets_2b = None

    displ2a, _err2a = track_displ_img_to_img(cube[0::2], points,
                                             offsets=offsets_2a,
                                             **params)
    displ2b, _err2b = track_displ_img_to_img(cube[1::2], points,
                                             offsets=offsets_2b,
                                             **params)

    displ2 = np.zeros((displ2a.shape[0] + displ2b.shape[0],
                       displ2b.shape[1],
                       displ2b.shape[2]))

    displ2[0::2] = displ2a
    displ2[1::2] = displ2b
    err2 = np.zeros_like(displ2)
    err2[0::2] = _err2a
    err2[1::2] = _err2b

    triangle_gap = displ1[:-1] + displ1[1:] - displ2
    triangle_gap = np.sqrt(np.sum(triangle_gap**2, axis=-1))

    return displ1, triangle_gap, _err1, err2


def track_displ_img_to_ref(images, start_points,
                           offsets=None,
                           verbose=True, **params):
    """
    
    .. error:: broken

    Parameters
    ----------
    images : [type]
        [description]
    start_points : [type]
        [description]
    offsets : [type], optional
        [description], by default None
    verbose : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    # params = {'window_half_size':window_half_size,
    #          'upsample_factor':upsample_factor,
    #          'method':method}

    if offsets is None:
        offsets = np.zeros((len(images)-1, len(start_points), 2))
    elif len(offsets.shape) == 2:
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



if __name__ == "__main__":
    import doctest
    doctest.testmod()

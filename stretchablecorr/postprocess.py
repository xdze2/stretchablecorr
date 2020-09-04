
import numpy as np


def integrate_displacement(displ_img_to_img):
    """Sum the image-to-image displacement value to
    obtain image-to-reference displacement,
    add zeros at the begining

    Parameters
    ----------
    displ_img_to_img : 3D array
        3D array of shape `(nbr images - 1, nbr points, 2)`

    Returns
    -------
    3D array of shape `(nbr images, nbr points, 2)`
    """
    # add zeros at the begining
    zeros = np.zeros_like(displ_img_to_img[0])[np.newaxis, :, :]
    displ_zero = np.concatenate([zeros, displ_img_to_img], axis=0)

    displ_image_to_ref = np.cumsum(displ_zero, axis=0)
    return displ_image_to_ref


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


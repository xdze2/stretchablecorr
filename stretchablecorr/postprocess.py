
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


def cellcentered_diff_2D(u, v):
    """
    for a given 2D vector field [u, v](x, y) sampled on a grid
    returns the centered finite difference for each cell
    
    Cell abcd:
        a───b
        │ + │
        c───d

    du_x = (ub+ud)/2 - (ua+uc)/2
    du_y = (ua+ub)/2 - (uc+ud)/2
    """
    u_center_y = 0.5*(u[1:, :] + u[:-1, :])
    u_center_x = 0.5*(u[:, 1:] + u[:, :-1])
    v_center_y = 0.5*(v[1:, :] + v[:-1, :])
    v_center_x = 0.5*(v[:, 1:] + v[:, :-1])

    delta_u_x = u_center_y[:, 1:] - u_center_y[:, :-1]
    delta_u_y = u_center_x[1:, :] - u_center_x[:-1, :]

    delta_v_x = v_center_y[:, 1:] - v_center_y[:, :-1]
    delta_v_y = v_center_x[1:, :] - v_center_x[:-1, :]
    
    return delta_u_x, delta_u_y, delta_v_x, delta_v_y


def cellcentered_grad_rect2D(xgrid, ygrid, u, v):
    """Finite difference gradient for the vector fields u and v
    evaluated at cell center
    
    This is not a proper bilinear interpolation (ie. quad4 element).
    The xy-grid has to be rectangular.

    used to computed the "Displacement gradient tensor"
    see Bower p.14

    output: (dudx, dudy), (dvdx, dvdy)
    """
    du_x, du_y, dv_x, dv_y = cellcentered_diff_2D(u, v)
    dx, _ydx, _xdy, dy = cellcentered_diff_2D(xgrid, ygrid)
    
    return [[du_x/dx, du_y/dy],
            [dv_x/dx, dv_y/dy]]


# --- test cellcentered_grad_rect2D
xgrid, ygrid = np.meshgrid(np.linspace(-1, 1, 5)**2,
                           np.linspace(1,  5, 7)**0.5)
u = 5*xgrid + 3*ygrid
v = 2*xgrid + 7*ygrid

(dudx, dudy), (dvdx, dvdy) = cellcentered_grad_rect2D(xgrid, ygrid, u, v)

np.testing.assert_almost_equal(dudx, 5*np.ones_like(dudx))
np.testing.assert_almost_equal(dudy, 3*np.ones_like(dudx))
np.testing.assert_almost_equal(dvdx, 2*np.ones_like(dudx))
np.testing.assert_almost_equal(dvdy, 7*np.ones_like(dudx))
# ---


def get_LagrangeStrainTensor(xgrid, ygrid, u, v):
    """Lagrange Strain Tensor (E)

        F = grad(u) + Id 
        E = 1/2*( FF^T - Id )

    Parameters
    ----------
    xgrid, ygrid : 2d arrays of shape (n_y, n_x)
        underformed grid points
    u, v : 2d arrays of shape (n_y, n_x)
        displacements values (u along x, v along y)

    Returns
    -------
    4D array of shape (n_y, n_x, 2, 2)
        Lagrange Strain Tensor for all grid points
    """
    grad_u, grad_v = cellcentered_grad_rect2D(xgrid, ygrid, u, v)

    grad_u = np.stack(grad_u, axis=2)
    grad_v = np.stack(grad_v, axis=2)

    # u = 1*xgrid + 3*ygrid
    # v = 5*xgrid + 7*ygrid
    G = np.stack([grad_u, grad_v], axis=3)
    G = np.transpose(G, axes=(0, 1, 3, 2))
    # G >>> array([[1., 3.],  [5., 7.]])

    Id = np.ones((*grad_u.shape[:2], 2, 2))
    Id[:, :] = np.eye(2, 2)
    # Id[0, 0] >> array([[1., 0.], [0., 1.]])

    F = G + Id

    # Lagrange Strain Tensor
    E = 0.5*( np.einsum('...ki,...kj', F, F) - Id )
    return E


# --- test get_LagrangeStrainTensor
xgrid, ygrid = np.meshgrid(np.linspace(-1, 1, 5),
                           np.linspace(1,  5, 7))
u = 1*xgrid + 3*ygrid
v = 5*xgrid + 7*ygrid

E = get_LagrangeStrainTensor(xgrid, ygrid, u, v)

# array([[[[14., 23.],
#          [23., 36.]],
np.testing.assert_almost_equal(E[:, :, 0 ,0], 14*np.ones_like(E[:, :, 0 ,1]))
np.testing.assert_almost_equal(E[:, :, 0 ,1], 23*np.ones_like(E[:, :, 0 ,1]))
np.testing.assert_almost_equal(E[:, :, 1 ,1], 36*np.ones_like(E[:, :, 0 ,1]))
np.testing.assert_almost_equal(E[:, :, 1 ,0], 23*np.ones_like(E[:, :, 0 ,1]))
# ---


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



# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pylab as plt
from skimage import io
from skimage import img_as_uint
from strechablecorr import *
# #!pip install scikit-image

import filetools as ft
import os, imghdr

# %load_ext autoreload
# %autoreload 2

# # Strechable Corr

# ## Search and select images

# Available samples list
input_data_dir = "./images/"
samples = os.listdir(input_data_dir)
print('Available samples')
print('=================')
ft.print_numbered_list(samples)

# Select a sample:
sample_id = input('Select an image directory:')
sample_name = samples[int(sample_id)]
print(sample_name)

# +
# ==================
#  Load image cube
# ==================
print('Loading image cube...', end='\r')

# List, select and sort images
sample_input_dir = os.path.join(input_data_dir, sample_name)
print(f'Load "{sample_name}" from {sample_input_dir}')

cube = ft.load_images(ft.list_images(sample_input_dir))
# -

# Create output directory
output_dir = './output/'
ft.create_dir(output_dir)
sample_output_path = os.path.join(output_dir, sample_name)
ft.create_dir(sample_output_path)
print('create outpur directory:', sample_output_path)

plt.figure(); plt.title(f'cube std - {sample_name}');
plt.imshow(np.std(cube, axis=0));
plt.savefig(os.path.join(sample_output_path, '01_cube_std.svg'));

cube.shape



# +
# ==================
#  Define the grid
# ==================

window_half_size = 60

grid_spacing = 150 //3
grid_margin = 350 //3

upsample_factor = 100

reference_image = 0

# ----
grid = build_grid(cube.shape[1:], margin=grid_margin, spacing=grid_spacing)
points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )

# Graph the grid
show_pts_number = False
show_window = True

plt.figure();
plt.imshow(cube[reference_image, :, :]);
plt.title(f'Grille - D={grid_spacing}px - {points.shape[0]} points');
plt.plot(*grid, 'o', color='white', markersize=3);

if show_pts_number:
    for k, (x, y) in enumerate(points):
        if len(points) > 10 and k % 5 != 0:
            continue
        text_offset = 10.0
        plt.text(x+text_offset, y+text_offset,
                 str(k), fontsize=8, color='white')
    
if show_window:
    # graph one of the ROI
    box = np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])*(window_half_size + 1)
    middle_point = tuple(np.array(grid[0].shape) // 2 - 1)
    plt.plot(box[0]+grid[0][middle_point], box[1]+grid[1][middle_point],
             color='white', linewidth=1)

plt.savefig(os.path.join(sample_output_path, '02_grid.svg'));

# +
# ============================================
#  Compute image to image displacement field
# ============================================

# 1. get image-to-image offsets
xy_center = cube.shape[1]//2, cube.shape[2]//2
central_window_halfsize = min(cube.shape[1]//3, cube.shape[2]//2) // 2
offsets = get_displacement_from_previous(cube, *xy_center, central_window_halfsize,
                                         upsample_factor=1,
                                         verbose=False)

print(f' the largest image-to-image offset is {int(np.max(np.abs(offsets))):d}px')

# +
# 2. get image-to-image displacements
    # shape: (Nbr points, 2, nbr_frames-1)
upsample_factor = 1
displ_from_previous = np.zeros((points.shape[0], cube.shape[0]-1, 2))
for point_id, coords in enumerate(points):
    print('Compute image-to-image displacement field:',
          f'{point_id: 4d}/{len(points)}',
          end='\r')
    displ_from_previous[point_id] = get_displacement_from_previous(cube, *coords, 
                                                                  window_half_size,
                                                                  upsample_factor,
                                                                  offsets=offsets,
                                                                  verbose=False)

# set dim order to (image_id, point_id, uv)
displ_from_previous = displ_from_previous.swapaxes(0, 1)
print('Compute image-to-image displacement field:',
      'done', ' '*10)

# +
all_coeffs = []
all_residuals = []
for displ_field in displ_from_previous:
    coeffs, residuals = bilinear_fit(points, displ_field)
    all_coeffs.append(coeffs)
    all_residuals.append(residuals)

linear_def_from_previous = np.stack(all_coeffs, axis=0)
residuals_from_previous = np.stack(all_residuals, axis=0)


# -

def plot_vector_field(points, displacements,
                      view_factor=None, color='white'):
    amplitudes = np.sqrt(np.nansum( displacements**2, axis=1 )) # disp. amplitude

    mask = ~np.any( np.isnan(displacements), axis=1, keepdims=False )
    
    plt.quiver(*points[mask, :].T, *displacements[mask, :].T,
               angles='xy', color=color,
               scale_units='xy',
               scale=1/view_factor if view_factor else None,
               minlength=1e-4);
    
    plt.text(10., 10.,
             f'max(|u|)={np.nanmax(amplitudes):.2f}px  mean(|u|)={np.nanmean(amplitudes):.2f}px',
             fontsize=12, color=color,
             verticalalignment='top')


# +
# =========
#  Graphs
# =========
output_dir = f'frame_to_frame_window{window_half_size}px'
image_ext = "svg"

# --
save_path = os.path.join(sample_output_path, output_dir)
ft.create_dir(save_path)

# +
# 1. Champ de déplacement
for image_id, displ in enumerate(displ_from_previous):
    plt.figure();
    plt.imshow(cube[image_id]);
    plot_vector_field(points, displ, view_factor=None)
    plt.title(f'champ de déplacement - images {image_id}→{image_id+1} - fenêtre:{window_half_size*2+1}px');
    filename = f'disp_field_{image_id:04d}.{image_ext}'
    plt.savefig(os.path.join(save_path, filename));
    print(f'figure saved: {filename}', end='\r')
    plt.close()
    
print('done', ' '*40)

# +
# 2. Champ de déplacement Sans la translation
for image_id, displ in enumerate(displ_from_previous):
    displ = displ - np.nanmean(displ, axis=0)
    
    plt.figure();
    plt.imshow(cube[image_id]);
    plot_vector_field(points, displ, view_factor=None)
    plt.title(f'sans la translation - images {image_id}→{image_id+1} - fenêtre:{window_half_size*2+1}px');
    filename = f'without_translation_{image_id:04d}.{image_ext}'
    plt.savefig(os.path.join(save_path, filename));
    print(f'figure saved: {filename}', end='\r')
    plt.close()
    
print('done', ' '*40)

# +
# 3. fit bilineaire
for image_id, residuals in enumerate(residuals_from_previous):
    view_factor = None  
    plt.figure();
    plt.title(f'résidus après fit linéaire - images {image_id}→{image_id+1} - fenêtre:{window_half_size*2+1}px');
    plt.imshow(cube[image_id]);
    plot_vector_field(points, residuals, view_factor=None, color='red')
    filename = f'residuals_{image_id:04d}.{image_ext}'  
    plt.savefig(os.path.join(save_path, filename));
    print(f'figure saved: {filename}', end='\r')
    plt.close()
    
print('done', ' '*40)

# +
from scipy.interpolate import RectBivariateSpline

def field_interpolation(grid, vector_field):
    u = vector_field[:, 0].reshape(grid[0].shape)
    v = vector_field[:, 1].reshape(grid[0].shape)
    
    interp_u = RectBivariateSpline(x_span, y_span, u.T)
    interp_v = RectBivariateSpline(x_span, y_span, v.T)
    
    return lambda x, y: np.stack((interp_u(x, y, grid=False),
                                  interp_v(x, y, grid=False)), axis=0)


# -

displ[:, 0].reshape(grid[0].shape).shape

grid.shape

displ_interpolators = [field_interpolation(grid, displ)
                       for displ in displ_from_previous]

u_prime = interp_u(grid[0], grid[1], grid=False)

plt.imshow(u-u_prime); plt.colorbar();

# +
starting_points = points.T
trajectory =  [starting_points, ]

for interp in displ_interpolators:
    uv = interp( *trajectory[-1] )
    trajectory.append(trajectory[-1] + uv)
    
trajectory = np.stack(trajectory)
# -

interp(-100, 0)

plt.plot(trajectory[:, 0, :], trajectory[:, 1, :]);

# # plt.imshow(grid[1])

displ.shape

# +
# ===============================
## Graph macro_strain Eulerian ?
# ===============================
# -



plt.plot( nu, label='coeff. Poisson' );
plt.xlabel('frame id');



filename = f'coeffs_bilinear_fit.csv'
save_path = os.path.join(sample_output_path, filename)
plt.savefig(os.path.join(save_path, filename));
print(f'figure saved: {filename}', end='\r')
numpy.savetxt("foo.csv", a, delimiter=",")

# +
plt.plot( linear_def_from_previous[:, 0, 0], "-o",label='eps_xx' )
plt.plot( linear_def_from_previous[:, 1, 1], label='eps_yy' )
plt.plot( linear_def_from_previous[:, 1, 0], label='a_12' )
plt.plot( linear_def_from_previous[:, 0, 1], label='a_21' )
plt.legend();
plt.xlabel('image id'); plt.title('image to image strain')
plt.ylabel('image to image strain (%)'); plt.legend();


filename = f'image_to_image_coefficients_window{window_half_size}px.{image_ext}'
plt.savefig(os.path.join(sample_output_path, filename));
print(f'figure saved: {filename}', end='\r')


a_11 = linear_def_from_previous[1:, 0, 0]
a_22 = linear_def_from_previous[1:, 1, 1]
a_12 = linear_def_from_previous[1:, 0, 1]
a_21 = linear_def_from_previous[1:, 1, 0]

nu = -a_22/a_11

data = np.vstack([a_11, a_22, a_12, a_21]).T
filename = f'image_to_image_coeffs_bilinear_fit_window{window_half_size}px.csv'
np.savetxt(os.path.join(sample_output_path, filename), data,
           delimiter=",", header='a11, a22, a12, a21')
print(f'data saved: {filename}', end='\r')
# -

dt = 10   # s
v = 2/60  # mm/s
x_0 = 10 # mm
x_i = np.arange(1, len(a_11)+1)*dt*v
eps_app = x_i / x_0

eps_app

plt.plot(eps_app, np.cumsum(a_11), '-o')
plt.plot([0, 0.4], [0, 0.4])

nu = -linear_def_from_previous[1:, 1, 1]/linear_def_from_previous[1:, 0, 0]
plt.plot( nu, label='eps_yy' );
plt.xlabel('frame id');

from scipy.linalg import solve

# +
def_centers = []
for coeffs in linear_def_from_previous:
    a = coeffs[0:2, 0:2]
    b = coeffs[:, 2]

    # solve a x = b
    def_centers.append( solve(a, -b) )

def_centers = np.stack(def_centers)
# -

plt.imshow(cube[0]);
plt.plot(*def_centers[1:12].T)

# +
print('')
print('bilinear regression for each frame')

eps_xx = np.array([p[0, 0] for p, sig in p_sigma])
eps_yy = np.array([p[1, 1] for p, sig in p_sigma])
sigma_hat_x = np.array([sig[0] for p, sig in p_sigma])
sigma_hat_y = np.array([sig[1] for p, sig in p_sigma])

plt.figure();
plt.plot(eps_xx*100, 'o-', label='$\Delta \epsilon_{xx}$')
plt.plot(eps_yy*100, 'o-', label='$\Delta \epsilon_{yy}$')
plt.xlabel('frame id'); plt.title('frame to frame strain')
plt.ylabel('frame to frame strain (%)'); plt.legend();

plt.savefig(os.path.join(sample_output_path, f'frame_to_frame_strain.png'));
print(' save frame_to_frame_strain.png')
# -

cube.dtype

plt.figure();
plt.plot(sigma_hat_x[:-1], 's-')
plt.plot(sigma_hat_y[:-1], 's-')
plt.xlabel('frame id'); plt.ylabel('std residuals (px)'); plt.title('residuals');
plt.savefig(os.path.join(sample_output_path, f'frame_to_frame_residuals.png'));
print(' save frame_to_frame_residuals.png')



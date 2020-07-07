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
from stretchablecorr import *
# #!pip install scikit-image

import filetools as ft
import os, imghdr

# %load_ext autoreload
# %autoreload 2

# # Stretchable Corr

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

# +
# ==================
#  Define the grid
# ==================

window_half_size = 60

grid_spacing = 300 //3
grid_margin = 350 //3

upsample_factor = 100

reference_image = 0

print('Correlation window size:', f'{1+window_half_size*2}px')
print('Upsample factor:', upsample_factor)

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

# 1. get image offsets
xy_center = [[cube.shape[1]//2, cube.shape[2]//2], ]
central_window_halfsize = min(cube.shape[1]//3, cube.shape[2]//2) // 2
offsets = displacements_img_to_img(cube, xy_center,
                                   central_window_halfsize,
                                   upsample_factor=1,
                                   verbose=True)
offsets = np.squeeze(offsets)
print(f' the largest image-to-image offset is {int(np.max(np.abs(offsets))):d}px')

# +
# 2. get image-to-image displacements
    # shape: (nbr_frames-1, Nbr points, 2)
    
print('Compute image-to-image displacement field:')
displ_Euler = displacements_img_to_img(cube, points,
                                       window_half_size,
                                       upsample_factor,
                                       offsets=offsets,
                                       verbose=True)

print('done', ' '*30)

# +
print('Do bilinear fits')
all_coeffs = []
all_residuals = []
for displ_field in displ_Euler:
    coeffs, residuals = bilinear_fit(points, displ_field)
    all_coeffs.append(coeffs)
    all_residuals.append(residuals)

linear_def_from_previous = np.stack(all_coeffs, axis=0)
residuals_from_previous = np.stack(all_residuals, axis=0)
# -

displ_Lagrangian = track_displ_img_to_img(cube, points,
                                          window_half_size, upsample_factor,
                                          offsets=offsets)

# +
positions = np.cumsum(displ_Lagrangian, axis=0) + points #- np.mean(displ_Lagrangian, axis=1)[:, np.newaxis, :]

plt.plot(positions[:, :, 0], positions[:, :, 1]);
# -

# 4. Displacement field relative to the reference image
displ_Lagrangian_ref = track_displ_img_to_ref(cube, points,
                                          window_half_size, upsample_factor,
                                          offsets=offsets)

# +
positions_ref = np.cumsum(displ_Lagrangian_ref, axis=0) + points

plt.plot(positions_ref[:, :, 0], positions_ref[:, :, 1]);
# -

vector_field = positions
x_prime = vector_field[:, 0].reshape(grid[0].shape)
y_prime = vector_field[:, 1].reshape(grid[0].shape)

positions_grid = positions.reshape((2, *grid[0].T.shape))

plt.pcolor(x_prime, y_prime, x_prime, edgecolors='black');
plt.pcolor(x_prime, y_prime, x_prime, edgecolors='black');
plt.axis('equal');

# def plot_vector_field(points, displacements,
#                       view_factor=None, color='white'):
#     amplitudes = np.sqrt(np.nansum( displacements**2, axis=1 )) # disp. amplitude
#
#     mask = ~np.any( np.isnan(displacements), axis=1, keepdims=False )
#     
#     plt.quiver(*points[mask, :].T, *displacements[mask, :].T,
#                angles='xy', color=color,
#                scale_units='xy',
#                scale=1/view_factor if view_factor else None,
#                minlength=1e-4);
#     
#     plt.text(10., 10.,
#              f'max(|u|)={np.nanmax(amplitudes):.2f}px  mean(|u|)={np.nanmean(amplitudes):.2f}px',
#              fontsize=12, color=color,
#              verticalalignment='top')

# +
# =========
#  Graphs
# =========
output_dir = f'frame_to_frame_window{window_half_size}px'
image_ext = "svg"

# --
save_path = os.path.join(sample_output_path, output_dir)
ft.create_dir(save_path)


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
# 1. Champ de déplacement
for image_id, displ in enumerate(displ_Euler):
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
for image_id, displ in enumerate(displ_Euler):
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
# -



# + active=""
# from scipy.interpolate import RectBivariateSpline
#
# def field_interpolation(grid, vector_field):
#     u = vector_field[:, 0].reshape(grid[0].shape)
#     v = vector_field[:, 1].reshape(grid[0].shape)
#     
#     x_span = grid[0][0, :]
#     y_span = grid[0][:, 0]
#     
#     interp_u = RectBivariateSpline(x_span, y_span, u.T)
#     interp_v = RectBivariateSpline(x_span, y_span, v.T)
#     
#     return lambda x, y: np.stack((interp_u(x, y, grid=False),
#                                   interp_v(x, y, grid=False)), axis=0)
#
# displ_interpolators = [field_interpolation(grid, displ)
#                        for displ in displ_from_previous]
#
# u_prime = interp_u(grid[0], grid[1], grid=False)
#
# plt.imshow(u-u_prime); plt.colorbar();

# + active=""
# starting_points = points.T
# trajectory =  [starting_points, ]
#
# for interp in displ_interpolators:
#     uv = interp( *trajectory[-1] )
#     trajectory.append(trajectory[-1] + uv)
#     
# trajectory = np.stack(trajectory)
#
# interp(-100, 0)
#
# plt.plot(trajectory[:, 0, :], trajectory[:, 1, :]);

# +
# =======================
#  Data post processing
# =======================

# Mechanical test parameters:
dt = 10   # s - time between each images
v = 2/60  # mm/s - Deben speed
x_0 = 10 # mm - starting position
x_i = np.arange(1, len(linear_def_from_previous)+1)*dt*v  # mm
eps_app = x_i / x_0

# +
plt.figure();
plt.plot(eps_app*100, linear_def_from_previous[:, 0, 0], "-o",label='eps_xx' )
plt.plot(eps_app*100, linear_def_from_previous[:, 1, 1], label='eps_yy' )
plt.plot(eps_app*100, 0.5*(linear_def_from_previous[:, 1, 0]+linear_def_from_previous[:, 0, 1]), label='eps_xy' )
plt.legend();  plt.xlim([0, np.max(eps_app[1:]*100)])
plt.xlabel('déformation appliquée (%)');
plt.title("variation de déformation\n d'image à image (%)")
plt.ylabel("variation de déformation\n d'image à image (%)"); plt.legend();


filename = f'image_to_image_coefficients_window{window_half_size}px.{image_ext}'
plt.savefig(os.path.join(sample_output_path, filename));
print(f'graph saved: {filename}')


a_11 = linear_def_from_previous[1:, 0, 0]
a_22 = linear_def_from_previous[1:, 1, 1]
a_12 = linear_def_from_previous[1:, 0, 1]
a_21 = linear_def_from_previous[1:, 1, 0]

nu = -a_22/a_11

data = np.vstack([a_11, a_22, a_12, a_21]).T
filename = f'image_to_image_coeffs_bilinear_fit_window{window_half_size}px.csv'
np.savetxt(os.path.join(sample_output_path, filename), data,
           delimiter=",", header='a11, a22, a12, a21')
print(f'data saved: {filename}')

# +
plt.figure();
plt.plot(eps_app[:-1]*100, 100*np.cumsum(a_11), '-o');
plt.plot([0, 30], [0, 30], '-k');
plt.xlabel('déformation appliquée (%)'); #plt.title('image to image strain')
plt.ylabel('DIC measured strain (%)'); #plt.legend();

filename = f'naive_strain_integration_{window_half_size}px.{image_ext}'
plt.savefig(os.path.join(sample_output_path, filename));
print(f'graph saved: {filename}')
# -

nu = -linear_def_from_previous[1:, 1, 1]/linear_def_from_previous[1:, 0, 0]
plt.figure();
plt.plot(eps_app[1:]*100,  nu, '-o' ); plt.title('Estimation du coefficient de Poisson')
plt.ylabel('nu = -a22 / a11'); 
plt.xlabel('déformation appliquée (%)'); plt.xlim([0, np.max(eps_app[1:]*100)])
filename = f'poisson_ratio_graph_{window_half_size}px.{image_ext}'
plt.savefig(os.path.join(sample_output_path, filename));
print(f'graph saved: {filename}')

# + active=""
# from scipy.linalg import solve
#
# def_centers = []
# for coeffs in linear_def_from_previous:
#     a = coeffs[0:2, 0:2]
#     b = coeffs[:, 2]
#
#     # solve a x = b
#     def_centers.append( solve(a, -b) )
#
# def_centers = np.stack(def_centers)
#
# plt.imshow(cube[0]);
# plt.plot(*def_centers[1:12].T)
# -



# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pylab as plt
from stretchablecorr import *
# #!pip install scikit-image

import filetools as ft
import os, imghdr

# %load_ext autoreload
# %autoreload 2

from scipy.integrate import cumtrapz

# # Stretchable Corr - correlations

# ## Search and select images

# +
#### Available samples list
input_data_dir = "./images/"
samples = os.listdir(input_data_dir)
print('Available samples')
print('=================')
ft.print_numbered_list(samples)

# Select a sample:
sample_id = input('> Select an image directory:')
sample_name = samples[int(sample_id)]
print(sample_name)

# +
# ==================
#  Load image cube
# ==================

# List, select and sort images
sample_input_dir = os.path.join(input_data_dir, sample_name)
print(f'Loading "{sample_name}" from {sample_input_dir} ...')

cube, image_names = ft.load_image_sequence(sample_input_dir)
# -

plt.figure(); plt.title(f'sequence standard deviation - {sample_name}');
plt.imshow(np.std(cube, axis=0), cmap='viridis');
ft.save_fig('01_cube_std', sample_name)

# +
# ==================
#  Define the grid
# ==================

window_half_size = 40

grid_spacing = 100 #//3
grid_margin = window_half_size * 3//2

# default:
reference_image = 0

print('Correlation window size:', f'{1+window_half_size*2}px')

# ----
grid = build_grid(cube.shape[1:], margin=grid_margin, spacing=grid_spacing)

# Graph the grid
plt.figure();
plt.title(f'Grille - D={grid_spacing}px - {points.shape[0]} points');
ft.plot_grid_points(grid, background=cube[0],
                    color='white', markersize=3,
                    window_half_size=window_half_size)
ft.save_fig('02_grid', sample_name)

# +
#output_dir = f'frame_to_frame_window{window_half_size}px'
#
# --
#save_path = os.path.join(sample_output_path, output_dir)
#ft.create_dir(save_path)

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
# -

# ## 1. Eulerian displacement field

points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )

# +
# 2. get image-to-image displacements
    # shape: (nbr_frames-1, Nbr points, 2)
    
window_half_size = 100
upsample_factor = 1
print('Compute image-to-image displacement field:')
displ_Euler_coarse = displacements_img_to_img(cube, points,
                                       window_half_size,
                                       upsample_factor,
                                       offsets=offsets,
                                       verbose=True)


# +
window_half_size = 30
upsample_factor = 100
displ_Euler = displacements_img_to_img(cube, points,
                                       window_half_size,
                                       upsample_factor,
                                       offsets=displ_Euler_coarse,
                                       verbose=True)

meta = {'window_half_size':window_half_size,
        'upsample_factor':upsample_factor}
ft.save_data((grid, displ_Euler, meta),
             'displ_Euler_img_to_img',
             sample_name)
# -

# ## 2. Lagrangian displacement

# +
grid_spacing = 20 #//3
window_half_size = 35
grid_margin = window_half_size * 3//2

# ----
grid = build_grid(cube.shape[1:], margin=grid_margin, spacing=grid_spacing)
points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )
# -

print('Compute image-to-image Lagrangian displacement field:')
displ_Lagrangian_coarse = track_displ_img_to_img(cube, points,
                                                 100, 1,
                                                 offsets=offsets)
# what about NaN in offsets... ?

# +
print('Compute image-to-image Lagrangian displacement field:')

upsample_factor  = 50
displ_Lagrangian = track_displ_img_to_img(cube, points,
                                          window_half_size, upsample_factor,
                                          offsets=displ_Lagrangian_coarse)
# -

meta = {'window_half_size':window_half_size,
        'upsample_factor':upsample_factor}
ft.save_data((grid, displ_Lagrangian, meta),
             f'displ_Lagrangian_img_to_img_{len(points)}pts',
              sample_name)

# +
# ===================
# integrate image-to-image fields to get image-to-ref displ.
displ_lagrangian_to_ref = cumtrapz(displ_Lagrangian, axis=0, initial=0)
positions = displ_lagrangian_to_ref + points

# keep only enterely visible path
mask = ~np.any(np.isnan(displ_Lagrangian), axis=(0, 2))
# -

color = 'white'
plt.figure();
#plt.imshow(cube[reference_image, :, :]);
plt.imshow(np.std(cube, axis=0), cmap='viridis');
plt.plot(positions[0, np.logical_not(mask), 0], positions[0, np.logical_not(mask), 1], 's',
         markersize=1, color='red', alpha=0.7);
plt.plot(positions[:, np.logical_not(mask), 0], positions[:, np.logical_not(mask), 1],
         color='red', alpha=0.5, linewidth=1);
plt.plot(positions[0, mask, 0], positions[0, mask, 1], 's', markersize=2, color=color);
plt.plot(positions[:, mask, 0], positions[:, mask, 1], color=color, linewidth=1);


def plot_Y_profile(nearest_x, image_id,
                   grid,
                   displ_lagrangian_to_ref,
                   color = 'darkorange'):
    
    dx = displ_lagrangian_to_ref[image_id, :, 0].reshape(grid[0].shape)
    dy = displ_lagrangian_to_ref[image_id, :, 1].reshape(grid[0].shape)

    # Profils Y
    x_span = grid[0][0, :]
    j = np.searchsorted(x_span, nearest_x)
    
    dy_profile = dy[:, j]
    mask_dy_profile = ~np.isnan(dy_profile)
    dy_profile = dy_profile[mask_dy_profile]

    y_profile = grid[1][:, j][mask_dy_profile]
    x_profile = grid[0][:, j][mask_dy_profile]
    
    plt.plot(y_profile,
             dy_profile,
             color=color,
             label=f'at x={x_span[j]} px')
    
    plt.xlabel('y [px]'); plt.ylabel('displ. v [px]');
    plt.title(f"displ. component v - image {image_names[image_id].split('.')[0]}")

    a, b = np.polyfit(y_profile, dy_profile, 1)
    plt.plot(y_profile, a*y_profile + b, ':',
             color=color, alpha=0.8,
             label=f'$\epsilon_y$=dv/dy={a*100:.2f}%');
    plt.legend();
    return x_profile, y_profile, dy_profile


# +
image_id = 8

x1050, y1050, v1050 = plot_Y_profile(850, image_id,
               grid,
               displ_lagrangian_to_ref,
               color = 'lightblue')

x450, y450, v450 = plot_Y_profile(450, image_id,
               grid,
               displ_lagrangian_to_ref,
               color = 'darkorange')


# -

plt.figure();
plt.imshow(cube[image_id, :, :]);
plt.plot(x450, y450, '.-', color='darkorange')
plt.plot(x1050, y1050, '.-', color='lightblue')
plt.title(f'{image_names[image_id]}');

# +
image_id = 10

view_factor = 20

x = grid[0]
y = grid[1]

dx = np.diff(x, axis=1, prepend=np.NaN)
dy = np.diff(y, axis=0, prepend=np.NaN)

u = displ_lagrangian_to_ref[image_id, :, 0].reshape(grid[0].shape)
v = displ_lagrangian_to_ref[image_id, :, 1].reshape(grid[0].shape)
dudx = np.diff(u, axis=0, prepend=np.NaN)
dvdy = np.diff(v, axis=0, prepend=np.NaN) / dy 

positions_amplified = displ_lagrangian_to_ref*view_factor + points
x_amplified = positions_amplified[image_id, :, 0].reshape(grid[0].shape)
y_amplified = positions_amplified[image_id, :, 1].reshape(grid[0].shape)

plt.pcolor(x_amplified, y_amplified, dvdy*100,
           edgecolors='black', linewidth=1, antialiased=True, cmap='viridis');
plt.axis('equal');
plt.colorbar();
plt.title(f'{image_names[image_id]} - displ x{view_factor}');
# -

dy

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

# --
save_path = os.path.join(sample_output_path, output_dir)
ft.create_dir(save_path)
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
7e3*0.8/100





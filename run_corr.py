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
from strechablescorr import *
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
print(' ')

# Select a sample:
sample_name = 'gris_e_zoom8'



# +
# ==================
#  Load image cube
# ==================
print(' ')
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
plt.imshow(np.std(cube, axis=2));
plt.savefig(os.path.join(sample_output_path, '01_cube_std.png'));

# +
# ==================
#  Define the grid
# ==================

window_half_size = 120
upsample_factor = 20

grid_spacing = 280
grid_margin = 350

reference_image = 0

# ----
grid = build_grid(cube.shape, margin=grid_margin, spacing=grid_spacing)
points = np.vstack( [grid[0].flatten(), grid[1].flatten()] ).T

# Graph the grid
plt.figure();
plt.imshow(cube[:, :, reference_image]);
plt.title(f'Grid - image #{reference_image:02d}');
plt.plot(*grid, 'o', color='white', markersize=3);

for k, (x, y) in enumerate(points):
    if len(points) > 10 and k % 5 != 0:
        continue
    text_offset = 10.0
    plt.text(x+text_offset, y+text_offset,
             str(k), fontsize=8, color='white')
    
# graph one of the ROI
box = np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])*(window_half_size + 1)
middle_point = tuple(np.array(grid[0].shape) // 2 - 1)
plt.plot(box[0]+grid[0][middle_point], box[1]+grid[1][middle_point],
         color='white', linewidth=1)

plt.savefig(os.path.join(sample_output_path, '02_grid.png'));

# +
# ============================================
#  Compute image to image displacement field
# ============================================
print(' ')
print('Compute image to image displacement field:')
# shape: (Nbr points, 2, nbr_frames)
displacements = np.zeros((*points.shape, cube.shape[2]))
for k, (x, y) in enumerate(points):
    print(f'{k: 4d}/{len(points)}', end='\r')
    disp_from_prev = get_displacement_from_previous(cube, x, y, 
                                          window_half_size, upsample_factor,
                                          verbose=False)
    displacements[k, :, :] = disp_from_prev.T
        
print('done', ' '*30)


# -

def plot_vector_field(displacements, view_factor=None):
    amplitudes = np.sqrt(np.sum( displacements**2, axis=1 )) # disp. amplitude

    plt.imshow(cube[:, :, image_id]);
    plt.quiver(*points.T, *displacements.T,
               angles='xy', color='white',
               scale_units='xy',
               scale=1/view_factor if view_factor else None,
               minlength=1e-4);
    #plt.title(f'champ de déplacement - image {reference_image}->{image_id} \n d_max={np.nanmax(amplitudes):.2f}px');
    plt.text(20.1, 50.1,
             f'd_max={np.nanmax(amplitudes):.2f}px', fontsize=8, color='white')


# +
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
    # np.linalg.inv(np.matmul(M.T, M))

    # unbiased estimator variance (see p47 T. Hastie)
    sigma_hat_x = np.sqrt(residual_x/(M.shape[0]-M.shape[1]-1))
    sigma_hat_y = np.sqrt(residual_y/(M.shape[0]-M.shape[1]-1))

    # Residus 
    u_linear = np.matmul( M, p_ux )
    v_linear = np.matmul( M, p_uy )

    residus_x = u - u_linear
    residus_y = v - v_linear

    residus_xy = np.vstack([residus_x, residus_y])
    
    return p, residus_xy#(sigma_hat_x, sigma_hat_y)
# -



# +
# Quiver Graph
output_dir = 'frame_to_frame_disp'
save_path = os.path.join(sample_output_path, output_dir)
ft.create_dir(save_path)

for image_id in range(displacements.shape[-1]):
    disp_k = displacements[:, :, image_id]
    # Champ de déplacement
    plt.figure();
    disp_k_ss_translation = disp_k - np.nanmean(disp_k, axis=0)
    plot_vector_field(disp_k_ss_translation, view_factor=None)
    plt.title(f'champ de déplacement - image {image_id-1}->{image_id}');
    filename = f'disp_field_{image_id:04d}.png'
    plt.savefig(os.path.join(save_path, filename));
    print(f'figure saved: {filename}')
    plt.close()
    
    # fit
    p, residus = bilinear_fit(points, disp_k)
    amplitudes = np.sqrt(np.nansum( residus**2, axis=1 )) # disp. amplitude
    view_factor = None
    plt.figure();
    plt.title(f'résidus après fit linéaire - image {image_id-1}->{image_id}');
    plt.imshow(cube[:, :, image_id]);
    plt.quiver(*points.T, *residus,
               angles='xy', color='red',
               scale_units='xy',
               scale=1/view_factor if view_factor else None,
               minlength=1e-4);
    plt.text(20.1, 50.1,
             f'max={np.nanmax(amplitudes):.2f}px',
             fontsize=8, color='red')
    filename = f'residuals_{image_id:04d}.png'  
    plt.savefig(os.path.join(save_path, filename));
    plt.close()
# -

disp_k_ss_translation = disp_k - np.mean(disp_k, axis=0)

# +
print('')
print('bilinear regression for each frame')
p_sigma = [ bilinear_fit(points, displacements[:, :, k]) for k in range(cube.shape[2])]

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



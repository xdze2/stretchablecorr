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

# Select a sample:
sample_name = samples[2]
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
plt.imshow(np.std(cube, axis=2));
plt.savefig(os.path.join(sample_output_path, '01_cube_std.svg'));

# +
# ==================
#  Define the grid
# ==================

window_half_size = 120

grid_spacing = 200 #//3
grid_margin = 350 #//3


upsample_factor = 100


reference_image = 0

# ----
grid = build_grid(cube.shape, margin=grid_margin, spacing=grid_spacing)
points = np.vstack( [grid[0].flatten(), grid[1].flatten()] ).T

# Graph the grid
show_pts_number = False
show_window = False

plt.figure();
plt.imshow(cube[:, :, reference_image]);
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
print('Compute image to image displacement field:')
# shape: (Nbr points, 2, nbr_frames)

offsets = get_displacement_from_previous(cube, cube.shape[0]//2, cube.shape[1]//2, 
                                          min(cube.shape[0]//2, cube.shape[1]//2)//2,
                                upsample_factor=1,
                                verbose=False)

displacements = np.zeros((*points.shape, cube.shape[2]))
for k, (x, y) in enumerate(points):
    print(f'{k: 4d}/{len(points)}', end='\r')
    disp_from_prev = get_displacement_from_previous(cube, x, y, 
                                          window_half_size, upsample_factor, offsets=offsets,
                                          verbose=False)
    displacements[k, :, :] = disp_from_prev.T
        
print('done', ' '*30)


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
    #plt.title(f'champ de déplacement - image {reference_image}->{image_id} \n d_max={np.nanmax(amplitudes):.2f}px');
    plt.text(10., 10.,
             f'max(|u|)={np.nanmax(amplitudes):.2f}px  mean(|u|)={np.nanmean(amplitudes):.2f}px',
             fontsize=12, color=color,
             verticalalignment='top')


# +
# Quiver Graph
output_dir = f'frame_to_frame_window{window_half_size}px'
image_ext = "svg"

save_path = os.path.join(sample_output_path, output_dir)
ft.create_dir(save_path)
for image_id in range(1, displacements.shape[-1]):
    
    # 1. Champ de déplacement
    disp_k = displacements[:, :, image_id]
    
    plt.figure();
    plt.imshow(cube[:, :, image_id]);
    plot_vector_field(points, disp_k, view_factor=None)
    plt.title(f'champ de déplacement - images {image_id-1}→{image_id} - fenêtre:{window_half_size*2+1}px');
    filename = f'disp_field_{image_id:04d}.{image_ext}'
    plt.savefig(os.path.join(save_path, filename));
    print(f'figure saved: {filename}')
    plt.close()
    
    # 2. Sans la translation
    disp_k_ss_translation = disp_k - np.nanmean(disp_k, axis=0)
    
    plt.figure();
    plt.imshow(cube[:, :, image_id]);
    plot_vector_field(points, disp_k_ss_translation, view_factor=None)
    plt.title(f'sans la translation - images {image_id-1}→{image_id} - fenêtre:{window_half_size*2+1}px');
    filename = f'without_translation_{image_id:04d}.{image_ext}'
    plt.savefig(os.path.join(save_path, filename));
    print(f'figure saved: {filename}')
    plt.close()
    
    # 3. fit bilineaire
    p, residus = bilinear_fit(points, disp_k)
    #amplitudes = np.sqrt(np.nansum( residus**2, axis=0 )) # disp. amplitude
    view_factor = None
    
    plt.figure();
    plt.title(f'résidus après fit linéaire - images {image_id-1}→{image_id} - fenêtre:{window_half_size*2+1}px');
    plt.imshow(cube[:, :, image_id]);
    plot_vector_field(points, residus, view_factor=None, color='red')
    filename = f'residuals_{image_id:04d}.{image_ext}'  
    plt.savefig(os.path.join(save_path, filename));
    print(f'figure saved: {filename}')
    plt.close()
# -

p

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



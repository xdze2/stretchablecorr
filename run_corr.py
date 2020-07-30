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
from skimage import io
from skimage import img_as_uint
from stretchablecorr import *
# #!pip install scikit-image

import filetools as ft
import os, imghdr

# %load_ext autoreload
# %autoreload 2

from scipy.integrate import cumtrapz

# # Stretchable Corr - correlations

# ## select input images

# +
#### Available samples list
input_data_dir = "./images/"
samples = next(os.walk(input_data_dir))[1]
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
print('Loading image cube...', end='\r')

# List, select and sort images
sample_input_dir = os.path.join(input_data_dir, sample_name)
print(f'Load "{sample_name}" from {sample_input_dir}')

cube, image_names = ft.load_image_sequence(sample_input_dir)
# -

plt.figure(); plt.title(f'sequence standard deviation - {sample_name}');
plt.imshow(np.std(cube, axis=0), cmap='viridis');
ft.save_fig('01_cube_std', sample_name)

# ## 1. Eulerian displacement field

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
points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )

# Graph the grid
plt.figure();
plt.title(f'Grille - D={grid_spacing}px - {points.shape[0]} points');
ft.plot_grid_points(grid, background=cube[0],
                    color='white', markersize=3,
                    window_half_size=window_half_size)
ft.save_fig('02_grid', sample_name)

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

# 2. get COARSE image-to-image displacements
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
# 3. get image-to-image displacements
window_half_size = 30
upsample_factor = 100
displ_Euler = displacements_img_to_img(cube, points,
                                       window_half_size,
                                       upsample_factor,
                                       offsets=displ_Euler_coarse,
                                       verbose=True)

# 4. Save data
meta = {'window_half_size':window_half_size,
        'upsample_factor':upsample_factor}
ft.save_data((grid, displ_Euler, meta),
             f'displ_Euler_img_to_img_{len(points)}pts',
             sample_name)
# -

# ## 2. Lagrangian displacement

# +
# ==================
#  Define the grid
# ==================

grid_spacing = 20 #//3
window_half_size = 35
grid_margin = window_half_size * 3//2

# ----
grid = build_grid(cube.shape[1:], margin=grid_margin, spacing=grid_spacing)
points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )
# -

# 2. get COARSE image-to-image displacements
print('Compute image-to-image Lagrangian displacement field:')
displ_Lagrangian_coarse = track_displ_img_to_img(cube, points,
                                                 100, 1,
                                                 offsets=offsets)
# what about NaN in offsets... ?

# +
# 3. get image-to-image displacements
print('Compute image-to-image Lagrangian displacement field:')

upsample_factor  = 50 
displ_Lagrangian = track_displ_img_to_img(cube, points,
                                          window_half_size, upsample_factor,
                                          offsets=displ_Lagrangian_coarse)

# Save data
meta = {'window_half_size':window_half_size,
        'upsample_factor':upsample_factor}
ft.save_data((grid, displ_Lagrangian, meta),
             f'displ_Lagrangian_img_to_img_{len(points)}pts',
              sample_name)
# -





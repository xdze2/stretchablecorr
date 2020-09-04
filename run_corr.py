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

# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pylab as plt
import stretchablecorr as sc
# #!pip install scikit-image

from scipy.integrate import cumtrapz

# # Stretchable Corr - correlations

# ## select input images

# ==================
#  Load image cube
# ==================
sample_name, sample_input_dir = sc.select_sample_dir('./images')
cube, image_names = sc.load_image_sequence(sample_input_dir)

plt.figure(); plt.title(f'sequence standard deviation - {sample_name}');
plt.imshow(np.std(cube, axis=0), cmap='viridis');
sc.save_fig('01_cube_std', sample_name)

# ## 1. Lagrangian displacement

# +
# ==================
#  Define the grid
# ==================

grid_spacing = 64
window_half_size = None
grid_margin = 100

# ----
grid = sc.build_grid(cube.shape[1:], margin=grid_margin, spacing=grid_spacing)
points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )

# Graph the grid
plt.figure();
plt.title(f'Grille - spacing={grid_spacing}px - {points.shape[0]} points');
sc.plot_grid_points(grid, background=cube[0],
                    color='white', markersize=3, show_pts_number=True,
                    window_half_size=window_half_size)

# +
# get image-to-image displacements
window_half_size = 20
save = True

displ_Lagrangian, err = sc.track_displ_img_to_img(cube, points,
                                                  method='opti',
                                                  coarse_search=True,
                                                  phase=False,
                                                  window_half_size=window_half_size,
                                                  offsets=None)

# Save data
if save:
    meta = {'window_half_size':window_half_size,
            'upsample_factor':upsample_factor}
    sc.save_data((grid, displ_Lagrangian, meta),
                 f'displ_Lagrangian_img_to_img_{len(points)}pts',
                 sample_name)
# +
# Integrate
# image-to-image fields to get image-to-ref displ.

displ_lagrangian_to_ref = sc.integrate_displacement(displ_Lagrangian)
trajectories = displ_lagrangian_to_ref + points

plt.imshow(cube[0])
for k in range(0, len(points), 2):
    plt.plot(*trajectories[0, k, :].T, 'ws', markersize=2)
    plt.plot(*trajectories[:, k, :].T, 'w')
# -

displ_Lagrangian, err = sc.track_displ_2steps(cube, points,
                                              method='opti',
                                              coarse_search=True,
                                              phase=False,
                                              window_half_size=15,
                                              offsets=None)

max_error = np.nanmax(err, axis=0).reshape(grid[0].shape)

plt.imshow(max_error); plt.colorbar();



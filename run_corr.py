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

def integrate_displacement(displ_Lagrangian):
    # add zeros at the begining
    zeros = np.zeros_like(displ_Lagrangian[0])[np.newaxis, :, :]
    displ_Lagrangian_zero = np.concatenate([zeros, displ_Lagrangian], axis=0)

    displ_lagrangian_to_ref = np.cumsum(displ_Lagrangian_zero, axis=0)
    return displ_lagrangian_to_ref


# +
# ==================
#  Define the grid
# ==================

grid_spacing = 30
window_half_size = 100
grid_margin = 55

# ----
grid = build_grid(cube.shape[1:], margin=grid_margin, spacing=grid_spacing)
points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )

# Graph the grid
plt.figure();
plt.title(f'Grille - D={grid_spacing}px - {points.shape[0]} points');
ft.plot_grid_points(grid, background=cube[0],
                    color='white', markersize=3, show_pts_number=True,
                    window_half_size=window_half_size)

# +
# get image-to-image displacements
print('Compute image-to-image Lagrangian displacement field:')

upsample_factor  = 50
window_half_size = 20
save = True

displ_Lagrangian, err = track_displ_img_to_img(cube, points,
                                               method='opti',
                                               coarse_search=True,
                                               phase=False,
                                               window_half_size=window_half_size,
                                               #upsample_factor=upsample_factor,
                                               offsets=None)

# Save data
if save:
    meta = {'window_half_size':window_half_size,
            'upsample_factor':upsample_factor}
    ft.save_data((grid, displ_Lagrangian, meta),
                 f'displ_Lagrangian_img_to_img_{len(points)}pts',
                  sample_name)
# +
# Integrate
# image-to-image fields to get image-to-ref displ.

displ_lagrangian_to_ref = integrate_displacement(displ_Lagrangian)
print(f'{len(displ_lagrangian_to_ref)=}')
positions = displ_lagrangian_to_ref + points

# keep only enterely visible path
mask = ~np.any(np.isnan(displ_Lagrangian), axis=(0, 2))
# -

displ_Lagrangian_dict = {}

# +
# ========================
# compare skimage vs optim
# ========================
upsample_factor  = 100
window_half_size = 12
save = False

displ_Lagrangian_to_ref = {}
displ_Lagrangian_dict = {}
displ_Lagrangian, err = track_displ_img_to_img(cube, points,
                                               method='skimage',
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor,
                                               offsets=None)

displ_Lagrangian_dict['sk_25px_x100'] = displ_Lagrangian
displ_Lagrangian_to_ref['sk_25px_x100'] = integrate_displacement(displ_Lagrangian)

# +
window_half_size = 12
save = False

displ_Lagrangian, err = track_displ_img_to_img(cube, points,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               offsets=None)

displ_Lagrangian_dict['opti_12px'] = displ_Lagrangian
displ_Lagrangian_to_ref['opti_12px'] = integrate_displacement(displ_Lagrangian)

# +
window_half_size = 12
save = False

displ_Lagrangian, err = track_displ_img_to_img(cube, points,
                                               method='opti',
                                               phase=False,
                                               window_half_size=window_half_size,
                                               offsets=None)

displ_Lagrangian_dict['opti_cc_12px'] = displ_Lagrangian
displ_Lagrangian_to_ref['opti_cc_12px'] = integrate_displacement(displ_Lagrangian)

# +
k = 11
for key, displ in displ_Lagrangian_to_ref.items():
    plt.plot(*displ[:, k].T, '-', label=key)
    
plt.legend();
plt.axis('equal');

#plt.xlim([-15, 0]);
#plt.ylim([-15, 0]);
# -

points[k:k+1, :]

displ_Lagrangian, err = track_displ_img_to_img(cube, points[k:k+1, :],
                                               method='opti',
                                               window_half_size=window_half_size,
                                               offsets=None)

dif = displ_Lagrangian_dict['sk_25px_x100'] - displ_Lagrangian_dict['opti_25px']
d = np.sqrt(dif[:, :, 0]**2 + dif[:, :, 1]**2)

np.argmax(d[:, k])

# +
window_half_size = 35
save = False

displ_Lagrangian, err = track_displ_img_to_img(cube, points,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               offsets=None)
displ_Lagrangian_dict['opti_35px'] = displ_Lagrangian

displ_Lagrangian_to_ref['opti_35px'] = integrate_displacement(displ_Lagrangian)

# +
window_half_size = 35
save = False

displ_Lagrangian, err = track_displ_img_to_img(cube, points,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               phase=False,
                                               offsets=None)
displ_Lagrangian_dict['opti_noPhase_35px'] = displ_Lagrangian
displ_Lagrangian_to_ref['opti_noPhase_35px'] = integrate_displacement(displ_Lagrangian)

# +
dif = displ_Lagrangian_dict['opti_noPhase_35px'] - displ_Lagrangian_dict['opti_35px']

d = np.sqrt(dif[:, :, 0]**2 + dif[:, :, 1]**2)
# -

np.nanargmax(d.flatten(), axis=0)

d[:, 75]

d[np.argmax(d, axis=0)].T

displ_Lagrangian_one, err = track_displ_img_to_img(cube, points,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               offsets=None)

displ_Lagrangian_to_ref_one = integrate_displacement(displ_Lagrangian_one)

displ_Lagrangian_two, err = track_displ_img_to_img(cube[::2], points,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               offsets=None)

displ_Lagrangian_to_ref_two = integrate_displacement(displ_Lagrangian_two)

# +
k = 25
plt.plot(*displ_Lagrangian_to_ref_two[:, k].T, '-')
plt.plot(*displ_Lagrangian_to_ref_one[:, k].T, '-')
    
plt.legend();
plt.axis('equal');
# -

# Plot all crop zones
point_id = 25
window_half_size = 35
for k, (B, displ) in enumerate(zip(cube, displ_lagrangian_to_ref)):
    offset = displ[point_id] + points[point_id]
    source, ij_src = crop(B, offset, window_half_size)
    plt.figure(figsize=(2, 2));
    plt.imshow(source)

# +
k = 10
window_half_size = 10
displ_Lagrangian_to_ref = {}

displ_Lagrangian_opti, err = track_displ_img_to_img(cube, points[k:k+1, :],
                                                    method='opti',
                                                   window_half_size=window_half_size,
                                                    coarse_search=True,
                                                    offsets=None)
displ_Lagrangian_to_ref['opti'] = integrate_displacement(displ_Lagrangian_opti)

displ_Lagrangian_opti, err = track_displ_img_to_img(cube, points[k:k+1, :],
                                                    method='opti',
                                                   window_half_size=2*window_half_size,
                                                    coarse_search=True,
                                                    offsets=None)
displ_Lagrangian_to_ref['opti 2x'] = integrate_displacement(displ_Lagrangian_opti)

displ_Lagrangian_skim, err = track_displ_img_to_img(cube, points[k:k+1, :],
                                                    method='skimage',
                                                    window_half_size=window_half_size,
                                                    coarse_search=True,
                                                    offsets=None)

displ_Lagrangian_to_ref['skimage'] = integrate_displacement(displ_Lagrangian_skim)


# +
for key, displ in displ_Lagrangian_to_ref.items():
    plt.plot(*displ[:, 0].T, '-', label=key)
    
plt.legend();
plt.axis('equal');
# -

if -3:
    print(3)

dif = displ_Lagrangian_to_ref['opti'] - displ_Lagrangian_to_ref['skimage']
d = np.sqrt(dif[:, :, 0]**2 + dif[:, :, 1]**2)

plt.plot(d, 'o-')

# +
# Phase optim  VS  cross corr
k = 10
window_half_size = 100
iA, iB = 17, 18

pts = points[k] + displ_Lagrangian_to_ref['skimage'][iA, :]

# +
window_half_size = 25
displ_Lagrangian_skim, err = track_displ_img_to_img(cube[iA:iB+1], pts,
                                                    method='skimage',
                                                    window_half_size=window_half_size,
                                                    offsets=None)

displ_Lagrangian_opti, err = track_displ_img_to_img(cube[iA:iB+1], pts,
                                                    method='opti',
                                                    window_half_size=window_half_size,
                                                    offsets=None)
# -

print(displ_Lagrangian_skim)
print(displ_Lagrangian_opti)

# get offset
dx, dy, err = get_shifts(cube[iA], cube[iB], *pts, upsample_factor=100,
                         window_half_size=45, method='skimage', coarse_search=False)
print(dx, dy)

# get offset
dx, dy, err = get_shifts(cube[iA], cube[iB], *pts, upsample_factor=100,
                         window_half_size=25, method='skimage', coarse_search=True)
print(dx, dy)

# get offset
dx, dy, err = get_shifts(cube[iA], cube[iB], *pts, 
                         window_half_size=45, method='opti', coarse_search=True)
print(dx, dy)

-11.22 + 11.04

-11.245+0.976

pts = pts.flatten()

# +
offset = np.array([-11, -2])

window_half_size = 45
A, ij_srca = crop(cube[iA], pts, window_half_size)
B, ij_srcb = crop(cube[iB], pts+offset, window_half_size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.imshow(A);
ax2.imshow(B);
# -

ij_srca, ij_srcb

# +
dx_span_ph, dy_span_ph, AdotB_pĥase = output_cross_correlation(A, B, upsamplefactor=5, phase=True)
dxy_phase, err = phase_registration_optim(A, B, phase=True, verbose=False)
print('phase:', dxy_phase)

dx_span, dy_span, AdotB_corr = output_cross_correlation(A, B, upsamplefactor=5, phase=False)
dxy_corr, err = phase_registration_optim(A, B, phase=False)

print('corr: ', dxy_corr)
print('skim: ', phase_cross_correlation(A, B, upsample_factor=100)[0])
zoom_size = 8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.pcolormesh(-dx_span_ph, -dy_span_ph, AdotB_pĥase); plt.axis('equal');
ax1.plot(*dxy_phase[::-1], "or")
ax2.plot(*dxy_phase[::-1], "or")

ax1.set_ylim([-zoom_size + dxy_phase[0], zoom_size + dxy_phase[0]]);
ax1.set_xlim([-zoom_size + dxy_phase[1], zoom_size + dxy_phase[1]]);

ax2.pcolormesh(-dx_span_ph, -dy_span_ph, AdotB_corr); plt.axis('equal');
ax2.plot(*dxy_corr[::-1], "xk")
ax1.plot(*dxy_corr[::-1], "xw")

ax2.set_ylim([-zoom_size + dxy_phase[0], zoom_size + dxy_phase[0]]);
ax2.set_xlim([-zoom_size + dxy_phase[1], zoom_size + dxy_phase[1]]);
# -

dx_span, dy_span, AdotB = output_cross_correlation(A, B, upsamplefactor=5, phase=False)
(dx, dy), err = phase_registration_optim(A, B, phase=False)
print(dx, dy)
plt.pcolormesh(dx_span, dy_span, AdotB); plt.axis('equal');
plt.plot(dy, dx, "or")
zoom_size = 2
plt.ylim([-zoom_size - dy, zoom_size - dy]);
plt.xlim([-zoom_size - dx, zoom_size - dx]);

phase_cross_correlation(A, B, upsample_factor=100)

dx, dy, AdotB = output_cross_correlation(A, B, upsamplefactor=5, phase=False)
plt.pcolormesh(dx, dy, AdotB); plt.axis('equal');

# +
k = 36

plt.plot(*displ_lagrangian_to_ref[:, k].T)
# -

plt.plot(*offsets[:].T)

# +
k = 35

plt.plot(*displ_lagrangian_to_ref[:, k].T)
# -

upsample_factor  = 50 
window_half_size = 50
displ_Lagrangian_ref_fine, err_ref = track_displ_img_to_ref(cube, points,
                                               window_half_size, upsample_factor,
                                               offsets=displ_lagrangian_to_ref[1:, :, :],
                                                           method='opti')


# +
k = 35

plt.plot(*displ_lagrangian_to_ref[:, k].T)
plt.plot(*displ_Lagrangian_ref_fine[:, k].T)
# -

err_ref[:, k]

err[:, k]


def test(a, b, c='a', **kwargs):
    print(a, b)
    print(c)
    print(kargs)


test(1, 2, yo='1')

# ## error RMS

# +
# ==================
#  Define the grid
# ==================

grid_spacing = 120
window_half_size = 100
grid_margin = 55

# ----
grid = build_grid(cube.shape[1:], margin=grid_margin, spacing=grid_spacing)
points = np.stack( (grid[0].flatten(), grid[1].flatten()), axis=-1 )

# Graph the grid
plt.figure();
plt.title(f'Grille - D={grid_spacing}px - {points.shape[0]} points');
ft.plot_grid_points(grid, background=cube[0],
                    color='white', markersize=3, show_pts_number=True,
                    window_half_size=window_half_size)
# -

k = 50
point_k = points[k, None]

point_k

# +
upsample_factor  = 100
window_half_size = 35


displ_Lagrangian1, err1 = track_displ_img_to_img(cube[::1], point_k,
                                               method='skimage',
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor,
                                               coarse_search=True,
                                               offsets=None)
displ_Lagrangian_to_ref1 = integrate_displacement(displ_Lagrangian1)

displ_Lagrangian2, err2 = track_displ_img_to_img(cube[::2], point_k,
                                               method='skimage',
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor,
                                               offsets=displ_Lagrangian_to_ref1)



displ_Lagrangian12, err12 = track_displ_img_to_img(cube[::1], point_k,
                                               method='skimage',
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor,
                                               coarse_search=False,
                                               offsets=displ_Lagrangian_to_ref1)

displ_Lagrangian3, err3 = track_displ_img_to_img(cube[::3], point_k,
                                               method='skimage',
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor,
                                               offsets=displ_Lagrangian_to_ref1)

displ_Lagrangian4, err4 = track_displ_img_to_ref(cube, point_k,
                                               method='skimage',
                                               window_half_size=window_half_size,
                                               upsample_factor=upsample_factor,
                                               offsets=displ_Lagrangian_to_ref1)

displ_Lagrangian5, err5 = track_displ_img_to_img(cube[::1], point_k,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               phase=True,
                                               coarse_search=True,
                                               offsets=None)

displ_Lagrangian6, err6 = track_displ_img_to_img(cube[::2], point_k,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               phase=True,
                                               coarse_search=True,
                                               offsets=displ_Lagrangian_to_ref1)

displ_Lagrangian7, err7 = track_displ_img_to_img(cube[::3], point_k,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               phase=True,
                                               coarse_search=True,
                                               offsets=displ_Lagrangian_to_ref1)

displ_Lagrangian8, err8 = track_displ_img_to_ref(cube, point_k,
                                               method='opti',
                                               window_half_size=window_half_size,
                                               phase=True,
                                               coarse_search=True,
                                               offsets=displ_Lagrangian_to_ref1)
# -


plt.plot(np.arange(0, len(err1), 1), err1, label='1 by 1')
plt.plot(np.arange(0, len(err1), 1), err12, label='1 by 1, no search')
plt.plot(np.arange(0, len(err1), 2), err2, label='2 by 2')
plt.plot(np.arange(0, len(err1)-1, 3), err3, label='3 by 3')
plt.plot(np.arange(0, len(err4), 1), err4, label='to ref')
plt.legend();

plt.semilogy(np.arange(0, len(err5), 1), err5, label='trace H-1')
plt.semilogy(np.arange(0, len(err5), 2), err6, label='trace H-1 2by2')
plt.semilogy(np.arange(0, len(err5)-1, 3), err7, label='trace H-1 3by3')
plt.semilogy(np.arange(0, len(err5), 1), err8, label='trace H-1 to ref')
plt.legend();

# +
displ_Lagrangian_to_ref2 = integrate_displacement(displ_Lagrangian2)
displ_Lagrangian_to_ref1 = integrate_displacement(displ_Lagrangian1)
displ_Lagrangian_to_ref3 = integrate_displacement(displ_Lagrangian3)

displ_Lagrangian_to_ref5 = integrate_displacement(displ_Lagrangian5)
displ_Lagrangian_to_ref6 = integrate_displacement(displ_Lagrangian6)
displ_Lagrangian_to_ref7 = integrate_displacement(displ_Lagrangian7)
# -

plt.plot(*displ_Lagrangian_to_ref1[:, 0, :].T, '-o', label='1to1 sk')
plt.plot(*displ_Lagrangian_to_ref2[:, 0, :].T)
plt.plot(*displ_Lagrangian_to_ref3[:, 0, :].T)
plt.plot(*displ_Lagrangian4[:, 0, :].T)
plt.plot(*displ_Lagrangian_to_ref5[:, 0, :].T, label='opti')
displ_Lagrangian8
plt.plot(*displ_Lagrangian8[:, 0, :].T, label='opti to ref')
plt.legend();

#













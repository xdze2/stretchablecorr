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

# %load_ext autoreload
# %autoreload 2

# # Strechable Corr

import numpy as np
import matplotlib.pylab as plt
from skimage import io
from skimage import img_as_uint
from strechablescorr import *

# +
# #!pip install scikit-image
# -

# ## Search and select images

import filetools as ft
import os
from tabulate import tabulate

#data_dir = "./images"
data_dir = "/media/etienne/Transcend/20200127_SOLEIL_DIFFABS/images"

# Sample list
samples = os.listdir(data_dir)
print(', '.join(samples))

# +
# Select a sample
sample_name = 'HS2'

# List, select and sort loading steps
sample_dir = os.path.join(data_dir, sample_name)
steps = os.listdir(sample_dir)

stepsinfo = [ft.parse_step_dir(step) for step in steps]

steps = [info for info in stepsinfo
         if info['direction'] == 'loading'
         and not info['tag']]

steps.sort(key=lambda x: x['strain'])

# Get image path list

def pick_onefile(path):
    filename = os.listdir(path)[0]
    return os.path.join(path, filename)

image_list = [pick_onefile(os.path.join(sample_dir, step['stepname']))
              for step in steps]

steps = [{**info, 'img path': p}
         for info, p in zip(steps, image_list)]

# Print table
print(' sample name:', sample_name)
print(tabulate(steps, headers='keys'))
# -

print( [(k,s['stepname']) for k, s in enumerate(steps)] )

# ## Set output directory

# +
# #!mkdir output

# +
output_dir = "./output"

output_path = os.path.join(output_dir, sample_name)
ft.create_dir(output_path)

# + [markdown] toc-hr-collapsed=false
# ## Export selected images
# -

image_dir = os.path.join(output_path, 'images')
ft.create_dir(image_dir)

# Check image histogram
image = ft.load_image(steps[14]['img path'])
plt.hist(image.flatten(), bins=100);
plt.xlabel('intensity value'); plt.ylabel('pixel counts');

# Rescale intensities
intensity_high = 4000#3600
intensity_low = 100

# Export images
for k, info in enumerate(steps):
    image_filename = f"{k:03d}_{info['stepname']}.png"
    image = ft.load_image(info['img path'])
    image = colorize_image(image, intensity_low, intensity_high, cmap='viridis')
    image = img_as_uint(image)
    img_path = os.path.join(image_dir, image_filename)
    io.imsave(img_path, image[:, :, 0:3])
    print(f'save {img_path}', ' '*10, end='\r')
print('done', ' '*40)

# +
# Export image cube
cube = []
for k, info in enumerate(steps):
    image_filename = f"{k:03d}_{info['stepname']}.tiff"
    image = ft.load_image(info['img path'])
    cube.append(image)

cube = np.dstack( cube )

cube_path = os.path.join(output_path, 'cube.npy')
np.save(cube_path, cube)
print(cube.shape, ' cube saved:', cube_path)
# -

# ## v2

# +
# Load image cube
cube = []
for k, info in enumerate(steps):
    image = load_image(info['img path'], verbose=False)
    cube.append(image)
    
cube = np.dstack( cube )
print('cube shape:', cube.shape, f'{cube.nbytes // 1024**2}Mo')
# -

plt.figure(); plt.title('cube std');
plt.imshow(np.std(cube, axis=2));
plt.savefig(os.path.join(output_path, '01_cube_std.png'));

# +
# Define the grid
reference_image = 6

grid = build_grid(cube.shape, margin=100, spacing=300)
x_flat, y_flat = grid[0].flatten(), grid[1].flatten()

# Graph the grid
plt.figure();
plt.imshow(cube[:, :, reference_image]); plt.title(f'grille - image {reference_image}');
plt.plot(*grid, 'o', color='white', markersize=3);

for k, (x, y) in enumerate(zip(x_flat, y_flat)):
    text_offset = 10.0
    plt.text(x+text_offset, y+text_offset,
             str(k), fontsize=7, color='white')
       
plt.savefig(os.path.join(output_path, '02_grille.png'));
# -

point_idx = 10
x, y = x_flat[point_idx], y_flat[point_idx]
print(x, y)
I = cube[:, :, reference_image]

# +
window_half_size = 70
offsets = np.zeros((cube.shape[2], 2))
step1 = np.zeros((cube.shape[2], 2))

# go forward, image_by_image
dx_ref, dy_ref = 0, 0
for k in range(reference_image+1, cube.shape[2]):
    J = cube[:, :, k]
    try:
        dx_ref, dy_ref, error = get_shifts(I, J, x, y,
                                   offset=(dx_ref, dy_ref),
                                   window_half_size=window_half_size,
                                   upsample_factor=20)
        offsets[k] = [dx_ref, dy_ref]

        previous = cube[:, :, k-1]
        dx1, dy1, error = get_shifts(previous, J, x+dx_ref, y+dy_ref,
                                       offset=(0, 0),
                                       window_half_size=window_half_size,
                                       upsample_factor=20)
        step1[k] = [dx1, dy1]
    except ValueError:
        print('out of limits', k)
        step1[k] = [np.NaN, np.NaN]
        
# go backward, image_by_image
dx, dy = 0, 0
for k in range(0, reference_image)[::-1]:
    J = cube[:, :, k]
    dx, dy, error = get_shifts(I, J, x, y,
                               offset=(dx, dy),
                               window_half_size=window_half_size,
                               upsample_factor=20)
    offsets[k] = [dx, dy]
    
# -

plt.figure(figsize=(4, 4)); plt.title('trajectoire')
plt.plot( *offsets[:15].T, '-o', label='ref. -> k' );
plt.plot( *np.cumsum(step1[:15], axis=0).T, '-xr', label='k-1 -> k' );
plt.axis('equal'); plt.legend();
plt.plot(0, 0, 's'); plt.xlabel('x'); plt.ylabel('y');

plt.figure(figsize=(4, 4)); plt.title('trajectoire')
plt.plot( *offsets.T, '-o', label='ref. -> k' );
plt.plot( *np.cumsum(step1, axis=0).T, '-or', label='k-1 -> k ' );
plt.axis('equal');
plt.plot(0, 0, 's'); plt.xlabel('x'); plt.ylabel('y');

# +
offsets = np.zeros((*grid[0].shape, cube.shape[2], 2))

# go forward, image_by_image

for ij in range(x_flat):
    dx_ref, dy_ref = 0, 0
    for k in range(reference_image+1, cube.shape[2]):
        J = cube[:, :, k]
        dx_ref, dy_ref, error = get_shifts(I, J, x, y,
                                   offset=(dx_ref, dy_ref),
                                   window_half_size=35,
                                   upsample_factor=10)
        
        i, j = np.unravel_index(ij, grid_x.shape)
        offsets[i, j, k, :] = [dx_ref, dy_ref]

# -

# ## Test correlation

# +
# init the loop
info_J = steps[3]
J = ft.load_image( info_J['img path'] )

# Define the grid
grid = build_grid(J.shape, margin=100, spacing=25)
x, y = grid[0].flatten(), grid[1].flatten()

I = J
info_I = info_J

info_J = steps[8]
J = ft.load_image( info_J['img path'] )

# Diff consecutive images
shift_x, shift_y, errors = compute_shifts(I, J, grid, 
                                          window_half_size=50)

# Fit
eps, residuals_x, residuals_y = bilinear_fit(x, y, shift_x, shift_y)
# -

plt.imshow(I);plt.colorbar();

# +
plt.imshow(shift_y);
plt.colorbar(); #plt.clim([1, 14])
plt.title('hpr1 - dy entre 0.8 et 0%');
plt.xlabel('i'); plt.ylabel('j');

plt.figure();
plt.imshow(residuals_y, cmap='Spectral');
c_lim = 3*np.std(residuals_y)
plt.colorbar(); plt.clim([-c_lim, +c_lim])
plt.title('hpr1 - dy entre 0.8 et 0% - résidu fit linéaire');
plt.xlabel('i'); plt.ylabel('j');
# -

plt.title('hpr1 - dy entre 0.8 et 0% - profils');
plt.plot(shift_y[:, 19], label='i=19')  # pour hs2  x=15
plt.plot(shift_y[:, 35], label='i=35')
plt.xlabel('j'); plt.ylabel('dy (px)'); plt.legend();

plt.plot(shift_y[:, 15])
plt.plot(shift_y[:, 25])



plt.imshow(residuals_y, cmap='Spectral');
c_lim = 3*np.std(residuals_y)
plt.colorbar(); plt.clim([-c_lim, +c_lim])


def graph_field(ax, field, name):
    color_limits = np.std(field)*5.
    ax.imshow(field.reshape(grid[0].shape),
              cmap='bwr',
              clim=(-color_limits, +color_limits));
    #plt.colorbar();
    ax.set_title(name);


ax = plt.subplot(1, 1, 1)
graph_field(ax, shift_y, 'y')

# ## Loop

# +
# init the loop
info_J = steps[0]
J = ft.load_image( info_J['img path'] )

# Define the grid
grid = build_grid(J.shape, margin=100, grid_spacing=30)
x, y = grid[0].flatten(), grid[1].flatten()

output_data = []

# loop
for k in range(1, len(steps)):
    I = J
    info_I = info_J
    info_J = steps[k]
    J = ft.load_image( info_J['img path'] )
    img_name = info_J['stepname']
    print(f"image {img_name}", end='\n')
    
    # Diff consecutive images
    shift_x, shift_y, errors = compute_shifts(I, J, grid, 
                                              window_half_size=45)
    
    
    # Fit
    eps, residuals_x, residuals_y = bilinear_fit(x, y, shift_x, shift_y)
    
    data_dict = {'eps':eps,
                 'shift_x':shift_x,
                 'shift_y':shift_y,
                 'shift_error':errors,
                 'residuals_x':residuals_x,
                 'residuals_y':residuals_y,
                 'info_I':info_I,
                 'info_J':info_J
                }
    output_data.append(data_dict)
    
print('..done..')

# +
applied_strain = [step['strain'] for step in steps[1:]]

delta_eps_x = np.array([d['eps'][0] for d in output_data])
eps_x = np.cumsum(delta_eps_x)*100 # %
delta_eps_y = np.array([d['eps'][1] for d in output_data])
eps_y = np.cumsum(delta_eps_y)*100 # %

plt.plot(applied_strain, eps_y, '-o');
plt.xlabel('applied strain (%)');
plt.ylabel('eps_yy (%)');

plt.figure();
plt.plot(applied_strain, eps_x, '-o');
plt.xlabel('applied strain (%)');
plt.ylabel('eps_xx (%)');
# -

# ## Create figures

output_dir = os.path.join(output_path, 'displacement')
ft.create_dir(output_dir)

grid[0].shape

residuals.shape

residuals = output_data[1]['residuals_y']
pcm = plt.pcolormesh(*grid, residuals.reshape(grid[0].shape),
                     cmap='Reds');

for k, data in enumerate(output_data):
    displacement_graph(grid, data, number=k,
                       samplename=sample_name, fit=True,
                       save=True, save_dir=output_dir)


def displacement_graph(grid, data, samplename='',
                       save=False, number=None, tag=None, fit=False,
                       save_dir=None):

    
    if fit:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    tagstr = f" ({tag})" if tag else ''
    numberstr = f'#{number:03d}' + ' '*5 if number is not None else ''
    samplenamestr = 'sample: '+ samplename + ' '*5 if samplename else ''
    title = f"{samplenamestr}{numberstr}{data['info_I']['stepname']} → {data['info_J']['stepname']} {tagstr}"
    fig.suptitle(title, fontsize=16)

    # displacement X
    ax1.set_title('displacement x')
    color_limits = np.std(data['shift_x'])*2.5
    pcm = ax1.pcolormesh(*grid, data['shift_x'].reshape(grid[0].shape),
                         cmap='bwr',
                         vmin=-color_limits, vmax=+color_limits);
    ax1.set_ylim(ax1.get_ylim()[::-1]);
    ax1.set_xlabel("x (px)"); ax1.set_ylabel("y (px)");
    ax1.set_aspect('equal');
    fig.colorbar(pcm, ax=ax1)
    
    # displacement Y
    color_limits = np.std(data['shift_y'])*2.5
    ax2.set_title('displacement y')
    ax2.pcolormesh(*grid, data['shift_y'].reshape(grid[0].shape),
                   cmap='bwr',
                   vmin=-color_limits, vmax=+color_limits);
    fig.colorbar(pcm, ax=ax2);
    ax2.set_aspect('equal');
    ax2.set_xlabel("x (px)"); ax2.set_ylabel("y (px)");
    ax2.set_ylim(ax2.get_ylim()[::-1]); # reverse axis
    
    # Fit
    if fit:
        text = f"""Least-square fit with a plane \n
$\\varepsilon_{{xx}}$ = {data['eps'][0]*100:.3f}%
$\\varepsilon_{{yy}}$ = {data['eps'][1]*100:.3f}%
$\\varepsilon_{{xy}}$ = {data['eps'][2]*100:.3f}%"""
        ax3.text(.1, .4, text, fontsize=14)
        ax3.axis('off')

        residuals = data['residuals_y']
        #np.sqrt( data['residuals_x']**2 + data['residuals_y']**2 )
        color_limits = np.std(residuals)*2
        ax4.set_title(f'residual Y displacement (px)  max:{np.max(residuals):.1f}px')
        pcm = ax4.pcolormesh(*grid, residuals.reshape(grid[0].shape),
                             cmap='bwr',
                             vmin=-color_limits, vmax=+color_limits);
        fig.colorbar(pcm, ax=ax4);
        ax4.set_xlabel("x (px)"); ax4.set_ylabel("y (px)");
        ax4.set_ylim(ax4.get_ylim()[::-1]); # reverse axis
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    if save:
        assert save_dir is not None
        numberstr = f'{number:03d}' if number is not None else ''
        tagstr = f"{tag}_" if tag else '_'
        figfilename = f"{numberstr}{tagstr}{data['info_I']['stepname']}-{data['info_J']['stepname']}.png"
        
        output_path = os.path.join(save_dir, figfilename)
        print(figfilename, "saved in", output_path)

        fig.savefig(output_path)
        
        plt.close()



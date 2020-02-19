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
#     display_name: py3 venv
#     language: python
#     name: py3
# ---

# # Strechable Corr

import numpy as np
import matplotlib.pylab as plt
from skimage import io
from skimage import img_as_uint

# ## Search and select images

# %load_ext autoreload
# %autoreload 2
import filetools as ft
import os
from tabulate import tabulate

data_dir = "./images"

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

# ## Set output directory

# +
output_dir = "./output"

output_path = os.path.join(output_dir, sample_name)
ft.create_dir(output_path)

# + [markdown] toc-hr-collapsed=false
# ## Export selected images
# -

image_dir = os.path.join(output_path, 'images')
ft.create_dir(image_dir)

# Check image histogram
image = ft.load_image(steps[1]['img path'])
plt.hist(image.flatten(), bins=100);
plt.xlabel('intensity value'); plt.ylabel('pixel counts');

# Rescale intensities
intensity_high = 3900
intensity_low = 100

from strechablescorr import *

# Export images
for k, info in enumerate(steps):
    image_filename = f"{k:03d}_{info['stepname']}.tiff"
    image = ft.load_image(info['img path'])
    image = colorize_image(image, intensity_low, intensity_high)
    image = img_as_uint(image)
    img_path = os.path.join(image_dir, image_filename)
    io.imsave(img_path, image)
    print(f'save {img_path}', ' '*10, end='\r')
print('done', ' '*40)

# ## Test correlation

# +
# init the loop
info_J = steps[0]
J = ft.load_image( info_J['img path'] )

# Define the grid
grid = build_grid(J.shape, margin=100, grid_spacing=25)
x, y = grid[0].flatten(), grid[1].flatten()

I = J
info_I = info_J

info_J = steps[1]
J = ft.load_image( info_J['img path'] )

# Diff consecutive images
shift_x, shift_y, errors = compute_shifts(I, J, grid, 
                                          window_half_size=45)

# Fit
eps, residuals_x, residuals_y = bilinear_fit(x, y, shift_x, shift_y)
# -

print(eps)

plt.imshow(residuals_y)
plt.colorbar()


def graph_field(ax, field, name):
    color_limits = np.std(field)*5.
    ax.imshow(field.reshape(grid[0].shape),
              cmap='bwr',
              clim=(-color_limits, +color_limits));
    #ax.colorbar();
    ax.set_title(name);


ax = plt.subplot(1, 1, 1)
graph_field(ax, shift_x, 'x')

# ## Loop

# +
# init the loop
info_J = steps[0]
J = ft.load_image( info_J['img path'] )

# Define the grid
grid = build_grid(J.shape, margin=100, grid_spacing=50)
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
create_dir(output_dir)

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

        residuals = np.sqrt( data['residuals_x']**2 + data['residuals_y']**2 )
        color_limits = np.std(residuals)*4
        ax4.set_title(f'norm of residual displacement (px)  max:{np.max(residuals):.1f}px')
        pcm = ax4.pcolormesh(*grid, residuals.reshape(grid[0].shape),
                             cmap='Reds',
                             vmin=0, vmax=+color_limits);
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

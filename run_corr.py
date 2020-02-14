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

# +
from glob import glob
import os
import re
from tabulate import tabulate

from collections import defaultdict
from itertools import groupby

from skimage import io
import numpy as np
import matplotlib.pylab as plt
# -

DATA_DIR = "./images"
IMAGE_EXT = "TIF"


# ## List all images and parse filename

def parse_path(img_path, verbose=False):
    info = defaultdict(str)
    info['path'] = img_path
    # pre - processing
    img_path = img_path.replace(DATA_DIR, '')
    img_path = img_path.replace(IMAGE_EXT, '').strip('.')
    img_path = img_path.replace("\\", "/") # for windows
    img_path = img_path.strip("/")

    parts = img_path.split("/")
    if len(parts) != 3:
        if verbose:
            print("dir structure error:", parts)
        info['msg'] += "dir structure error "
        return info
    
    parts = [s.lower() for s in parts]
    sample_name, step_name, image_name = parts

    info['sample'] = sample_name
    info['step_name'] = step_name

    if not image_name.startswith(sample_name) or step_name not in image_name:
        if verbose:
            print('warning: no in prefix in filename', parts)
        info['msg'] += "warning: no in prefix in filename "
    
    if 'u' in step_name:
        info['direction'] = "unloading"
    else:
        info['direction'] = 'loading'
    
    strain = step_name.replace('u', '').replace('p', '.')
    try:
        info['applied_strain'] = float(strain)
    except ValueError:
        info['applied_strain'] = 0
        
    image_name = image_name.replace(sample_name, '')
    image_name = image_name.replace(step_name, '')

    image_name = image_name.replace('_', '')

    img_pattern = re.compile( r'(u?)(\D*)(\d*)' )
    matchs = re.findall(img_pattern, image_name)

    if matchs:
        m = matchs[0]
        #info['direction'] = "unloading" if m[0] == 'u' else 'loading'
        info['tag'] = m[1]
        info['file_idx'] = int(m[-1])
    else:
        if verbose:
            print("filename error:", parts)
        info['msg'] += "filename error: " + parts

    info['label'] = f'{sample_name} {step_name}'#' {image_name}'
    return info


def load_image(path):
    """ returns 2d array
    """
    try:
        I = io.imread(path)
        # convert to grayscale if needed:
        I = I.mean(axis=2) if I.ndim == 3 else I  
    except FileNotFoundError:
        print("File %s Not Found" % path)
        I = None

    return I


# ## list all images

# +
pattern = os.path.join(DATA_DIR, "**/*.%s" % IMAGE_EXT)
all_path = glob(pattern, recursive=True)

images_info = []
for img_path in all_path:
    info = parse_path(img_path)
    images_info.append(info)
    #if info['msg']:
    #    print(info)
# -

samples = set(info['sample'] for info in images_info)
print(samples)

# ## Sample selection

# +
# Selection and sort
sample = 'ss2'
direction = 'loading'

selection = [info for info in images_info
             if info['sample'] == sample
             and info['direction'] == direction]

groups = groupby(selection, key=lambda x:x['step_name'])

# take first element for each step
selection = [next(group) for key, group in groups]

selection = sorted(selection, key=lambda x:x['applied_strain'])

print("nbr images: ", len(selection))
# -

# ### Show selected images

# Show selected image
show = input('show %i images? [no]'% len(selection))
if show:
    for info in selection:
        I = load_image(info['path'])
        plt.figure(figsize=(12, 6));
        plt.title(info['label']);
        plt.imshow(I);

# ### Correlation

from strechablescorr import *


def displacement_graph(grid, shift_x, shift_y, info_I, info_J,
                       save=False, number=None, tag=None, fit=False,
                       save_dir=None):
    # grid, shift_x, shift_y, info_I, info_J
    tagstr = f" ({tag})" if tag else ''
    
    if fit:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
    title = f"{number:03d}  {info_I['label']} → {info_J['label']} {tagstr}"
    fig.suptitle(title, fontsize=16)

    ax1.set_title('displacement x')
    color_limits = np.std(shift_x)*2.5
    pcm = ax1.pcolormesh(*grid, shift_x.reshape(grid[0].shape),
                         cmap='bwr',
                         vmin=-color_limits, vmax=+color_limits);
    ax1.set_ylim(ax1.get_ylim()[::-1]);
    ax1.set_xlabel("x (px)"); ax1.set_ylabel("y (px)");
    ax1.set_aspect('equal');
    fig.colorbar(pcm, ax=ax1)

    color_limits = np.std(shift_y)*2.5
    ax2.set_title('displacement y')
    ax2.pcolormesh(*grid, shift_y.reshape(grid[0].shape),
                   cmap='bwr',
                   vmin=-color_limits, vmax=+color_limits);
    fig.colorbar(pcm, ax=ax2);
    ax2.set_aspect('equal');
    ax2.set_xlabel("x (px)"); ax2.set_ylabel("y (px)");
    ax2.set_ylim(ax2.get_ylim()[::-1]); # reverse axis

    # Fit
    if fit:
        x = grid[0].flatten()
        y = grid[1].flatten()
        (eps_x, eps_y, eps_xy), residuals = bilinear_fit(x, y, shift_x, shift_y)
        text = f"""Least-square fit with a plane \n
eps_xx = {eps_x*100:.3f}%
eps_yy = {eps_y*100:.3f}%
eps_xy = {eps_xy*100:.3f}%"""
        ax3.text(.1, .4, text, fontsize=14)
        ax3.axis('off')

        color_limits = np.std(residuals)*4
        ax4.set_title(f'norm of residual displacement (px)  max:{np.max(residuals):.1f}px')
        pcm = ax4.pcolormesh(*grid, residuals.reshape(grid[0].shape),
                             cmap='Reds',
                             vmin=0, vmax=+color_limits);
        fig.colorbar(pcm, ax=ax4);
        ax4.set_xlabel("x (px)"); ax4.set_ylabel("y (px)");
        ax4.set_ylim(ax4.get_ylim()[::-1]); # reverse axis

    # Save
    if save:
        numberstr = f"{number:03d}" if number else ''
        tagstr = f"{tag}_" if tag else '_'
        figfilename = f"{numberstr}{tagstr}{info_I['step_name']}-{info_J['step_name']}.png"
        
        output_path = os.path.join(save_dir, figfilename)
        print(figfilename, "saved in", output_path)

        fig.savefig(output_path)
        
        plt.close()


# !mkdir output

# +
OUTPUT_DIR = "./output"
output_path = os.path.join(OUTPUT_DIR, sample)
if not os.path.isdir(output_path):
    os.mkdir(output_path)
    print("create", output_path)
    
print('output dir:', output_path)

# +
# init the loop
info_J = selection[0]
J = load_image( info_J['path'] )

# Define the grid
grid = build_grid(J.shape, margin=100, grid_spacing=20)
x, y = grid[0].flatten(), grid[1].flatten()

#data = [{'image name':get_image_name(image_paths[0])}]
shift_x_cumulative = np.zeros(x.shape)
shift_y_cumulative = np.zeros(x.shape)

# loop
for k in range(1, len(selection)):
    I = J
    info_I = info_J
    info_J = selection[k]
    J = load_image( info_J['path'] )
    img_name = info_J['label']
    print(img_name+'...', end='\r')
    
    # Diff consecutive images
    shift_x, shift_y, errors = compute_shifts(I, J, grid, 
                                              window_half_size=45)
    
    
    #mean_disp = np.mean( np.sqrt(shift_x**2 + shift_y**2) )
    #eps, residuals = bilinear_fit(x, y, shift_x, shift_y)

    displacement_graph(grid, shift_x, shift_y, info_I, info_J,
                       save=True, save_dir=output_path,
                       number=k, tag=None, fit=True)
   
    
    #shift_x_cumulative += shift_x
    #shift_y_cumulative += shift_y
    #
    #displacement_graph(grid, shift_x_cumulative, shift_y_cumulative, info_I, info_J,
    #                   save=True, fit=True, save_dir=output_path,
    #                   number=k, tag='cumulative')
    
print('..done..')
# -

# # done 
# hpr1
# hs2
# spp1
# spp2
# ss3
# ss2



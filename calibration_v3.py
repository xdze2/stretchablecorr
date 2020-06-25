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
from strechablescorr import *
from glob import glob
import os, sys

# List sample dirs
dirs = glob('./images/**/', recursive=True)
print('\n'.join([f'{k} - {n}' for k, n in enumerate(dirs)]))

# List images
dir_choice = 3
print(dirs[dir_choice])
images_list = glob(os.path.join(dirs[dir_choice], '*'))
images_list = sorted(images_list)
print('\n'.join([f'{k} - {n}' for k, n in enumerate(images_list)]))

# +
# ==========================
#  Charge 2 images: A et B
# ==========================
path_A = images_list[0]  # 7, 8 -- 3, 4 no move
path_B = images_list[3]

image_A = load_image(path_A)
image_B = load_image(path_B)

# +
#from scipy.ndimage import gaussian_filter
#sigma = 10
#image_A = gaussian_filter(image_A, sigma=sigma)
#image_B = gaussian_filter(image_B, sigma=sigma)

# +
# look for global displacement 
global_dx, global_dy, error = get_shifts(image_A, image_B, *np.array(image_A.shape)//2,
                           window_half_size=5000,
                           upsample_factor=5)
print('Déplacement global:', f"dx={dx}px, dy={dy}px")

# Graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
ax1.imshow(image_A); ax1.set_title('image A');
ax1.plot(*np.array(image_A.T.shape)//2, 's', color='white', markersize=2)
ax2.imshow(image_B); ax2.set_title('image B');
ax2.plot(*np.array(image_A.T.shape)//2, '.', color='white', markersize=2)
ax2.plot(*np.array(image_A.T.shape)//2 + np.array([global_dx, global_dy]), 's',
         color='white', markersize=2)
plt.savefig('01_images_AB.png');

# +
# =============
#  Paramètres
# =============
window_half_size = 40   # Taille des carrés utilisés pour la corrélation
spacing = 80  # Distance entre les points de la grille

# Construit une grille de points:
margin = max(abs(global_dx), abs(global_dy))*1.05 + window_half_size
grid_x, grid_y = build_grid(image_A.shape, margin=margin, spacing=spacing)

# Graph
plt.figure();
plt.imshow(image_A); plt.title('grille - image A');
plt.plot(grid_x, grid_y, 'o', color='white', markersize=2);
box = np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])*(window_half_size + 1)
middle_point = tuple(np.array(grid_x.shape) // 2 - 1)
plt.plot(box[0]+grid_x[middle_point], box[1]+grid_y[middle_point],
         color='white', linewidth=1)
plt.savefig('02_grille.png');

# +
# ==========================
#  Calcul des déplacements
# ==========================

print('Calcul du déplacement pour chaque point de la grille:')
shift_x, shift_y, errors = compute_shifts(image_A, image_B, (grid_x, grid_y), 
                                          window_half_size=window_half_size,
                                          upsample_factor=100,
                                          offset=(global_dx, global_dy))

print('  translation:', 'tx=', shift_x.mean())
print('              ', 'ty=', shift_y.mean())

# Graph
r = np.sqrt(shift_x**2 + shift_y**2)
plt.figure();
plt.imshow(image_A);
plt.quiver(grid_x, grid_y, shift_x, shift_y,
           angles='xy', color='white');
plt.title(f'champ de déplacement - r_avg={r.mean():.3f}px');

# +
# ============================
#  Soustraire la translation
# ============================
delta_x = shift_x - np.mean( shift_x )
delta_y = shift_y - np.mean( shift_y )

print("Champ de déplacement sans la translation:")
print(' Moyenne:', np.mean(delta_x), np.mean(delta_y))
print(' Écart type:', np.std(delta_x), np.std(delta_y))

# Graph
plt.figure();
plt.imshow(image_A);
plt.quiver(grid_x, grid_y, delta_x, delta_y, angles='xy',
           color='white');
plt.title('champ de déplacement\n sans la translation (A->B)');
plt.savefig('03_champ_de_deplacement.png');
# -

plt.figure(figsize=(3, 3)); plt.title('all grid points')
plt.plot(delta_x.flatten(), delta_y.flatten(), 'o', alpha=0.2)
plt.axvline(x=0, linewidth=0.5, color='black');
plt.axhline(y=0, linewidth=0.5, color='black');
plt.axis('equal');
plt.xlabel('dx'); plt.ylabel('dy');

# +
# ============================
#  Correction de la rotation
# ============================

print('Champ de déplacement sans translation ni rotation:')

from scipy.optimize import least_squares

def rigid_body_motion(xy, tr_x, tr_y, theta):
    # Matrice de rotation
    R = np.array([[+np.cos(theta), -np.sin(theta)],
                  [+np.sin(theta), +np.cos(theta)]])
    xy_prime = np.matmul(R, xy) + np.array([[tr_x,], [tr_y, ]])
    return (xy_prime - xy)

x_flat, y_flat, delta_x_flat, delta_y_flat = [u.flatten()
                                              for u in (grid_x, grid_y, delta_x, delta_y)]

xy = np.vstack([x_flat, y_flat])
u = np.vstack([delta_x_flat, delta_y_flat])

def cost(parameters):
    tr_x, tr_y, theta = parameters
    u_prime = rigid_body_motion(xy, tr_x, tr_y, theta)
    delta = (u_prime - u)
    return delta.flatten()

tr_x, tr_y = 0, 0
res = least_squares(cost, [tr_x, tr_y, 0])
print(' ...', res.message)
print(' final cost value:', res.cost)
print(' optim:', res.x)

theta_deg  = res.x[2]*180/np.pi
print(' theta (deg):', theta_deg)

# +
u_rigid = rigid_body_motion(xy, *res.x)

residus = u - u_rigid

residus_x = residus[0, :].reshape(grid_x.shape)
residus_y = residus[1, :].reshape(grid_y.shape)
residus_r = np.sqrt( residus_x**2 + residus_y**2 )

print('Moyenne:', np.mean(residus, axis=1))
print('Écart type:', np.std(residus, axis=1))

plt.figure();
plt.imshow(image_A);
plt.quiver(grid_x, grid_y, residus_x, residus_y, angles='xy',
           color='white');
plt.title(f'champ de déplacement sans translation \n ni rotation. écart type={np.std(residus_r):.3}px');
plt.savefig('04_champ_de_deplacement_sans_rotation.png');
# -
plt.figure(figsize=(3, 3)); plt.title('all grid points')
plt.plot(residus_x.flatten(), residus_y.flatten(), 'o', alpha=0.2)
plt.axvline(x=0, linewidth=0.5, color='black');
plt.axhline(y=0, linewidth=0.5, color='black');
plt.axis('equal');
plt.xlabel('dx'); plt.ylabel('dy');


# +
# ========================================
# Acquisition noise (without correction)
# ========================================

diff = (image_A - image_B)
diff -= np.mean(diff)

plt.hist(diff.flatten(), bins=71);
# -

plt.imshow(diff);
clim = 2*np.std(diff.flatten())
plt.clim(-clim, clim)



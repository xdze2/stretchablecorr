# stretchable corr

a Python code for Digital Image Correlation (DIC)

- **local method**: the displacemenent field between images is obtained using local cross-correlation computation. The function [`phase-cross-correlation`](https://scikit-image.org/docs/stable/api/skimage.registration.html#phase-cross-correlation) from scikit-image is used. Sub-pixel precision is obtained employing an upsampled matrix-multiplication DFT [1].

[1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, “Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156


## Code structure

There are 3 modules:
- filetools: functions used to load and sort images
- stretchablecorr: main set of function performing the processing
- graphtools: post-processing functions


## Workflow

- Load images from a given directory, sort by alphabetical order: get an image cube.
- Construct a regular grid (provide spacing & margin)
- Run the cross-correlation for every points:
    - Lagragian (surface) vs Eulerian (lab.) ref. frame ?
    - tinkering to obtain more robust machinery
- Post-process, graph and think


## Data structures

* `cube` : 3D array of floats with shape (nbr_images, height, width)  
    Sequence of gray-level images.  
    _note: ij convention instead of xy_

* `points` : 2D array of floats with shape (nbr_points, 2)  
    Coordinates of points (could be unstructured, i.e. not a grid).  
    (x, y) order

* `offsets` : 2D array of floats with shape (nbr_images - 1, 2)   
    image-to-image overall displacement  
    could include NaNs

* `displ_from_previous` : 3D array of floats with shape (nbr_points, 2, nbr_images - 1)  
    image-to-image displacements for each points (Eulerian)  
    could include NaNs

## pseudo multi-scale approach for robust processing


>_eternal question:_ graph while computing or store and post-process: how long is the run ? how big are the data ? are the data needed afterwards ?

> 2nd eternal question : loop order ? image then points, or points then images ?  now it is points then images - to allow unstructured points

> 3rd eternal question : Dimension order ? numpy loops and unpacks along first dim, so first dim is the outer loop -- here points

    for image in cube:
        ...

* first, run correlation image-to-image on a large ROI (i.e. the central part of the image) → obtain `offsets` values
* second, run correlation image-to-image for all points of the grid, (using the offsets) → obtain `displ_from_previous` values 
    - run bilinear fit to get sample-scale Eulerian image-to-image deformations (`lin_def_from_previous` and `residuals`)
* third, re-run correlation tracking points: get Lagrangian (attached to the surface) 
    - for this we need an image range -> it defines a the sample surface area visible from start to finish. Problems: it will be not necessaryly stay a regular and nice grid 


## Next

- Global (high order method)
- Error estimation using Likelihood approach 

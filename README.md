# stretchable corr

a Python code for Digital Image Correlation (DIC)

- **local method**: the displacemenent field between images is obtained using local cross-correlation computation. The function [`phase-cross-correlation`](https://scikit-image.org/docs/stable/api/skimage.registration.html#phase-cross-correlation) from scikit-image is used. Sub-pixel precision is obtained employing an upsampled matrix-multiplication DFT [1].

[1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, “Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156


## Code structure

There are 3 modules:
- filetools: functions used to load and sort images
- stretchablecorr: main set of function performing the processing
- graphtools: post-processing functions

_note:_ [numpy's style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) of docstrings is used

## Workflow

- Load images from a given directory, sort by alphabetical order: get an image cube.
- Construct a regular grid (provide spacing & margin)
- Run the cross-correlation for every points:
    - Lagragian (surface) vs Eulerian (lab.) ref. frame ?
    - tinkering to obtain more robust machinery
- Post-process, graph and think


## Displacement field description 

[Eulerian and Lagragian](https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field) are two different way to describe the displacement field (or flow field) depending on which frame of reference is used:
- *Eulerian*: the laboratory or the camera field of view is used as reference, i.e. field evaluation points remain fixed on the images.
- *Lagrangian*: points on the sample surface are tracked. The frame of reference is fixed to the sample surface.

_Eulerian description_ corresponds to the simplest data processing approach, whereas _Lagrangian description__ require more complex data processing. 

similar for small displacement

Another important consideration is related to the choice of the reference state. The displacement field is defined relatively to a non-deformed state. This, usualy, correpsonds to the first image of the sequence. 

Then, each correlation, should be computed between the image `i` and the reference image. However, large displacement or deformation could occur between these two images, leading to a wrong correlation estimation. Performing the correlation image-to-image is more robust albeit leading to the summation of correlation errors.

Thus there are, at least, four different combinaison to estimate the displacement field: either Eulerian or Lagrangian, and image-to-image or image-to-reference correlation.

Only two of the combinaison are used in practice: Eulerian image-to-image, and Lagrangian image-to-reference

--
High order correlation method (for instance global DIC) are used to reduce the correlation error in the image-to-reference case.

We could think of...



**Multiscale approachs (pyramids):**
Similarly to iterative optimisation method where the choice of the initial guess is important. The two images to be correlated have to be similar enough. When large displacement (>>50 pixels, larger the the ROI window size) occurs mutli-step methods are used. 

-> quad-tree decomposition 1, 1/4, 1/16

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
* third, (re-)run correlation now by tracking individuals points: get Lagrangian (attached to the surface) deformation field 
    - for this we need an image range -> it defines a the sample surface area visible from start to finish. Problems: it will be not necessaryly stay a regular and nice grid

There are many ways to do this:  
* sum image-to-image displacement (use previous position as offset)
* run correlation with ref. image (use previous position as offset)
* ... mix the two, mix all possible duo of images

## Next

- Global (high order method)
- Error estimation using Likelihood approach 

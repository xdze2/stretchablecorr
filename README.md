# stretchable corr

a Python code for Digital Image Correlation (DIC)

- **local method**: the displacemenent field between images is obtained using local cross-correlation computation. The function [`phase-cross-correlation`](https://scikit-image.org/docs/stable/api/skimage.registration.html#phase-cross-correlation) from scikit-image is used. Sub-pixel precision is obtained employing an upsampled matrix-multiplication DFT [1].

[1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, “Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156

The method used here corresponds to the figure (d) below:


![deformation order](./schema/def_states.png)


## Workflow

- Load images from a given directory, sort by alphabetical order: get an **image sequence**.
- Construct a regular **grid** with a given spacing and margin value.
- Compute **cross-correlation** for every points[*]:
  * Using **Eulerian** reference frame i.e. points remain fixed to the camera frame.
  * Using **Lagrangian** reference i.e. points are attached to the sample surface (track points).
- Post-process, **graph** and think

[*] and image-to-image vs image-to-reference (see below)

## Tips 

- Extract images from video using [`ffmpeg`](https://ffmpeg.org/):

        ffmpeg -i test\ 2\ input_file.avi -qscale:v 4  ./output/output_%04d.tiff

- When naming the images, use number padding (0001.jpg, 0002.jpg, ...etc) to keep the correct order (use alphabetical sort)

- Record as many as possible images

## Displacement field description 

[Eulerian and Lagragian](https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field) are two different way to describe the displacement field (or flow field) depending on which frame of reference is used:
- *Eulerian*: the laboratory or the camera field of view is used as reference, i.e. field evaluation points remain fixed on the images.
- *Lagrangian*: points on the sample surface are tracked. The frame of reference is fixed to the sample surface.

_Eulerian description_ corresponds to the simplest data processing approach, whereas _Lagrangian description_ require more complex data processing. Both are similar for small displacement.

## image-to-image vs image-to-reference

Another important consideration is related to the choice of the **reference state**. The displacement field is defined relatively to a non-deformed state. This, usualy, corresponds to the first image of the sequence.


Ideally, displacement for image `i` are directly computed using correlation between image `i` and the reference image. This is the **image-to-reference** approach.

However, large displacement or deformation could occur between these two images, leading to a wrong correlation estimation or simply displacement larger than the window size. An other method is to perform correlations between susccesive images and integrate (sum) instantateous displacements to obtain the absolute displacement. Performing correlation **image-to-image** is more robust albeit leading to the summation of correlation errors.

Therefore, at least four different combinations are possible to estimate the displacement field:

* either Eulerian or Lagrangian, 
* and image-to-image (relative) or image-to-reference (absolute) correlations.

> For the moment, Image-to-reference is not used. Image-to-reference displacement field is obtained by summation (`cumsum()`) of the image-to-image displacement field.

_note:_ **High order** correlation methods (for instance global DIC) are used to reduce the correlation error in the image-to-reference case.

We could think of another approach...


## Data structures

* `cube` : 3D array of floats with shape (nbr_images, height, width)  
    Sequence of gray-level images.  
    _note: ij convention instead of xy_ --> height first

* `points` : 2D array of floats with shape (nbr_points, 2)  
    Coordinates of points (could be unstructured, i.e. not a grid).  
    (x, y) order

* `offsets` : 2D array of floats with shape (nbr_images - 1, 2)   
    image-to-image overall displacement  
    could include NaNs

* `displ_field` : 3D array of floats with shape (nbr_images - 1, nbr_points, 2)  
    generic displacements field  
    could include NaNs



## Pseudo multi-scale approach for robust processing

**Multiscale approachs (pyramids):**
Similarly to iterative optimisation methods where choice of initial guess is important. The two images to be correlated have to be similar enough. When large displacement (>>50 pixels, larger the the ROI window size) occurs mutli-step methods are used.

Here a pseudo-multi-scale approach is used:
* first, run correlation image-to-image on a large ROI (i.e. the central part of the image) → obtain `offsets` values
* second, run correlation image-to-image for all points of the grid, using offsets, a large window (100px) and no upsample → obtain a **coarse** displacement field
* third, (re-)run correlation using offset from the coarse displ. field, a smaller window size (35px) and a large enough upsample factor (20-100)


## The two important functions


* Eulerian image-to-image correlation

```python
displacements_img_to_img(
    images,
    points,
    window_half_size,
    upsample_factor,
    offsets=None,
    verbose=True,
    )
```

* Lagrangian image-to-image correlation

```python
track_displ_img_to_img(
    images,
    start_points,
    window_half_size,
    upsample_factor,
    offsets=None,
    verbose=True
    )
```


## Development & code structure

There are 3 modules:
- filetools: functions used to load and sort images
- stretchablecorr: main set of function performing the processing
- graphtools: post-processing functions (to be done)


### Documention (to be done)


_note:_ [numpy's style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) of docstrings is used

docstring to html is done using [pdoc](https://pdoc3.github.io/pdoc/)

    $ pdoc --html stretchablecorr.py



### note
>_eternal question:_ graph while computing or store and post-process: how long is the run ? how big are the data ? are the data needed afterwards ?

> 2nd eternal question : loop order ? image then points, or points then images ?  now it is points then images - to allow unstructured points

> 3rd eternal question : Dimension order ? numpy loops and unpacks along first dim, so first dim is the outer loop -- here points


## Next

- Allow rescaling: px -> mm/cm
- Error estimation
- Length scales estimation (PSD): ROI size choice
- Global (high order method)
- 




**Digital Image Correlation** is part of the more general problem of finding the best geometric transformation allowing to match two digital images. This class of problem is refered as _image registration_ and are encounter in various domains:

* Remote sensing community to study satellite images
* Medical imagery
* [Super-resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging)
* Image stacking (astronomy)
* [Particle image velocimetry](https://en.wikipedia.org/wiki/Particle_image_velocimetry)

Other problems in computer vision are also related:

* [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) used in computer vision. See for instance the [Lucas–Kanade method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method)
* feature matching
* [Stereophotogrammetry](https://en.wikipedia.org/wiki/3D_reconstruction_from_multiple_images) (3D reconstruction from a set of pictures) (i.e. stereoscopic vision)

--

The general formulation of the image registration is to find the transformation between the two images that optimizes a particular criterion: 

$$
\underset{\theta}{\text{argmin}} \;\big\langle A(x),\, B(\, \small{T_\theta(x)}\,) \big\rangle
$$

where $T_\theta(\bullet)$ is a family of geometric transformation function paramatrized by the parameter set $\theta$ and $\langle \bullet ,\, \bullet \rangle$ is a cost function. 

Possible transformations could be classified regarding their order:

- Constant: translation
  $u( x) = Const.$
- Linear: rotation, scaling and deformation
  $u(x) = C + M\times x$
- Non-linear: quadratic, splines... etc


![./files/def_states.png](./files/def_states.png)

Distinction between **local** and **global** methods are sometime used for Digital Image Correlation methods, which could be understood as low or high order methods. 

Any high order deformation could be reduced to a low order transformations provided that:

* small enough local parts of the image are considered ([ROI](https://en.wikipedia.org/wiki/Region_of_interest)) (Taylor decomposition)
* and that the two images are similar enough

One advantage of high order methods is that deformation field $\varepsilon_{ij}(x)$ is directly obtained, in addition to the displacement field $u_i(x)$. To numericaly derive the deformation from the estimated displacement field, interpolation and smoothing methods have to be used. 

Further hypothesis which are obvious in case of mechanical test analysis:

* the two observed sampled images represent the same scene
* obtained under similar conditions (same set-up and same camera)
* sampled on identical grids



### Next

* [math of images registration](./images_registration.html)
* [List of DIC softwares](./list_DICsoftwares.html)
* Bibliography on  DIC (specific to mecanical test)

* [Code documentation (docstrings)](./stretchablecorr/index.html)

### Some references 

- Pan, Bing, Kemao Qian, Huimin Xie, et Anand Asundi. « Two-Dimensional Digital Image Correlation for in-Plane Displacement and Strain Measurement: A Review ». Measurement Science and Technology 20, nᵒ 6 (1 juin 2009): 062001. https://doi.org/10.1088/0957-0233/20/6/062001.

- Stone, H.S., M.T. Orchard, Ee-Chien Chang, et S.A. Martucci. « A Fast Direct Fourier-Based Algorithm for Subpixel Registration of Images ». IEEE Transactions on Geoscience and Remote Sensing 39, nᵒ 10 (octobre 2001): 2235‑43. https://doi.org/10.1109/36.957286.



### Remarks

[1]  $T_\theta(B(x))$ could be written

[2] Continuous notation is used here $A(x)$. However, digital images are discrete. At the end, intensity of pixels $\mathtt A[i, j]$ will be used. Also, computing the cost function will require to obtain image intensity at non-integer pixel position. Displacement smaller than one pixel could be deduced with appropriate method (subîxel accuracy). For these reason, $A(x)$ refere to the theoretical "true" image, before the acquisition and the sampling. Interpolation and smoothing methods will be used to get an estimation of $A(x)$ from the actual measured image $\mathtt A[i, j]$

---
[← back to github page](https://github.com/xdze2/stretchablecorr)
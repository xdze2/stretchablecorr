# DIC - Introduction

At the reading of :

> Stone, H.S., M.T. Orchard, Ee-Chien Chang, et S.A. Martucci. « A Fast Direct Fourier-Based Algorithm for Subpixel Registration of Images ». IEEE Transactions on Geoscience and Remote Sensing 39, nᵒ 10 (octobre 2001): 2235‑43. https://doi.org/10.1109/36.957286.

**image registration**: aligns the pixels of one image to corresponding pixels of a second image, used in:

* Remote sensing community to study satellite images
* Medical imagery.
* Super-resolution
* Image stacking (astronomy)
* PIV

Other problems are related to that:

* [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) used in computer vision. See for instance the [Lucas–Kanade method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method)
* feature matching. 
* [Stereophotogrammetry](https://en.wikipedia.org/wiki/3D_reconstruction_from_multiple_images) (3D reconstruction from a set of pictures) (i.e. stereoscopic vision)

The general formulation of the image registration is to find the transformation between the two images that optimizes a particular criterion:

$$
\underset{\theta}{\text{argmin}} \;\big\langle A(x),\, B(\, \small{T_\theta(x)}\,) \big\rangle
$$

where $T_\theta(\bullet)$ is a family of transformation function paramatrized by the parameter set $\theta$ and $\langle \bullet ,\, \bullet \rangle$ is a cost function. 

> _note 1:_ we could also write $T_\theta(B(x))...$

> _note 2:_ Continuous notation is used here $A(x)$. However, digital images are discrete. At the end, intensity of pixels $\mathtt A[i, j]$ will be used. Also, computing the cost function will require to obtain image intensity at non-integer pixel position. Displacement smaller than one pixel could also be deduced with appropriate method. For these reason, $A(x)$ refere to the theoretical "true" image, before the acquisition and the sampling. Interpolation and smoothing methods will be used to get an estimation of $A(x)$ from the actual measured image $\mathtt A[i, j]$

Possible transformation could be classified regarding their order:

- Constant: translation
  $u( x) = Const.$
- Linear: rotation, scaling and deformation
  $u(x) = C + M\times x$
- Non-linear: quadratic, splines... etc

Some authors use the distinction between **local** and **global** methods for Digital Image Correlation methods, which I understand as low or high order methods. 

Any high order deformation could be reduced to a low order transformations provided that:

* small enough local parts of the image are considered ([ROI](https://en.wikipedia.org/wiki/Region_of_interest)) (Taylor decomposition)
* and that the two images are similar enough

One advantage of high order methods is to directly obtain the deformation field $\varepsilon_{ij}(x)$ in addition to the displacement field $u_i(x)$. To numericaly derive the deformation from the estimated displacement field, interpolation and smoothing methods have to be used. 

Further hypothesis:

* two observed sampled images represent the same scene
* obtain using similar condition (same set-up and camera)
* sampled on identical grids

### Next:

* [Mathematical analysis](https://)
* Bibliography on  DIC (specific to mecanical test)
* List of some DIC softwares

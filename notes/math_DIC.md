## DIC

- Presentation of the problem, pratical and theoretical

- Application, and link to other field

Mathematical notations adapted from the article:
G. Besnard, F. Hild, and S. Roux, “‘Finite-Element’ Displacement Fields Analysis from Digital Images: Application to Portevin–Le Châtelier Bands,” *Experimental Mechanics*, vol. 46, no. 6, pp. 789–803, Dec. 2006.

The reference image is noted $f(x)$ and the image corresponding to the deformed state is noted $g(x)$, where $x$ is the position in the camera reference frame (it is a 2D or 3D vector). Images are usually considered grayscale, i.e. $f(x)$ and $g(x)$ are scalar value. (*note:* the images are considered continious, i.e. coordinates of $x$ are reals, not pixel index. This means that an interpolation step is used somewhere.)

We are looking for the displacement field $u(x)$ such that 

$$
f( x + u{\small(x)} ) \approx g(x)
$$

The displacement field is solution of a minimization problem. The cost function $\eta$ is defined as:

$$
\eta^2 = \int_{\Omega}  C\!\left[ \; f(x+u{\small(x)}),\; g(x) \; \right] \,dx
$$

where $C[\cdot]$ is a correlation criterion (for instance Sum of Square Differences, normalized or zero-normalized). $\Omega$ is the studied domain (i.e. the visible part of the sample). By forgetting the normalization of the two images, the cost function writes:

$$
\eta^2 = \int_{\Omega}  \left[ \; f(x+u{\small(x)}) - g(x) \; \right]^2 \,dx
$$

*note:* Cross-correlation ($f\cdot g$) and square of the difference are related, when images are normalized (See review by B. Pan et al. 2009).  

*question:* Could the normalization be local or global? 

DIC algorithm involves the three following parts:

- **Discretization** of the integrale over the spatial domain $\Omega$: local/global
  continuity of the displacement field across adjacent elements is enforced for global element (FE), whereas elements are independant in local methods (point based, sparse)
  
  $$
  \eta^2 = \sum_i \int_{\Omega_i} \left[\, f(x+u_i{\small (x)})-g(x)\, \right]^2 \, dx
  $$

- The solution $u_i$ is approximated as a **parameterized function**, usually using an affine transformation (or higher order polynomial): (using notation from FEA)
  
  $$
  \hat u_i(x) = \sum_j a_{ij} \, \psi_{ij}(x)
  $$
  
  where $\psi_{ij}(\cdot)$ is the shape function $j$ for the element $i$.
  
  Another way to write the affine transformation is matrix notation: $u_i(x)=A \times x + d$... link with the strain tensor+translation+rotation?

- The specificity of DIC algorithm is mainly in how the minimization problem is solved.
  The first approach is to linearise the problem using a Taylor expansion: $f(x+u)\sim f(x) + f_x(x)\cdot u + ... $ provided that $u$ is small
  The definition of the gradient needs to define a spatial scale.
  This leads to methods involving image pyramid (or hiearchical) methods. Indeed, gradient estimation based on simple finite difference formula at pixel scale, i.e. $df/dx\sim f(x+1)-f(x)$, is valid only when the displacement amplitude $|u|$ is about one pixel. This is not usually the case. Moreover, the finite difference on experimentale data is very sensitive to noise. For this reason a smoothing method is usually applied prior to differentiation (for instance a moving average).    

- 

*note:* there are two discretizations, the pixels and the elements. How they are related? enforce that they are the same?
range: pixel<==>image size   and   ROI size <==>spatial res. in displacement field

More generaly, the minimization problem could be solved  

At the level $k+1$,  the following minimization is performed, based on transformation obtained on the previous level ($u_i^k$):

$$
u^{(k+1)} = \underset{u_i}{\operatorname{argmin}}\; \sum_i \int_{\Omega_i^k} [f(x)-g(x)+u_i(x)\cdot f_x^{k+1}(x) ]^2 \,dx 
$$

$f_x^{k+1}$ is the gradient of the image $f$ using spatial wavelength of the level $k+1$.

The integration domain used is the one deduced from the previous level. 

- Symmetrize f and g? $f\leftarrow (f+g)/2$ and $g \leftarrow (g-f)/2$





## Regularization

impose solution smoothness (in addition to continuity)
i.e. set a stiffness on the mesh... see [1]

## References

[1] F. Hild, B. Raka, M. Baudequin, S. Roux, and F. Cantelaube, “Multiscale displacement field measurements of compressed mineral-wool samples by digital image correlation,” *Applied Optics*, vol. 41, no. 32, p. 6815, Nov. 2002.

[2] G. Besnard, F. Hild, and S. Roux, “‘Finite-Element’ Displacement Fields Analysis from Digital Images: Application to Portevin–Le Châtelier Bands,” *Experimental Mechanics*, vol. 46, no. 6, pp. 789–803, Dec. 2006.

[3] J. Le Moigne, “Introduction to remote sensing image registration,” in *2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, Fort Worth, TX, 2017, pp. 2565–2568.

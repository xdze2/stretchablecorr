
# Images Registration

Least square cost function is usually used to compare the images. When only translation offset is searched between image blocks, i.e. zero-order registration case, minimizing least square cost function is equivalent to search for the maximum of the cross-correlation product. Then, the power of the Fast Fourier Transform algorithm (FFT) could used to speed-up the computation.

Other cost function, including some normalization, could also envisionned [see Pan2009].

> _note:_ mathematical notations are reduced as much as possible. For example $x$ is used for the coordinates of a point on a plane , i.e. $x = \vec x = (\mathtt x \,, \mathtt y)$

## Mathematical analysis

We have two signals or images: $A(x)$ and $B(x)$. We want to find the displacement $\delta$ such as $A(x+\delta)\approx B(x)$

> _note:_ first, only translation is considered. More general transformation, affine transformation $x + Mx$ for instance could be classified as higher order method (1st order vs 0th)

Using the least-square approach leads to formulating the problem as follows:

$$
\underset{\delta}{\text{argmin}} \int_x \left[ A(x+\delta) - B(x) \right]^2 \, dx
$$

By expanding the product, another formulation is obtained, involving the convolution product:

$$
\underset{\delta}{\text{argmax}} \int_x A(x+\delta) \cdot B(x) \,  dx
$$

> _note 1 :_ convolution is the same as cross-correlation plus or minus a sign change. Complex conjugate $B^\star$ appears due to the sign change.

> _note 2 :_  $\int A^2 \,dx$ and $\int B^2 \,dx$ are constant relative to the translation parameter $\delta$ for a given integration domain (ROI). There should be no need to normalize the images ROI  before (see [Pan2009] review).

The key benefit of the correlation formulation is the possible use of the Fourier transform and the Fast Fourier Transform algorithm for the computation. 


$$
\operatorname{\mathscr F} \left\lbrace A \otimes B \right\rbrace =
\operatorname{\mathscr F}\left\lbrace A\right\rbrace  \cdot   \operatorname{\mathscr F}\left\lbrace B\right\rbrace
$$

where $\otimes$ is the convolution product and $\operatorname{\mathscr F}\left\lbrace \bullet \right\rbrace$ is the Fourier Transform. It is in the following noted using calligraphic letter $\operatorname{\mathscr F}\left\lbrace A \right\rbrace = \mathcal A$. The complex conjugate is noted with a star $\mathcal A^\star$.

Then the registration problem becomes:

$$
\underset{\delta}{\text{argmax}} \; \operatorname{\mathscr F^{-1}}\left\lbrace \, \mathcal A \cdot \mathcal B^\star \, \right\rbrace
$$

At this point, it is interresting to note that the actual information about a signal shift (translation) is embodied in the phase of the Fourier Transform (complex number). (see the translation property of the FT)

$$
\mathscr{F} \lbrace A(x+\delta) \rbrace(k) =  \mathcal{A} (k) \, e^{-2\pi i \, k \cdot \delta}
$$

Thus the [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) approach:

$$
\underset{\delta}{\text{argmax}} \; \operatorname{\mathscr F^{-1}} \left\lbrace \,
\frac{\mathcal A \cdot \mathcal{B^\star}}{ |\mathcal A \cdot \mathcal B^\star| } \,
\right\rbrace
$$

> _question :_  What are the differences and advantages between the phase correlation approach (working with the phase only) and the "cross-correlation" approach (working with the absolute value)

## Sub-pixel accuracy

Computing the discrete convolution product, using either FFT or simple summation, leads to a 1-pixel sampled result. However, sub-pixel accuracy is possible. 

> It is similar to fitting a peak curve to find the central position at a resolution smaller than the sampling. Or averaging a serie of integer values to get a float. 

Now there is a different problem, which is the get the **sub-pixel** estimation of the position of the maximum of a peak (1D or 2D).

Continuous desciption of the peak using interpolation and smoothing, and then either up-sampling or use iterative optimization methods to locate the sub-pixel argmax.

Possible sub-pixel accuracy methods are: 

- local polynomial fit of the peak
  - using linear least-square method
  - using semi-analytcal approximation [Foroosh2002]
- an interresting approach is to use the Fourier transform itself for interpolation (upsampling using zero padding) [Guizar-Sicairos2008]
- Non-linear optimization method (gradient descent) on the interpolated peak function (see also [Guizar-Sicairos2008]). Initial guess using argmax on the FFT-based cross-correlation without upsampling.
- Phase unwrapping approach, see for instance [Balci2006]. _(My though: it is a non-linear problem and it involves computation of inverse Fourier transform or another... better to use non-linear optimisation directly)._ See also [Stone2001].
- Centroïd computation (weighted average): possible bias when off-center sampling

> It is similar to the peak fitting problem. However, here no known mathematical function of the peak shape is assumed and only position of the maximum is searched for.  


 
> _question :_ Why the method proposed by Guizar-Sicairos et al. (i.e. brute force optimization) is preferred over gradient descent optimization? 


## Phase correlation and appropriate windowing

Phase correlation works better when a windowing function is applied before computation of the fourier transforms. The [Blackman window](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackman.html) is used. See [Stone2001].


## Maximum peak value and sum of square difference

To 


## References

- Stone, H.S., M.T. Orchard, Ee-Chien Chang, et S.A. Martucci. « A Fast Direct Fourier-Based Algorithm for Subpixel Registration of Images ». IEEE Transactions on Geoscience and Remote Sensing 39, nᵒ 10 (octobre 2001): 2235‑43. https://doi.org/10.1109/36.957286.
- Foroosh, H., J.B. Zerubia, et M. Berthod. « Extension of Phase Correlation to Subpixel Registration ». IEEE Transactions on Image Processing 11, nᵒ 3 (mars 2002): 188‑200. https://doi.org/10.1109/83.988953.
- Guizar-Sicairos, Manuel, Samuel T. Thurman, et James R. Fienup. « Efficient Subpixel Image Registration Algorithms ». Optics Letters 33, nᵒ 2 (15 janvier 2008): 156. https://doi.org/10.1364/OL.33.000156.
- Balci, Murat, et Hassan Foroosh. « Subpixel Registration Directly from the Phase Difference ». EURASIP Journal on Advances in Signal Processing 2006, nᵒ 1 (décembre 2006). https://doi.org/10.1155/ASP/2006/60796.



## Error estimation

- Kybic, J. « Bootstrap Resampling for Image Registration Uncertainty Estimation Without Ground Truth ». IEEE Transactions on Image Processing 19, nᵒ 1 (janvier 2010): 64‑73. https://doi.org/10.1109/TIP.2009.2030955.

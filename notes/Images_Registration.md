---
tags: DIC
---

# Images Registration

Here only the zero-order registration problem is discussed, i.e. only a translation offset is searched between the two images.

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

> _note:_ convolution is the same as cross-correlation plus or minus a sign change. Also, complex conjugate $B^\star$ appears due to the sign change.

> _note 2:_ because of the indefinite integrale notation, $\int A^2 \,dx$ and $\int B^2 \,dx$, could be forgotten (they are constant). However, if the integration domain is of a fixed size, the image ROI could be normalized before (see [Pan2009] review).

The key benefit of the correlation formulation is the possible use of the Fourier transform and the Fast Fourier Transform algorithm for the computation. 

> big O notation: n^2 vs nlog(n) ?

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
\mathscr F \lbrace A(x+\delta) \rbrace(k) =  \mathcal A(k) \, e^{-2\pi i \, k \cdot \delta}
$$

Thus the [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) approach. 

> _note:_ est-ce que ce n'est au final qu'une histoire de normalisation ? et gestion du bruit

Computing the discrete convolution product, using either FFT or simple summation, leads to a 1-pixel sampled result. However, sub-pixel accuracy is possible. 

> It is similar to fitting a peak curve to find the central position at a resolution smaller than the sampling. Or averaging a serie of integer values to get a float. 

Now there is a different problem, which is the get the **sub-pixel** estimation of the position of the maximum of a peak (1D or 2D).

Continuous desciption of the peak using interpolation and smoothing, and then either up-sampling or use iterative optimization methods to locate the sub-pixel argmax.

- local quadratic fit of the peak and semi-analytcal approximation
- an interresting approach is to use the Fourier transform itself for the interpolation (zero padding, sinc)

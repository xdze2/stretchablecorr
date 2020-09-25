# Error estimation

## How to verify

- artificial images
  - gausian noise
  - other type of noise, or deformation ?

- Triangulation on real image

- Real images with rigid body motion only



## Multi-image alignement
- https://twitter.com/docmilanfar/status/1305028507026644993
- A Practical Guide to Multi-image Alignment https://arxiv.org/abs/1802.03280
- [ Constrained, globally optimal, multi-frame motion estimation ](https://ieeexplore.ieee.org/document/1628814)
- Fundamental Limits in Multi-image Alignment https://arxiv.org/pdf/1602.01541.pdf

## Error estimators

- Integral breadth: 
  - `(sum(cc) - N*min(CC))/(max - min)`
- z-score (same as IB ?)
- based on curvature at max.


1. RMS (relative error). method implemented in skimage  
--> corresponds to peak amplitude: $max(A \otimes B)$

2.   Cramér–Rao Bound & Fisher information matrix (??)  
--> corresponds to peak curvature 

*rationale:* FFT based cross-correlation gives function values for whole parameter space, at least at a 1-pixel resolution. How to use this available data... ?

- https://en.wikipedia.org/wiki/Standard_score
- 2nd moment approach: estimation of the width of the peak (don't work, because intertia is given by outer shell)

## log likelihood maths

$$
\underset{\delta}{\text{argmin}} \int_x \left[ A(x) - B(x+\delta) \right]^2 \, dx
$$


Let's consider the theoritical case where image $B$ is a pure translation of image $A$ plus a white centered noise:

$$
B = A(x + \delta) + \eta
$$

where $\eta$ is a white noise $\eta \sim \mathcal{N}(\sigma, \, 0)$


$$
P(B|\delta) = \prod_{pixels}\; \mathtt{N}( A(x)-B(x+\delta), \, \sigma ) 
$$

where $\mathtt{N}$ is the centered normal distribution:

$$
\mathtt{N}(u, \, \sigma) = \frac{1}{\sigma \sqrt{2\pi}}\exp(-\frac{u^2}{2\sigma^2}  )
$$

thus

$$
-\log P(B|\delta) = \sum_{pixels}\; \log(\sigma  \sqrt{2\pi}) + \frac{(A-B_\delta)^2}{2 \sigma^2}
$$

$$
-\log P(B|\delta) =  N\log(\sigma  \sqrt{2\pi}) + \frac{1}{2\sigma^2}\sum_{pixels}\; A^2 + B_\delta^2 - 2\,A\cdot B_\delta
$$

if $A$ and $B$ are normed:

$$
-\log P(B|\delta) =  N\log(\sigma  \sqrt{2\pi}) + \frac{1}{\sigma^2}\sum_{pixels}\; 1 - A\cdot B_\delta
$$

$$
-\log P(B|\delta) =  N\log(\sigma  \sqrt{2\pi}) + \frac{N}{\sigma^2} - \frac{1}{\sigma^2}\sum_{pixels}\, A\cdot B_\delta
$$

---

$$
log\,P(B|\delta) = \frac{1}{\sigma^2} \int (A(x)-B(x+\delta))^2 \,dx
$$

$$
log\,P(B|\delta) = -\frac{2}{\sigma^2} \left[ 1-  \int A(x)\cdot B(x+\delta) \,dx  \right]
$$ 
if $A$ and $B$ are normed, i.e. $\int A^2 = 1$

$$
log\,P(B|\delta) = -\frac{2}{\sigma^2} \left[ 1-  A\otimes B  \right]
$$ 

> integration bounds are lost, thus the normalization... $\mathbb{1}\otimes \mathbb{1}$ is not constant but pyramidal... or images are periodic...

entropy:

$$
P \cdot log\,P(B|\delta) = \sum \frac{2}{\sigma^2} \left[ 1-  A\otimes B  \right] \cdot \exp(-\frac{2}{\sigma^2} \left[ 1-  A\otimes B  \right])
$$ 

## Craméer-Rao bounds

$$
\frac{\partial^2}{\partial \delta^2} log\,P
$$

## FRAE (Fast Registration Accuracy Evaluation)

- see Kybic2008 and Kybic2010
- uses local quadratic approximation, given by BFGS optimization routine (Hessian)


$$
C_\theta = \lambda \sigma_J H^{-1}
$$

- $\lambda=1.68$
- $\sigma_J^2 = Var[J] \approx$


- geometrical error $\epsilon = \sqrt{tr\, C_\theta}$

units:
- $\theta$ is in pixel (px)
- images intensity $A$, noted $I$
- cost function $J(\theta) = A \otimes B (\theta) \quad [I^2]$ _(dx=1)_
- $H \sim \partial^2 J / \partial^2 \theta \quad [I^2 \, px^{-2}]$
- $\sigma_J \quad [I^2]$
- $C_\theta \quad [px^2]$
- $\epsilon \rightarrow [px]$

seems not to work... sub-pixel ? input image

## Boostrap

- Monte-Carlo like approach
- statistics obtained from random subsets of the data (images)
- do not work using FFT (need continuous array)



## References

- Kybic, J. « Bootstrap Resampling for Image Registration Uncertainty Estimation Without Ground Truth ». IEEE Transactions on Image Processing 19, nᵒ 1 (janvier 2010): 64‑73. https://doi.org/10.1109/TIP.2009.2030955.

- Kybic, Jan. « Fast No Ground Truth Image Registration Accuracy Evaluation: Comparison of Bootstrap and Hessian Approaches ». In 2008 5th IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 792‑95. Paris, France: IEEE, 2008. https://doi.org/10.1109/ISBI.2008.4541115.

- I. S. Yetik and A. Nehorai, “Performance bounds on image registration,” IEEE Transactions on Signal Processing, vol. 54, no. 5, pp. 1737–1749, May 2006, doi: 10.1109/TSP.2006.870552.

- D. Robinson and P. Milanfar, “Fundamental Performance Limits in Image Registration,” IEEE Transactions on Image Processing, vol. 13, no. 9, pp. 1185–1199, Sep. 2004, doi: 10.1109/TIP.2004.832923.



## How to evaluate

- using artificial data
  - problem: artificial noise, for instance light and surface change, is difficult to model

- using real data
  - problem: no ground truth
  - workaround:
    - triangulate (AB + BC - AC)
    - bootstrap... but how with FFT
    - using mutliple image: get only camera noise




## Auto-correlation of residuals

- https://stats.stackexchange.com/q/55658
- [How to test the autocorrelation of the residuals?](https://stats.stackexchange.com/q/14914)  

- [Residual Analysis with Autocorrelation - mathworks](https://www.mathworks.com/help/signal/ug/residual-analysis-with-autocorrelation.html)

Statistical tests:
- [Box-Jenkins framework](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method)
- Ljung–Box test
- Breusch-Godfrey test
- [Durbin-Watson test](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic)

but that's for time series, auto-regressive approachs  

-> What about Power Spectral Density (PSD)?

- [How to determine cut-off frequency using power spectral density?](https://dsp.stackexchange.com/q/19672)  

- https://en.wikipedia.org/wiki/Coherence_(signal_processing)



## Variance-Covariance matrix

book : STRUTZ, Tilo. Data fitting and uncertainty. A practical introduction to weighted least squares and beyond. Vieweg+ Teubner, 2010.



##  Multi-image Alignment

- A Practical Guide to Multi-image Alignment https://arxiv.org/abs/1802.03280
- CONSTRAINED, GLOBALLY OPTIMAL, MULTI-FRAME MOTION ESTIMATION  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.124.2674&rep=rep1&type=pdf

https://image-registration.readthedocs.io/en/latest/_modules/image_registration/chi2_shifts.html#chi2_shift

```python
def chi2_shift(im1, im2, err=None, upsample_factor='auto', boundary='wrap',
    nthreads=1, use_numpy_fft=False, zeromean=False, nfitted=2,
    verbose=False, return_error=True, return_chi2array=False,
    max_auto_size=512, max_nsig=1.1):
```

Find the offsets between image 1 and image 2 using the DFT upsampling method
 (http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html)
 combined with $\chi^2$ to measure the errors on the fit

 Equation 1 gives the $\chi^2$  value as a function of shift, where Y
 is the model as a function of shift:

$$
\chi^2(dx,dy) = \Sigma_{ij} \frac{(X_{ij}-Y_{ij}(dx,dy))^2}{\sigma_{ij}^2}  \\       
          =  \Sigma_{ij} \left[ X_{ij}^2/\sigma_{ij}^2 - 2X_{ij}Y_{ij}(dx,dy)/\sigma_{ij}^2 + Y_{ij}(dx,dy)^2/\sigma_{ij}^2 \right]  
$$

 Equation 2-4: blahha

$$
Term~1: f(dx,dy) = \Sigma_{ij} \frac{X_{ij}^2}{\sigma_{ij}^2}\\
f(dx,dy) =  f(0,0) ,  forall \, dx,dy
$$

$$
Term~2:  g(dx,dy) = -2 \Sigma_{ij} \frac{X_{ij}Y_{ij}(dx,dy)}{\sigma_{ij}^2} = -2 \Sigma_{ij} \left(\frac{X_{ij}}{\sigma_{ij}^2}\right) Y_{ij}(dx,dy)
$$

$$
Term~3:  h(dx,dy)  = \Sigma_{ij} \frac{Y_{ij}(dx,dy)^2}{\sigma_{ij}^2} = \Sigma_{ij} \left(\frac{1}{\sigma_{ij}^2}\right) Y^2_{ij}(dx,dy)
$$

The cross-correlation can be computed with fourier transforms, and is defined

$$
CC_{m,n}(x,y) = \Sigma_{ij} x^*_{ij} y_{(n+i)(m+j)}
$$

which can then be applied to our problem, noting that the cross-correlation has the same form as term 2 and 3 in $\chi^2$ (term 1 is a constant, with no dependence on the shift)

$$
Term~2:  CC(X/\sigma^2,Y)[dx,dy] = \Sigma_{ij} \left(\frac{X_{ij}}{\sigma_{ij}^2}\right)^* Y_{ij}(dx,dy)
$$

$$
Term~3:  CC(\sigma^{-2},Y^2)[dx,dy]  = \Sigma_{ij} \left(\frac{1}{\sigma_{ij}^2}\right)^* Y^2_{ij}(dx,dy)
$$

Technically, only terms 2 and 3 has any effect on the resulting image,
since term 1 is the same for all shifts, and the quantity of interest is
:math:`\Delta \chi^2` when determining the best-fit shift and error.

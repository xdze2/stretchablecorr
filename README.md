# stretchable corr

Digital Image Correlation in Python for mechanical test

- based on cross-correlation computation (FFT)
i.e. looking for the argmax of the inverse FT of the product of FT of images I anf J

in particular using the `register_translation` algo: ... uses cross-correlation in Fourier space, optionally employing an upsampled matrix-multiplication DFT to achieve arbitrary subpixel precision [1].


[1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, “Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156


- Error estimation using Likelihood approach --> 
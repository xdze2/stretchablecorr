# stretchable corr

Digital Image Correlation (DIC) in Python for mechanical test

- Displacemenent field between images is obtained using local cross-correlation computation (FFT)
i.e. looking for the argmax of the inverse FT of the product of FT of images I anf J

in particular using the function [`phase-cross-correlation`](https://scikit-image.org/docs/stable/api/skimage.registration.html#phase-cross-correlation) from skimage. Sub-pixel precision is obtained employing an upsampled matrix-multiplication DFT [1].

[1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, “Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156


- Load image from given directory, sort by alphabetical order
- construct a regulr grid (provide spacing margin )

- Error estimation using Likelihood approach --> 



- notes: xy is transposed relatively to ij

    array first dimension corresponds to y
    array second dimension corresponds to x
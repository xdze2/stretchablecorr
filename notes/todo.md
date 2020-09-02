## 2020-07-17

- online repo (gitlab)
- online static site + doc ?
    -> pdoc .. to md .. to html
    -> but maths...
    -> pandoc & by hand

- python package
- clean the existiing
- error estmation + smart stuff
- clean log output tool & params


## 2020-08-25

1. windowing & phase correlation **vs** cross-correlation
    pas évident, semble dépendre du cas test (artif, reel...)
2. opti vs upsampling DFT: speed, robustness

- high order method. Use DFT for interpolation.
- Locate the maximum using optiminsation (BFGS). Gives the Hessian matrix.
  - Cramér–Rao bound
  - Fast Registration Accuracy Estimation (FRAE) [Kybic2010]
- Co-variance matrix to ellipis confidance intervals

- Benchmark bench
  - Generated images: interpolation & rotation
  - Real pictures
  [x] pack all in a function 

- Entropy & locality error estimation ?
  - erreur dans l'estimation de la position du maximum. C'est l'erreur du bruit
  - erreur maximum non global
  - what about rotation ? (i.e. non parameterized)

## 2020-08-30

- noise is genrally not white noise:
  - sensor 8x8 pattern (lab bino)
  - change in light (hs2)

## 2020-02-09



# DIC

## Local vs Global

- **local** : to harvest FFT efficiency

    * "local" dans le sens où nécessite l'hypothèse que le **déplacement local est une translation pure**
        * en temps --> les images doivent être proches (...les rotations et def. faibles)
        * en espace --> la zone de contrôle doit être relativement petite

    -> sub-pixel range
   
 
 - **global** (FE like) --> large displacement (+100 pixels)
    _en fait c'est plus une distinction sur **l'ordre** de la méthode que global/local_

## Error estimation
- Error estimation based on cross-corr -bridge with likelihood-
    * density spectral de puissance... bruit blanc

    → 1. synthetic image generator (FFT inverse [ok] + sampling + noise)
        ou shift real image -- interpolation ?
    → 2. deal with sub-pixel stuff:
            a. without
    → 3. https://en.wikipedia.org/wiki/Phase_correlation

- Multi-images approach
    * ~ Kalman-Filter, or Baysian approach, ... weighted average

    * need to know the error for each estimation



- Trade-off robustness vs accuracy
    - the main error is not in the fine optimisation of the registration method
    - robustness --> correct error estimation and calibration

## Code workflow

1. image cube
2. points   (a regular grid or a set of unstructured points)
3. Compute displacement:
    - Eulerian (fixed grid): camera (or image) coordinates systeme 
    - Lagrangian (particule tracking): one particular image is used as reference, sample surface coordinates

    - image to images


## to look at & try 

- Fourier-Mellin domain: add rotation and scaling

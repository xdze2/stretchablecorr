# notes
## image name & structure

dir pattern:
/SAMPLE_NAME/STEP/IMAGE_NAME.TIF

STEP: 0p4 + u? + tag
IMAGE_NAME: SAMPLE_NAME + STEP + ID

## Working method
1. correlation method skimage
2. take one image

Problems
--> displacement larger than correlation area 
        displacement ~ def% * 1000px => 1%-->10px 
----> 1. increase the correlation area...  increases computation time
----> or succesive (correlation+fit+affine transform) or (correlation+correlation with offset)... :(


Output the results
- correl(I, J, grid)
  --> dx, dy, cor_error images
  --> 


## improvements

a. average images:  

1. Feature placed on a regular grid
2. Feature selection (Laplace, Corners, ...)
3. Dense optical flow
4. Weighted dense correlation 
---- weight:  entropy, auto-correlation, ?? , in-plane standart deviation... 
    ...how sharp the peak is... error estimation

4. FE based... i.e. fit by part 
- rotation


optical_flow_tvl1 --> super slow
https://scikit-image.org/docs/dev/api/skimage.registration.html#optical-flow-tvl1


https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html
https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

## Notes about images
### HS2
- image acquisition error for HS2/0p9, HS2/1p0 --> wrong position, del
- loses correlation after 10p0



---> Downsample array in Python ?
https://stackoverflow.com/q/18666014/8069403

---> Astropy  chiu2 method for error estimation
https://image-registration.readthedocs.io/en/latest/
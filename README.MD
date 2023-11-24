Enlarge the image 2 times using bicubic interpolation.
Test was performed on an image with dimensions of 6112 x 11658 pixels weighing 2.9 MB.

sequential code on cpu:
  time taken ~ 17.19s

cuda implementation:
  time taken ~ 14.57ms

Iplementation in openmp is planned, as well as a combination of cuda and openmp.

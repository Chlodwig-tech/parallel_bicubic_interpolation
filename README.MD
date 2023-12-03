Enlarge the image 2 times using bicubic interpolation.
Test was performed on an image with dimensions of 6112 x 11658 pixels weighing 2.9 MB.
Used for testing:
  - gpu - NVIDIA GeForce RTX 4070 Ti
  - cpu - 13th Gen Intel(R) Core(TM) i7-13700K

sequential code on cpu:
  time taken ~ 17.19s

cuda implementation:
  time taken ~ 14.57ms

openmp implementation:
  time taken ~ 1.21s

Implemnentation with combination of cuda and openmp is planned.
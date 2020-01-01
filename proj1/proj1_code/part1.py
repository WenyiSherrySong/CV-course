#!/usr/bin/python3

import numpy as np


def create_Gaussian_kernel(cutoff_frequency):
  """
  Returns a 2D Gaussian kernel using the specified filter size standard
  deviation and cutoff frequency.

  The kernel should have:
  - shape (k, k) where k = cutoff_frequency * 4 + 1
  - mean = floor(k/2)
  - standard deviation = cutoff_frequency
  - values that sum to 1

  Args:
  - cutoff_frequency: an int controlling how much low frequency to leave in
    the image.
  Returns:
  - kernel: numpy nd-array of shape (k, k)

  HINT:
  - The 2D Gaussian kernel here can be calculated as the outer product of two
    vectors drawn from 1D Gaussian distributions.
  """

  ############################
  ### TODO: YOUR CODE HERE ###
  
  cutoff_frequency = int(cutoff_frequency)
  k = cutoff_frequency*4 + 1
  mean = np.floor(k/2)
  std_deviation = cutoff_frequency
  x1 = np.ones(k)

  for i in range(0,k):
    x1[i] = i

  x1 = np.sqrt(7/44)*(1/std_deviation)*np.exp((-1/(2*std_deviation*std_deviation))*(x1-mean)*(x1-mean))
  kernel = np.outer(x1,x1)
  alpha = np.sum(kernel)
  kernel = kernel/alpha
    
  ### END OF STUDENT CODE ####
  ############################

  return kernel

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of shape (m, n, c)
  - filter: numpy nd-array of shape (k, j)
  Returns
  - filtered_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using OpenCV or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  filter_size = filter.shape

  ##Create pad image
  pad_size = [int(np.floor(x/2)) for x in filter_size]
  filtered_image = np.empty(image.shape)

  pad_image = np.lib.pad(image, ((pad_size[0],),(pad_size[1],),(0,)), 'symmetric')

  if(image.shape[2] == 1): # grayscale image
    for i in range(0, filtered_image.shape[0]):
      for j in range(0, filtered_image.shape[1]):
        window = pad_image[i:i+filter_size[0], j:j+filter_size[1]]
        filtered_image[i][j] = np.sum(np.multiply(window, filter))

    return filtered_image

  else: # RGB Image
    filter = filter.reshape((filter_size[0],filter_size[1],1))
    for i in range(0, filtered_image.shape[0]):
      for j in range(0, filtered_image.shape[1]):
        window = pad_image[i:i+filter_size[0], j:j+filter_size[1],:]
        filtered_image[i:i+1, j:j+1, :]= np.sum(np.multiply(window, filter), axis=(0,1))
    return filtered_image
  

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and a low-pass filter and creates a hybrid image. Returns
  the low frequency content of image1, the high frequency content of image 2,
  and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - low_frequencies: numpy nd-array of shape (m, n, c)
  - high_frequencies: numpy nd-array of shape (m, n, c)
  - hybrid_image: numpy nd-array of shape (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values of the hybrid image are between
    0 and 1. This is known as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  assert filter.shape[0] <= image1.shape[0]
  assert filter.shape[1] <= image1.shape[1]
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  filter_size = filter.shape
  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)
  # low_frequencies = np.clip(low_frequencies,a_min = 0,a_max = 1)
  # high_frequencies = np.clip(high_frequencies,a_min = 0,a_max = 1)
  img = low_frequencies+high_frequencies
  hybrid_image = np.clip(img, 0,1)

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image

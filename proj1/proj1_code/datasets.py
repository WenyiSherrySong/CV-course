#!/usr/bin/python3

"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import numpy as np
import os
import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import re
import glob
from PIL import Image

from typing import List, Tuple


def make_dataset(path: str) -> Tuple[List[str],List[str]]:
  """
  Creates a dataset of paired images from a directory.

  The dataset should be partitioned into two sets: one contains images that
  will have the low pass filter applied, and the other contains images that
  will have the high pass filter applied.

  Args
  - path: string specifying the directory containing images
  Returns
  - images_a: list of strings specifying the paths to the images in set A,
      in lexicographically-sorted order
  - images_b: list of strings specifying the paths to the images in set B,
      in lexicographically-sorted order
  """

  ############################
  ### TODO: YOUR CODE HERE ###
  # pattern_a = r'[0-9]+a_[a-zA-Z0-9]+.bmp'
  # pattern_b = r'\d+b_[a-zA-Z]+.bmp'
  
  # images_a = re.findall(pattern_a,path)
  # images_b = re.findall(pattern_b,path)

  images_a=[]
  images_b=[]

  # for file in glob.glob("path/r'[0-9]+a_[a-zA-Z0-9]+.bmp'"):
  #   images_a = images_a.append(file)

  # for file in glob.glob("path/r'[0-9]+b_[a-zA-Z0-9]+.bmp'"):
  #   images_b = images_b.append(file)  

  for x in os.listdir(path):
    # if x == path.match(r'[0-9]+a_[a-zA-Z0-9]+.bmp'):
    if re.match(r'[0-9]+a_[a-zA-Z0-9]+.bmp', x):
      x = os.path.join(path, x)
      images_a.append(x)
    else:
      x = os.path.join(path, x)
      images_b.append(x)  
  images_b.sort()
  images_a.sort() 


  ### END OF STUDENT CODE ####
  ############################

  return images_a, images_b


def get_cutoff_frequencies(path: str) -> List[int]:
  """
  Gets the cutoff frequencies corresponding to each pair of images.

  The cutoff frequencies are the values you discovered from experimenting in
  part 1.

  Args
  - path: string specifying the path to the .txt file with cutoff frequency
    values
  Returns
  - cutoff_frequencies: numpy array of ints. The array should have the same
    length as the number of image pairs in the dataset
  """

  ############################
  ### TODO: YOUR CODE HERE ###

  File_object = open(path,"r")
  cutoff_frequency = File_object.readlines()
  cutoff_frequencies = []

  for x in cutoff_frequency:
     x.replace("\n","")
     x = int(x)
     cutoff_frequencies.append(x)


  ### END OF STUDENT CODE ####
  ############################

  return cutoff_frequencies


class HybridImageDataset(data.Dataset):
  """Hybrid images dataset."""

  def __init__(self, image_dir: str, cf_file: str) -> None:
    """
    HybridImageDataset class constructor.

    You must replace self.transform with the appropriate transform from
    torchvision.transforms that converts a PIL image to a torch Tensor. You can
    specify additional transforms (e.g. image resizing) if you want to, but
    it's not necessary for the images we provide you since each pair has the
    same dimensions.

    Args:
    - image_dir: string specifying the directory containing images
    - cf_file: string specifying the path to the .txt file with cutoff
      frequency values
    """
    images_a, images_b = make_dataset(image_dir)
    cutoff_frequencies = get_cutoff_frequencies(cf_file)

   
    ############################
    ### TODO: YOUR CODE HERE ###

    self.transform = transforms.ToTensor()

    ### END OF STUDENT CODE ####
    ############################

    self.images_a = images_a
    self.images_b = images_b
    self.cutoff_frequencies = cutoff_frequencies

  def __len__(self) -> int:
    """Returns number of pairs of images in dataset."""

    ############################
    ### TODO: YOUR CODE HERE ###
   
    return len(self.images_a)

    ### END OF STUDENT CODE ####
    ############################

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor,torch.Tensor,int]:
    """
    Returns the pair of images and corresponding cutoff frequency value at
    index `idx`.

    Since self.images_a and self.images_b contain paths to the images, you
    should read the images here and normalize the pixels to be between 0 and 1.
    Make sure you transpose the dimensions so that image_a and image_b are of
    shape (c, m, n) instead of the typical (m, n, c), and convert them to
    torch Tensors.

    Args
    - idx: int specifying the index at which data should be retrieved
    Returns
    - image_a: Tensor of shape (c, m, n)
    - image_b: Tensor of shape (c, m, n)
    - cutoff_frequency: int specifying the cutoff frequency corresponding to
      (image_a, image_b) pair

    HINTS:
    - You should use the PIL library to read images
    - You will use self.transform to convert the PIL image to a torch Tensor
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    image1 = Image.open(self.images_a[idx])
    image2 = Image.open(self.images_b[idx])
    image_a = self.transform(image1)
    image_b = self.transform(image2)
    # cutoff_frequency = get_cutoff_frequencies(cf_file) 
    cutoff_frequency = self.cutoff_frequencies[idx] 


    ### END OF STUDENT CODE ####
    ############################

    return image_a, image_b, cutoff_frequency

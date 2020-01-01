import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
    '''
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    '''

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    dataset = []
    scaler = StandardScaler()
    # train_f = os.listdir(dir_name + '/train')
    # test_f = os.listdir(dir_name + '/test')
    for x in os.listdir(dir_name):
        x = os.path.join(dir_name, x)
        for xa in os.listdir(x):
            xa = os.path.join(x, xa)
            for xyz in os.listdir(xa):
                xyz = os.path.join(xa, xyz)
                # print(xyz)
                i = Image.open(xyz)
                i = Image.Image.split(i)
                i = np.array(i[0])
                i = i/255
                i = np.reshape(i, (-1, 1))
                scaler.partial_fit(i)
                # dataset.append(np.array(i))
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    # print(mean, std)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std

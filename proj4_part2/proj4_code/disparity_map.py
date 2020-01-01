"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np

from typing import Callable, Tuple


def calculate_disparity_map(left_img: torch.Tensor,
                            right_img: torch.Tensor,
                            block_size: int,
                            sim_measure_function: Callable,
                            max_search_bound: int = 50) -> torch.Tensor:
    """
    Calculate the disparity value at each pixel by searching a small
    patch around a pixel from the left image in the right image

    Note:
    1.  It is important for this project to follow the convention of search
        input in left image and search target in right image
    2.  While searching for disparity value for a patch, it may happen that there
        are multiple disparity values with the minimum value of the similarity
        measure. In that case we need to pick the smallest disparity value.
        Please check the numpy's argmin and pytorch's argmin carefully.
        Example:
        -- diparity_val -- | -- similarity error --
        -- 0               | 5
        -- 1               | 4
        -- 2               | 7
        -- 3               | 4
        -- 4               | 12

        In this case we need the output to be 1 and not 3.
    3. The max_search_bound is defined from the patch center.

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                  C will be >= 1.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   block_size: the size of the block to be used for searching between
                    left and right image
    -   sim_measure_function: a function to measure similarity measure between
                              two tensors of the same shape; returns the error value
    -   max_search_bound: the maximum horizontal distance (in terms of pixels)
                          to use for searching
    Returns:
    -   disparity_map: The map of disparity values at each pixel.
                       Tensor of shape (H-2*(block_size//2),W-2*(block_size//2))
    """

    assert left_img.shape == right_img.shape
    disparity_map = torch.zeros(1)  # placeholder, this is not the actual size
    ############################################################################
    # Student code begin
    ############################################################################
    disparity_map_np = []
    # print("MSB", max_search_bound)
    # print("bs", block_size)
    # print("correct size: ",
    #       left_img.shape[1]-2*(block_size//2), left_img.shape[0]-2*(block_size//2))
    # print("img_size", left_img.shape)
    for i in range(0, left_img.shape[0]):  # H
        for j in range(0, left_img.shape[1]):  # W
            if ((block_size//2) <= i and i < left_img.shape[0] - (block_size//2)) and ((block_size//2) <= j and j < (left_img.shape[1] - (block_size//2))):
                disp_array = []
                # print("i,j: ", i, j)
                if max_search_bound != 0:
                    for k in range(0, max_search_bound):
                        if ((block_size//2) <= j-k):  # always going left
                            patch1 = left_img[i - (block_size//2):i+(block_size//2)+1,
                                              j - (block_size//2):j+(block_size//2)+1, :]
                            patch2 = right_img[i - (block_size//2):i+(
                                block_size//2)+1, j - k - (block_size//2):j-k+(block_size//2)+1, :]
                            disp_value = sim_measure_function(patch1, patch2)
                            # print('disp_value', disp_value)
                            disp_array.append(disp_value)
                            # print(disp_array)
                        else:
                            break
                else:  # max_search_bound is 0
                    patch1 = left_img[i - (block_size//2):i+(block_size//2)+1,
                                      j - (block_size//2):j+(block_size//2)+1, :]
                    patch2 = right_img[i - (block_size//2):i+(
                        block_size//2)+1, j - (block_size//2):j+(block_size//2)+1, :]
                    disp_value = sim_measure_function(patch1, patch2)
                    # print('disp_value', disp_value)
                    disp_array.append(disp_value)

                min_disparity = np.argmin(np.array(disp_array))
                # print("disparity_val", min_disparity)
                disparity_map_np.append(min_disparity)
                # print(disparity_map_np)
    # print(len(disparity_map_np))
    disparity_map = torch.FloatTensor(disparity_map_np)
    disparity_map = disparity_map.reshape(
        left_img.shape[0]-2*(block_size//2), left_img.shape[1]-2*(block_size//2))
    # print(disparity_map)

    ############################################################################
    # Student code end
    ############################################################################
    return disparity_map


def calculate_cost_volume(left_img: torch.Tensor,
                          right_img: torch.Tensor,
                          max_disparity: int,
                          sim_measure_function: Callable,
                          block_size: int = 9):
    """
    Calculate the cost volume. Each pixel will have D=max_disparity cost values
    associated with it. Basically for each pixel, we compute the cost of
    different disparities and put them all into a tensor.

    Note:
    1.  It is important for this project to follow the convention of search
        input in left image and search target in right image
    2.  If the shifted patch in the right image will go out of bounds, it is
        good to set the default cost for that pixel and disparity to be something
        high(we recommend 255), so that when we consider costs, valid disparities will have a lower
        cost.

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                  C will be 1 or 3.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   max_disparity:  represents the number of disparity values we will consider.
                    0 to max_disparity-1
    -   sim_measure_function: a function to measure similarity measure between
                    two tensors of the same shape; returns the error value
    -   block_size: the size of the block to be used for searching between
                    left and right image
    Returns:
    -   cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
                  dimensions, and D is max_disparity. cost_volume[x,y,d]
                  represents the similarity or cost between a patch around left[x,y]
                  and a patch shifted by disparity d in the right image.
    """
    # placeholder
    H = left_img.shape[0]
    W = right_img.shape[1]
    cost_volume = torch.zeros(H, W, max_disparity)
    ############################################################################
    # Student code begin
    ############################################################################
    # print("BS: ", block_size)
    # print(cost_volume.shape)
    for i in range(0, H):  # H
        for j in range(0, W):  # W
            if ((block_size//2) <= i and i < H - (block_size//2)) and ((block_size//2) <= j and j < (W - (block_size//2))):
                # non edge cases
                for k in range(0, max_disparity):
                    if ((block_size//2) <= j-k):  # always going left
                        patch1 = left_img[i - (block_size//2):i+(block_size//2)+1,
                                          j - (block_size//2):j+(block_size//2)+1, :]
                        patch2 = right_img[i - (block_size//2):i+(block_size//2)+1,
                                           j - k - (block_size//2):j-k+(block_size//2)+1, :]
                        # print(i, j, k)
                    # print("p1", patch1.shape)
                    # print("p2", patch2.shape)
                        cost_volume[i, j, k] = torch.from_numpy(
                            sim_measure_function(patch1, patch2))
                        # print(cost_volume[i, j, k])
            else:  # edge cases
                cost_volume[i, j, :] = 255
                # print(cost_volume[i, j, k])

    ############################################################################
    # Student code end
    ############################################################################
    return cost_volume

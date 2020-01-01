"""
This file constrains different similarity measures used to compare blocks 
between two images
"""
import torch


def ssd_similarity_measure(patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:
    """
    Tests if two patches are similar by the SSD distance measure.

    SSD measure is sum of squared difference of pixel values in two patches.
    It is a good measure when the system has Gaussian noise.

    Args:
    -   patch1: one of the patch to compare (tensor of any shape/dimensions)
    -   patch2: the other patch to compare (tensor of the same shape as patch1)
    Returns:
    -   ssd_value: a single ssd value of the patch
    """
    assert patch1.shape == patch2.shape
    ssd_value = 0  # placeholder
    ############################################################################
    # Student code begin
    ############################################################################
    ssd_value = torch.sum(((patch1 - patch2)**2))
    ssd_value = ssd_value.data.cpu().numpy()
    # print(ssd_value)
    # print('xyz', ssd_value.dtype())

    ############################################################################
    # Student code end
    ############################################################################
    return ssd_value


def sad_similarity_measure(patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:
    """
    Tests if two patches are similar by the SAD distance measure.

    SAD is the sum of absolute difference. In general, absolute differences
    are more robust to large noise/outliers than squared differences.
    Ref: https://en.wikipedia.org/wiki/Sum_of_absolute_differences

    Args:
    -   patch1: one of the patch to compare (tensor of any shape/dimensions)
    -   patch2: the other patch to compare (tensor of the same shape as patch1)
    Returns:
    -   sad_value: the scalar sad value of the patch
    """

    assert patch1.shape == patch2.shape
    sad_value = 0  # placeholder
    ############################################################################
    # Student code begin
    ############################################################################

    sad_value = torch.sum(torch.abs(patch1 - patch2))
    sad_value = sad_value.data.cpu().numpy()
    ############################################################################
    # Student code end
    ############################################################################
    return sad_value

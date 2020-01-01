from proj6_code.simple_net import SimpleNet
from proj6_unit_tests.test_models import *

import numpy as np
import torch
from PIL import Image


def test_simple_net():
    '''
    Tests the SimpleNet contains desired number of corresponding layers
    '''
    this_simple_net = SimpleNet()
    # print(extract_model_layers(this_simple_net))
    all_layers, output_dim, counter, *_ = extract_model_layers(this_simple_net)

    print(counter['Conv2d'] >= 2)
    print(counter['Linear'] >= 2)
    print(counter['ReLU'] >= 2)
    print(output_dim == 15)

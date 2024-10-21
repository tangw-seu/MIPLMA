#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
import torch
import scipy.io as io
import random
import argparse
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v): 
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed):
    torch.manual_seed(seed)                     # Current CPU
    torch.cuda.manual_seed(seed)                # Current GPU
    np.random.seed(seed)                        # Numpy module
    random.seed(seed)                           # Python random module
    torch.backends.cudnn.benchmark = False      # Close optimization
    torch.backends.cudnn.deterministic = True   # Close optimization
    torch.cuda.manual_seed_all(seed)            # All GPU (Optional)

import torch
import numpy as np
import torch.nn as nn
import os
from main import get_argparser



def get_one_hot(labels, number_class):

    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    size = list(labels.size())
    labels = labels.view(-1)
    ones = torch.sparse.torch.eye(number_class).to(device)
    #labels = labels.to(device, dtype=torch.long)

    ones = ones.index_select(0, labels)
    size.append(number_class)
    ones = ones.view(*size)
    ones = ones.permute(0, 3, 1, 2)

    return ones
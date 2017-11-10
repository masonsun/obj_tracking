import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime as dt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from sample_generator import *
from training.options import opts
from training.train_rl import set_optimizer
from module.actnet import ActNet

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


def run_actnet():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default=None, help='input seq')
    parser.add_argument('-j', '--json', default=None, help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    assert args.seq is not None or args.json is not None

    # generate sequence configuration

import os
import sys
import pickle
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import PIL

from gen_dataset_region import *

#from model import *
from options import *

#from matplotlib import pylab as plt
#import matplotlib.patches as patches
#%matplotlib inline


def load_data(data_path, img_home):
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    K = len(data)
    print("data length:", K)
    dataset = [None] * K

    for k, (seqname, seq) in enumerate(data.items()):
        img_list = seq['images']
        gt = seq['gt']
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = GenDatasetRegion(img_dir, img_list, gt, opts)
    return dataset

    #image = Image.open(img_list[0]).convert('RGB')

if __name__ == "__main__":

    img_home = '../dataset/'
    data_path = 'data/vot2013.pkl'
    dataset = load_data(data_path, img_home)
    Img, bbox = dataset[0].next_frame()

    print(dataset[0].index)
    print(bbox)

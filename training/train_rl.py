import os
import sys
import pickle
import datetime.datetime as dt

import torch
import torch.optim as optim
from torch.autograd import Variable

from options import opts
from module.actnet import ActNet
from module.simulator import Simulator

IMG_PATH = '../data/'
DATA_PATH = '../data/vot-otb.pkl'


def train_actnet():
    # open data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # data objects
    K = len(data)
    dataset = [None] * K
    for k, (seq_name, seq) in enumerate(data.items()):
        img_list = seq['images']
        ground_truth = seq['gt']
        img_dir = os.path.join(IMG_PATH, seq_name)
        dataset[k] = Dataset(img_dir, img_list, ground_truth, opts)

    # model
    model = ActNet(model_path=opts['vgg_model_path'], k=K,
                   epsilon=opts['epsilon'], epsilon_decay=opts['epsilon_decay'])

    if torch.cuda.is_available() and opts['gpu']:
        model = model.cuda()
    model.set_trainable_params(opts['trainable_layers'])

    # tracking simulation
    sim = Simulator()







    return model


if __name__ == '__main__':
    train_actnet()

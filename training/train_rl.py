import os
import pickle
import numpy as np
from datetime import datetime as dt

import torch
import torch.optim as optim
from torch.autograd import Variable

from gen_dataset_region import GenDatasetRegion
from options import opts
from module.actnet import ActNet
from module.metrics import BinaryLoss, Precision

SEQ_HOME = '../dataset'
SEQ_LIST_PATH = 'data/vot2013.txt'
OUTPUT_PATH = 'data/vot2013.pkl'


def load_data(data_path, img_home):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    K = len(data)
    print("Data length: {}".format(K))
    dataset = [None] * K

    for k, (seq_name, seq) in enumerate(data.items()):
        img_list, gt = seq['images'], seq['gt']
        img_dir = os.path.join(img_home, seq_name)
        dataset[k] = GenDatasetRegion(img_dir, img_list, gt, opts)
    return dataset


def set_optimizer(model, vary_lr=False,
                  lr_base=opts['lr'], lr_mult=opts['lr_multiplier'],
                  momentum=opts['momentum'], w_decay=opts['w_decay']):
    # same setting across whole network
    if not vary_lr:
        return optim.RMSprop(model.parameters(), lr=lr_base, momentum=momentum, weight_decay=w_decay)

    # vary learning rate for different components
    params = model.get_trainable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        # increase learning rate if need be
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    return optim.RMSprop(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)


def train_actnet():
    # data
    dataset = load_data(OUTPUT_PATH, SEQ_HOME)

    # model
    model = ActNet(model_path=opts['vgg_model_path'])

    if torch.cuda.is_available() and opts['gpu']:
        model = model.cuda()
    model.set_trainable_params(opts['trainable_layers'])

    # model evaluation
    criterion = BinaryLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, vary_lr=False)

    # training loop
    best_prec = 0.0
    for i in range(int(opts['n_cycles'])):
        print("===== Cycle {} =====".format(i+1))

        k_list = np.random.permutation(len(dataset))
        prec = np.zeros(len(dataset))

        for j, k in enumerate(k_list):
            start_time = dt.now()
            img, bbox = dataset[k].next_frame()

            img, bbox = Variable(img), Variable(bbox)
            if torch.cuda.is_available() and opts['gpu']:
                img, bbox = img.cuda(), bbox.cuda()

            final_patch, past_bboxes = model(torch.unsqueeze(img, 0))

            # TO-DO:
            # 1. Use final_patch to calculate reward based on IoU with ground-truth bounding box
            # 2. Define a loss function to back-prop (BinaryLoss is not correct)

            prec[k] = evaluator()  # modify
            loss = criterion()     # modify

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            print("Cycle: {}, K: {} ({}), Loss: {:.3f}, Prec: {:.3f}, Time: {:.3f}"
                  .format(i, j, k, loss.data[0], prec[k], (dt.now() - start_time).total_seconds()))

        curr_prec = prec.mean()
        print("Mean precision: {:.3f}".format(curr_prec))
        if curr_prec > best_prec:
            best_prec = curr_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {
                'vggm_layers': model.vggm_layers.state_dict(),
                'actor_layers': model.actor_layers.state_dict(),
                'critic_layers': model.critic_layers.state_dict(),
            }
            print("Saved model to {}".format(opts['model_path']))
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == '__main__':
    train_actnet()

import os
import numpy as np
import scipy.io as sio
from collections import OrderedDict

import torch
import torch.nn as nn
from module.layers import LRN
from training.options import opts


class ActNet(nn.Module):
    def __init__(self, model_path=None):
        super(ActNet, self).__init__()

        # Parameters
        self.img_size = opts['img_size']
        self.num_actions = opts['num_actions']

        # VGG-M conv
        self.vggm_layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU()))]))

        # Actor-Critic
        self.lstm = nn.LSTMCell(512 * 3 * 3, 512)
        self.actor = nn.Linear(512, self.num_actions)
        self.critic = nn.Linear(512, 1)

        # Set biases to zero
        self.actor.bias.data.fill_(0)
        self.critic.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # Load weights
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unknown model format: {}".format(model_path))

        # Build dictionary of all parameters
        self.params = OrderedDict()
        self.build_params_dict()

    # Load weights for the whole network
    def load_model(self, model_path):
        states = torch.load(model_path)
        self.vggm_layers.load_state_dict(states['vggm_layers'])
        self.lstm.load_state_dict(states['lstm'])
        self.actor.load_state_dict(states['actor'])
        self.critic.load_state_dict(states['critic'])

    # Mainly to load VGG-M's pre-trained conv weights
    def load_mat_model(self, mat_file):
        mat = sio.loadmat(mat_file)
        mat_layers = list(mat['layers'])[0]
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.vggm_layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.vggm_layers[i][0].bias.data = torch.from_numpy(bias[:, 0])

    # Build a dictionary of all parameters in the network
    def build_params_dict(self):
        all_layers = [self.vggm_layers, self.actor, self.critic]
        for layers in all_layers:
            for name, submodule in layers.named_children():
                for child in submodule.children():
                    for i, param in child._parameters.items():
                        if param is None:
                            continue
                        if isinstance(child, nn.BatchNorm2d):
                            name = "{}_bn_{}".format(name, i)
                        else:
                            name = "{}_{}".format(name, i)

                        if name not in self.params:
                            self.params[name] = param
                        else:
                            raise RuntimeError("Duplicate param name: {}".format(name))

    # Set specified params to be trainable
    def set_trainable_params(self, layers):
        for i, param in self.params.items():
            if any(i.startswith(l) for l in layers):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Returns a dictionary of all trainable params
    def get_trainable_params(self):
        p = OrderedDict()
        for i, param in self.params.items():
            if param.requires_grad:
                p[i] = param
        return p

    # Forward pass of ActNet
    def forward(self, x):
        assert len(x) == 2 and len(x[1]) == 2, 'wrong number of inputs'
        # inputs
        x, (hx, cx) = x
        # vgg-m
        for name, submodule in self.vggm_layers.named_children():
            x = submodule(x)
            if name == 'conv3':
                x = x.view(-1, 512 * 3 * 3)
        # actor-critic
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic(x), self.actor(x), (hx, cx)

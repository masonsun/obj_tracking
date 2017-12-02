import os
import numpy as np
import scipy.io as sio
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

#from layers import LRN
#from options import opts

from torch import autograd

class ActNet(nn.Module):
    def __init__(self, opts, model_path=None):
        super(ActNet, self).__init__()

        # Parameters
        self.img_size = opts['img_size']
        self.num_actions = opts['num_actions']
        ff = 48
        sf = 64
        tf = 80
        self.tf = tf

        ## Conv Layers
        self.conv1 = nn.Conv2d(3, ff, kernel_size=3, stride=1, padding=0)
        self.conv1_act = nn.LeakyReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(ff, ff, kernel_size=3, stride=2)
        self.conv2_act = nn.LeakyReLU()
        #self.conv2_act_bn = nn.BatchNorm2d(ff)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(ff, sf, kernel_size=3, stride=1, padding=0)
        self.conv3_act = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(sf, sf, kernel_size=3, stride=2)
        self.conv4_act = nn.LeakyReLU()
        #self.conv4_act_bn = nn.BatchNorm2d(sf)
        self.conv5 = nn.Conv2d(sf, tf, kernel_size=3, stride=1, padding=0)
        self.conv5_act = nn.LeakyReLU()
        #self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Actor-Critic
        #self.lstm = nn.LSTMCell(96, 256)
        self.lstm = nn.LSTMCell(tf * 2 * 2, 384)
        self.lstm_act = nn.LeakyReLU()
        #self.linear = nn.Linear(384, 192)
        #self.linear_act = nn.LeakyReLU()
        #self.lstm3 = nn.Linear(512, 256)
        #self.lstm3_act = nn.LeakyReLU()
        self.actor = nn.Linear(384, self.num_actions)
        self.actor_act = nn.LeakyReLU()
        self.critic = nn.Linear(384, 1)
        self.critic_act = nn.LeakyReLU()
        #self.actor.bias.data.fill_(10)
        #self.critic.bias.data.fill_(10)

        self.hidden = (None, None)

        # Load weights


    def init_hidden(self, num_examples, cuda = False):
                # Before we've done anything, we dont have any hidden state.
                # Refer to the Pytorch documentation to see exactly
                # why they have this dimensionality.
                # The axes semantics are (num_layers, minibatch_size, hidden_dim)
                # 20 = batch_pos + batch_neg in gen_batch class
                hx, cx = (autograd.Variable(torch.zeros(num_examples, 384), requires_grad=True), autograd.Variable(torch.zeros(num_examples, 384), requires_grad=True))
                if cuda:
                    hx, cx = hx.cuda(), cx.cuda()
                self.hidden = (hx, cx)

    def set_hidden(self, hidden):
        hx, cx = hidden
        self.hidden = (autograd.Variable(hx.data, requires_grad=True), autograd.Variable(cx.data, requires_grad=True))

    # Forward pass of ActNet
    def forward(self, x):

        #assert len(x) == 2 and len(x[1]) == 2, 'wrong number of inputs'
        # inputs
        #x, hx, cx = x

        # Convolution Layers
        x = self.conv1(x)
        x = self.conv1_act(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.conv2_act(x)
        #x = self.conv2_act_bn(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.conv3_act(x)
        x = self.conv4(x)
        x = self.conv4_act(x)
        #x = self.conv4_act_bn(x)
        x = self.conv5(x)
        x = self.conv5_act(x)

        # actor-critic
        x = x.view(-1, self.tf * 2 * 2)
        hx, cx = self.lstm(x, (self.hidden))
        x = hx
        x = self.lstm_act(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return self.critic_act(critic), self.actor_act(actor), (hx, cx)


class ActNetClassifier(nn.Module):
    def __init__(self, actnet, num_actions):
        super(ActNetClassifier, self).__init__()
        self.actnet = actnet
        self.classifier = nn.Linear(num_actions+1, 2)

    def forward(self, x):
        c, a, (hx, cx) = self.actnet(x)
        x = self.classifier(torch.cat((c, a), 1))
        return F.softmax(x), (hx, cx)



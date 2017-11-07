import os
import numpy as np
import scipy.io as sio
from collections import OrderedDict

import torch
import torch.nn as nn
from layers import LRN
from utils import fifo_update


class ActNet(nn.Module):
    def __init__(self, model_path=None, k=10, num_actions=11):
        super(ActNet, self).__init__()

        # VGG-M conv layers
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

        # Actor
        self.actor_layers = nn.Sequential(OrderedDict([
            ('fc4_a', nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(512 * 3 * 3, 512),
                                    nn.ReLU())),
            ('fc5_a', nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(512, 512),
                                    nn.ReLU())),
            ('fc6_a', nn.Sequential(nn.Linear(512, num_actions),
                                    nn.Softmax()))]))

        # Critic
        self.critic_layers = nn.Sequential(OrderedDict([
            ('fc4_c', nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(512 * 3 * 3, 512),
                                    nn.ReLU())),
            ('fc5_c', nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(512, 512),
                                    nn.ReLU())),
            ('fc6_c', nn.Sequential(nn.Linear(512, num_actions),
                                    nn.Softmax()))]))

        # Past actions and q-values
        self.past_actions = torch.zeros(k, num_actions)
        self.past_qvalues = torch.zeros(k, num_actions)

        # Other parameters
        self.k = k
        self.num_actions = num_actions
        self.epsilon = 0.1
        self.epsilon_decay_rate = 0.1

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
        vggm, actor, critic = states['vggm_layers'], states['actor_layers'], states['critic_layers']
        self.vggm_layers.load_state_dict(vggm)
        self.actor_layers.load_state_dict(actor)
        self.critic_layers.load_state_dict(critic)

    # Mainly to load VGG-M's pre-trained conv weights
    def load_mat_model(self, mat_file):
        mat = sio.loadmat(mat_file)
        mat_layers = list(mat['layers'])[0]
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.vggm_layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.vgg_mlayers[i][0].bias.data = torch.from_numpy(bias[:, 0])

    # Build a dictionary of all parameters in the network
    def build_params_dict(self):
        all_layers = [self.vggm_layers, self.actor_layers, self.critic_layers]
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

    # Select one-hot encoded action based on epsilon-greedy strategy
    def epsilon_greedy(self, action):
        # assign probabilities to each action
        explore_prob = self.epsilon / self.num_actions
        p = np.full(self.num_actions, explore_prob)
        p[np.argmax(action.numpy())] = 1 - self.epsilon + explore_prob

        # one-hot encoding of selected action
        one_hot_action = torch.zeros(self.num_actions)
        index = np.random.choice(np.arange(self.num_actions), p=p)
        one_hot_action[index] = 1

        # decay epsilon
        self.epsilon *= self.epsilon_decay_rate

        return one_hot_action

    # Forward pass of ActNet
    def forward(self, x):
        # vggm
        for name, submodule in self.vggm_layers.named_children():
            x = submodule(x)
            # flatten
            if name == 'conv3':
                x = x.view(x.size(0), -1)

        # critic
        c = x.clone()
        for name, submodule in self.critic_layers.named_children():
            c = submodule(c)
            # concatenate with past actions
            if name == 'fc5_c':
                c = torch.cat((c, self.past_actions), 1)

        # update past q-values
        self.past_qvalues = fifo_update(self.past_qvalues, c)

        # actor
        a = x.clone()
        for name, submodule in self.actor_layers.named_children():
            a = submodule(a)
            # concatenate with past q-values
            if name == 'fc5_a':
                a = torch.cat((a, self.past_qvalues), 1)

        # select one-hot action via epsilon-greedy
        a = self.epsilon_greedy(a)

        # update past actions
        self.past_actions = fifo_update(self.past_actions, a)

        return a
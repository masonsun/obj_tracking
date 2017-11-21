import os
print(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
sys.path.insert(0,'/Users/RobinYen/Documents/DQN/actnet/obj_tracking/module')

from options import opts, kwargs
from load_data import load_data
from utils import crop_image
from actnet import ActNet


# File paths
SEQ_HOME = './dataset'
SEQ_LIST_PATH = '/Users/RobinYen/Documents/DQN/actnet/obj_tracking/training/data/vot2013.txt'
OUTPUT_PATH = '/Users/RobinYen/Documents/DQN/actnet/obj_tracking/training/data/vot2013.pkl'

# Frequency of prints
LOG_INTERVAL = 50

# Make stuff reproducible
np.random.seed(123)
torch.manual_seed(456)
if opts['gpu']:
    torch.cuda.manual_seed(789)


class ActNetClassifier(nn.Module):
    def __init__(self, actnet):
        super(ActNetClassifier, self).__init__()
        self.actnet = actnet
        self.classifier = nn.Linear(opts['num_actions'] + 1, 2)

    def forward(self, x):
        c, a, (hx, cx) = self.actnet(x)
        x = self.classifier(torch.cat((c, a), 1))
        return F.softmax(x), (hx, cx)


def apply_dataloader(data, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=opts['batch_size'],
        shuffle=shuffle,
        **kwargs)


def preprocess(data):
    x = torch.Tensor(len(data), 3, opts['img_size'], opts['img_size'])
    y = torch.Tensor(len(data), 1)
    for i in range(len(data)):
        img, bbox, ind = data[i]
        patch = crop_image(img, bbox)
        x[i], y[i] = patch, ind
    return x, y


def train_sl(train_loader, model, optimizer, steps_per_epoch):
    model.train()
    cx, hx = None, None
    for batch_index in range(steps_per_epoch):

        pos_data, pos_labels, neg_data, neg_labels = train_loader.next_frame()

        data = torch.cat((pos_data, neg_data), axis=0)
        labels = torch.cat((pos_labels, neg_labels), axis=0)
        shuffled_idx = torch.randperm(len(data))
        data = data[shuffled_idx]
        labels = labes[shuffled_idx]

        if opts['gpu']:
            data, target = data.cuda(), labels.cuda()
            data, target = Variable(data), Variable(labels)

        
        optimizer.zero_grad()
        output, (hx, cx) = model(data)
        loss = nn.BCEWithLogitsLoss(output, labels)
        loss.backward()
        optimizer.step()

    return cx, hx
'''
        if batch_index % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                100 * batch_index / len(train_loader), loss.data[0]))
'''



def test_sl(test_loader, model):
    model.eval()
    test_loss, correct = 0, 0

    for data, target in test_loader:
        if opts['gpu']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, _ = model(data)
        # sums up batch loss
        test_loss += nn.BCEWithLogitsLoss(output, target, size_average=False).data[0]
        # prediction (returns index of max probability)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n=== Test set ====\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


if __name__ == '__main__':
    # data
    # disjoint list of tuples (img, bbox, {0,1})
    #train_data, test_data = load_data(OUTPUT_PATH, SEQ_HOME, 'SL')

    # pre-processing
    # turn into (patch, indicator) which corresponds to (data, target)
    # data shape => (length, channel, width, height)
    # target shape => (value)
    
    #train_x, train_y = preprocess(train_data)
    #test_x, test_y = preprocess(test_data)

    # TO-DO:
    # train_loader = ...
    # test_loader = ...

#----------------
    train_data = load_data(OUTPUT_PATH, SEQ_HOME, 'SL')
    #test_data = load_data(TEST_PATH, SEQ_HOME, 'SL')
    #train_loader = 
    pos_data, pos_labels, neg_data, neg_labels = train_data[0].next_frame()
    
#----------------
'''
    # model
    assert opts['vgg_model_path'].split('.')[-1] in ['mat', 'pth'], 'Use pre-trained weights.'
    model = ActNetClassifier(ActNet(model_path=opts['vgg_model_path']))

    if opts['gpu']:
        model = model.cuda()
    model.set_trainable_params(opts['trainable_layers'])

    # initializations
    cx = Variable(torch.zeros(1, 512))
    hx = Variable(torch.zeros(1, 512))

    # evaluation
    best_loss = np.inf
    optimizer = optim.RMSprop(model.parameters(), lr=opts['lr'],
                              momentum=opts['momentum'], weight_decay=opts['w_decay'])

    # epochs loop
    for epoch in range(1, opts['epochs'] + 1):

        cx, hx = train_sl(train_loader, model, optimizer)
        #curr_loss = test_sl(test_loader, model)

        if curr_loss < best_loss:
            best_loss = curr_loss

            states = {
                'vggm_layers': model.vggm_layers.state_dict(),
                'lstm': model.lstm.state_dict(),
                'actor': model.actor.state_dict(),
                'critic': model.critic.state_dict()
            }

            if opts['use_gpu']:
                model = model.cpu()

            print("Saved model to {}".format(opts['model_sl_path']))
            torch.save(states, opts['model_sl_path'])
            torch.save((cx, hx), opts['lstm_path'])

            if opts['use_gpu']:
                model = model.cuda()

''' 

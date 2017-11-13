import os
import pickle
import numpy as np
from collections import namedtuple
from datetime import datetime as dt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from gen_dataset_region import GenDatasetRegion
from options import opts
from module.utils import overlap_ratio, get_reward, get_bbox, epsilon_greedy, crop_image
from module.actnet import ActNet

SEQ_HOME = '../dataset'
SEQ_LIST_PATH = 'data/vot2013.txt'
OUTPUT_PATH = 'data/vot2013.pkl'

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


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


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train_actnet():
    # data
    dataset = load_data(OUTPUT_PATH, SEQ_HOME)
    transition = namedtuple('transition', ['state', 'action', 'log_prob', 'entropy',
                                           'reward', 'next_state', 'done'])

    # model
    model = ActNet(model_path=opts['vgg_model_path'])

    if torch.cuda.is_available() and opts['gpu']:
        model = model.cuda()
    model.set_trainable_params(opts['trainable_layers'])
    model.train()

    # evaluation
    best_loss = np.inf
    optimizer = optim.RMSprop(model.parameters(), lr=opts['lr'],
                              momentum=opts['momentum'], weight_decay=opts['w_decay'])

    # initializations
    cx = Variable(torch.zeros(1, 256))
    hx = Variable(torch.zeros(1, 256))

    # training loop
    k_list = np.random.permutation(len(dataset))
    for j, k in enumerate(k_list):

        data_length = len(dataset[k].img_list)
        losses = np.full(data_length, np.inf)

        for f in range(data_length):
            img, bbox, gt = dataset[k].next_frame()
            img, bbox, gt = Variable(img), Variable(bbox), Variable(gt)
            if torch.cuda.is_available() and opts['gpu']:
                img, bbox, gt = img.cuda(), bbox.cuda(), gt.cuda()

            state = Variable(crop_image(img, bbox))
            epsilon = opts['epsilon']

            episode = []
            values = []

            start_time = dt.now()
            for i in range(opts['max_actions']):
                value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
                prob, log_prob = F.softmax(logit), F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1, keepdim=True)

                action, index = epsilon_greedy(prob, epsilon)
                log_prob = log_prob.gather(-1, index)
                epsilon /= opts['epsilon_decay']

                # take a step
                bbox, done = get_bbox(action, bbox)
                next_state = Variable(crop_image(img, bbox))
                done = done or (i+1) >= opts['max_actions']
                reward = 0 if not done else get_reward(overlap_ratio(bbox, gt))

                # keep track of transitions
                values.append(value)
                episode.append(transition(state=state, action=action, log_prob=log_prob, entropy=entropy,
                                          reward=reward, next_state=next_state, done=done))

                if done:
                    break

            v = Variable(torch.zeros(1, 1))
            values.append(v)

            policy_loss = 0
            value_loss = 0

            gae = torch.zeros(1, 1)
            gamma = opts['gamma']
            tau = opts['tau']
            entropy_coeff = opts['entropy_coeff']

            for i in reversed(range(len(episode))):
                v = episode[i].reward + (gamma * v)
                adv = v - values[i]
                value_loss += 0.5 * adv.pow(2)

                # generalized advantage estimation (GAE)
                delta_t = episode[i].reward + (gamma * values[i + 1].data) - values[i].data
                gae = (gamma * tau * gae) + delta_t
                policy_loss -= (episode[i].log_prob * Variable(gae)) - (entropy_coeff * episode[i].entropy)

            loss = policy_loss + opts['value_loss_coeff'] * value_loss
            losses[f] = loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            # print progress
            print("Class {}/{}, Frame {}/{}, Policy loss: {:.3f}, Value loss: {:.3f},\
                   Time: {:.3f}".format(j+1, len(k_list),
                                        f+1, data_length,
                                        policy_loss, value_loss,
                                        (dt.now() - start_time).total_seconds()))

        if np.any(np.isinf(losses)):
            raise RuntimeError("Infinite loss detected")

        curr_loss = losses.mean()
        print("Mean loss: {:.3f}".format(curr_loss))
        if curr_loss < best_loss:
            best_loss = curr_loss

            states = {
                'vggm_layers': model.vggm_layers.state_dict(),
                'lstm': model.lstm.state_dict(),
                'actor': model.actor.state_dict(),
                'critic': model.critic.state_dict()
            }

            if opts['use_gpu'] and torch.cuda.is_available():
                model = model.cpu()

            print("Saved model to {}".format(opts['model_path']))
            torch.save(states, opts['model_path'])

            if opts['use_gpu'] and torch.cuda.is_available():
                model = model.cuda()


if __name__ == '__main__':
    train_actnet()

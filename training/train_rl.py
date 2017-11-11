import os
import pickle
import numpy as np
from datetime import datetime as dt
from collections import namedtuple

import torch
import torch.optim as optim
from torch.autograd import Variable

from gen_dataset_region import GenDatasetRegion
from options import opts
from module.utils import crop_image, apply_action_to_bbox
from module.actnet import ActNet
from module.metrics import overlap_ratio, get_reward

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


def train_actnet():
    # data
    dataset = load_data(OUTPUT_PATH, SEQ_HOME)
    transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'done'])

    # model
    model = ActNet(model_path=opts['vgg_model_path'])

    if torch.cuda.is_available() and opts['gpu']:
        model = model.cuda()
    model.set_trainable_params(opts['trainable_layers'])

    # model evaluation
    actor_loss =
    critic_loss =

    # training loop
    k_list = np.random.permutation(len(dataset))
    for j, k in enumerate(k_list):
        print("Class {}/{}".format(k+1, len(k_list)))

        for _ in range(len(dataset[k].img_list)):
            img, bbox, gt = dataset[k].next_frame()
            img, bbox, gt = Variable(img), Variable(bbox), Variable(gt)
            if torch.cuda.is_available() and opts['gpu']:
                img, bbox, gt = img.cuda(), bbox.cuda(), gt.cuda()

            for i in range(opts['max_actions']):
                episode = []

                # take a step
                patch = crop_image(img, bbox)
                action = model(patch, eval_policy=True)
                bbox, done = apply_action_to_bbox(action, bbox)
                new_patch = crop_image(img, bbox)
                reward = 0 if not done else get_reward(overlap_ratio(bbox, gt))

                # keep track of transitions
                episode.append(transition(
                    state=patch, action=action, reward=reward, next_state=new_patch, done=done))

                # calculate TD target
                value = model(new_patch, eval_policy=False)
                td_target = reward + opts['reward_discount'] * value
                td_error = td_target - model(patch, eval_policy=False)

                # update value estimator
                model.zero_grad()




                # update policy estimator
                # use the TD error as our advantage estimate


                if done:
                    break


    # model evaluation
    criterion = BinaryLoss()
    evaluator = Precision()
    optimizer = optim.RMSprop(model.parameters(), lr=opts['lr'], momentum=opts['momentum'], weight_decay=opts['w_decay'])

    prec = np.zeros(len(dataset))
    for i in range(int(opts['n_cycles'])):

        for j, k in enumerate(k_list):
            start_time = dt.now()

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

import numpy as np
from collections import namedtuple
from datetime import datetime as dt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from options import opts
from load_data import load_data
from utils import overlap_ratio, get_reward, get_bbox, epsilon_greedy, crop_image, view_image
from actnet import ActNet, ActNetClassifier, ActNetRL

SEQ_HOME = '../dataset'
SEQ_LIST_PATH = 'data/human.txt'
OUTPUT_PATH = 'data/human.pkl'

np.random.seed(123)
torch.manual_seed(456)
if opts['gpu']:
    torch.cuda.manual_seed(789)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train_rl():
    # data
    dataset = load_data(OUTPUT_PATH, SEQ_HOME, 'RL')
    transition = namedtuple('transition', ['state', 'action', 'log_prob', 'entropy',
                                           'reward', 'next_state', 'done'])

    # model
    assert opts['model_sl_path'].split('.')[-1] == 'pth', 'Use pre-trained weights.'
    model_classifier = ActNetClassifier(ActNet(opts=opts,model_path=None), opts['num_actions'])#model_path=opts['model_sl_path'])
    #model_classifier.load_state_dict(torch.load('../model/actnet_sl_weight.pth'))
    model = ActNetRL(model_classifier.actnet, opts['num_actions'])
    #model = model_classifier.actnet
    model = model.float()
 
#    for params in model.actnet.parameters():
#        params.requires_grad = False
    
    '''
    trainable_params = [
                        {"params":model.lstm.parameters()},
                        {"params":model.dense1.parameters()},
                        {"params":model.dense2.parameters()},
                        {"params":model.actor.parameters()},
                        {"params":model.critic.parameters()}
    ]
    '''

    if opts['gpu']:
        model = model.cuda()
    model.train()
    
    # evaluation
    best_loss = np.inf
    optimizer = optim.RMSprop(model.parameters(), lr=opts['lr'],
                              momentum=opts['momentum'], weight_decay=opts['w_decay'])

    # initializations
    try:
        cx, hx = torch.load(opts['lstm_path'])
        cx, hx = Variable(cx), Variable(hx)
    except IOError:
        print('LSTM state values not found. Initializing as zero valued tensors')
        cx = Variable(torch.zeros(1, 512), requires_grad=True)
        hx = Variable(torch.zeros(1, 512), requires_grad=True)

    if opts['gpu']:
        cx, hx = cx.cuda(), hx.cuda()

    #model.set_hidden((hx, cx))

    # training loop
    k_list = np.random.permutation(len(dataset))

    start_time = dt.now()
    while True:
        for j, k in enumerate(k_list):
            print(dataset[k].img_list[0])
            #if j != 0:
            #    continue
            data_length = len(dataset[k].img_list)
            #data_length = 10## debug
            losses = np.full(data_length, np.inf)
            model.init_hidden(1, opts['gpu'])

            for f in range(data_length): 
 
                #print(data_length)
                model.init_hidden(1, opts['gpu'])
                img_n, bbox_n, gt_n = dataset[k].next_frame()
                #exit()
                print("bbox:", bbox_n)
                #print("img shape:", img_n.shape[0])
                #img_n = img_n.transpose(2,0,1)
                img, bbox, gt = torch.from_numpy(img_n), torch.from_numpy(bbox_n), torch.from_numpy(gt_n)
                img, bbox, gt = Variable(img), Variable(bbox), Variable(gt)
                if opts['gpu']:
                    img, bbox, gt = img.cuda(), bbox.cuda(), gt.cuda()
                    #img_n, bbox_n = img_n.cuda(), bbox_n.cuda()

                state = crop_image(img_n, bbox_n)
                #view_image(state)
                #exit()
                state = state.transpose(2,0,1)

                state = Variable(torch.from_numpy(state))
                if opts['gpu']:
                    state = state.cuda()

                epsilon = opts['epsilon']

                episode = []
                values = []
                deep_copy_gt = gt_n.copy()
                #start_time = dt.now()
                action = torch.Tensor(opts['num_actions']).zero_()

                print("~~~~~~~~~~~~Start Training Frame~~~~~~~~~~~~~")
                action_history = []
                for i in range(opts['max_actions']):
                    #print("step {} in an episode:".format(i))
                    #print(state.unsqueeze(0))
                    #print((hx.float(), cx.float()))
                    action = Variable(action)
                    if opts['gpu']:
                        action = action.cuda()
                    value, logit = model(state.unsqueeze(0).float(), action)
                    #model.set_hidden((hx, cx))
                    prob, log_prob = F.softmax(logit), F.log_softmax(logit)
                    entropy = -(log_prob * prob).sum(1, keepdim=True)
                    
                    print("Prob:", prob.data[0].cpu())
                    #prob = np.asarray(prob)
                    #print("abc", isinstance(prob, torch.Tensor))
                    #print(prob.shape)
                    
                    if opts['gpu']:
                        prob = prob.cpu()
                    action, index = epsilon_greedy(prob, epsilon)
                    #print("Action index: ", index.numpy()[0,0])
                    action_history.append(index.numpy()[0,0])
                    if opts['gpu']:
                        log_prob = log_prob.gather(-1, Variable(index).long().cuda())
                    else:
                        log_prob = log_prob.gather(-1, Variable(index).long())
                    
                    epsilon *= opts['epsilon_decay']

                    # take a step
                    
                    bbox, done = get_bbox(action, bbox.cpu(), img_n.shape)
                    #print("gt:", gt_n)
                    #print("bbox:", bbox.data.cpu())
                    next_state = crop_image(img_n, bbox.data.cpu().numpy())
                    next_state = next_state.transpose(2,0,1)
                    next_state = Variable( torch.from_numpy(next_state) )
                    #next_state = Variable(crop_image(img_n, bbox))
                    done = done or (i+1) >= opts['max_actions']
                    reward = 0 if not done else get_reward(overlap_ratio(bbox.data.cpu().numpy(), gt.data.cpu().numpy()))
                    #reward = 1 if index.numpy() == 10 else -1
                    #print("overlap: {}, reward: {}".format(overlap_ratio(bbox.data.cpu().numpy(), gt.data.cpu().numpy()), reward))
                    #exit()
                    # keep track of transitions
                    values.append(value)
                    episode.append(transition(state=state, action=action, log_prob=log_prob, entropy=entropy,
                                            reward=reward, next_state=next_state, done=done))

                    if done:
                        break
                print("action history: ", action_history)
                print("overlap: {}, reward: {}".format(overlap_ratio(bbox.data.cpu().numpy(), gt.data.cpu().numpy()), reward))
                #if reward == 1:
                    #input('press enter~~~~')
                v = Variable(torch.zeros(1, 1))
                values.append(v)

                policy_loss = 0
                value_loss = 0

                gae = torch.zeros(1, 1)
                gamma = opts['gamma']
                tau = opts['tau']
                entropy_coeff = opts['entropy_coeff']

                for i in reversed(range(len(episode))):
                    #print("reversed i", i )
                    v = episode[i].reward + (gamma * v)
                    adv = v - values[i].cpu()
                    value_loss += 0.5 * adv.pow(2)
                    #print("value loss", value_loss.data.shape)
                    
                    # generalized advantage estimation (GAE)
                    delta_t = episode[i].reward + (gamma * values[i + 1].cpu().data) - values[i].cpu().data                    
                    gae = (gamma * tau * gae) + delta_t
                    if opts['gpu']:
                        policy_loss -= (episode[i].log_prob * Variable(gae).cuda())# - (entropy_coeff * episode[i].entropy)
                    else:
                        policy_loss -= (episode[i].log_prob * Variable(gae))# - (entropy_coeff * episode[i].entropy)
                    #print('adv: ', adv)
                    #input("press enter")
                    #print('log_prob', episode[i].log_pro)
                if opts['gpu']:
                    loss = policy_loss + opts['value_loss_coeff'] * value_loss.cuda()
                else:
                    loss = policy_loss + opts['value_loss_coeff'] * value_loss
                #print("LOSS:", loss.data[0])
                
                losses[f] = loss.data[0][0]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
                optimizer.step()

                # print progress

                print("Class {}/{}, Frame {}/{}, Policy loss: {:.3f}, Value loss: {:.3f},\
                    ".format(j+1, len(k_list),
                                            f+1, data_length,
                                            policy_loss.data[0][0], value_loss.data[0][0],
                                            ))
                
            if np.any(np.isinf(losses)):
                raise RuntimeError("Infinite loss detected")

            curr_loss = losses.mean()
            print("Mean loss: {:.3f}".format(curr_loss))
            #if curr_loss < best_loss:
            if True:
                best_loss = curr_loss



                if opts['gpu']:
                    model = model.cpu()

                print("Saved model to {}".format('../model/actnet-rl-v2.pth'))
                torch.save(model.state_dict(), '../model/actnet-rl-v2.pth')
                #torch.save(states, opts['model_rl_path'])

                if opts['gpu']:
                    model = model.cuda()
        #print("Mins: {:.3f}".format((dt.now()-start_time).total_seconds() / 60))

if __name__ == '__main__':
    train_rl()

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime as dt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import sys
sys.path.insert(0,'../training')
sys.path.insert(0,'../module')
from options import opts
#from train_rl import set_optimizer
from actnet import ActNet , ActNetClassifier, ActNetRL

from sample_generator import *

np.random.seed(123)
torch.manual_seed(456)
if opts["gpu"]:
    torch.cuda.manual_seed(789)

def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name

        seq_home = '../dataset/evaluation/vot2013'
        save_home = '../dataset/evaluation/vot2013/result_fig'
        result_home = '../dataset/evaluation/vot2013/result'
        
        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        #print(img_dir)
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir,x) for x in img_list]
        #print(img_list)
        gt = np.loadtxt(gt_path,delimiter=',')
        init_bbox = gt[0]
        #print(init_bbox)
        savefig_dir = os.path.join(save_home,seq_name)
        #print(savefig_dir)
        result_dir = os.path.join(result_home,seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir,'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json,'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None
        
    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path

def run_actnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list),4))
    result_bb = np.zeros((len(img_list),4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    #model = ActNet(opts=opts)
    #model_classifier = ActNetClassifier(ActNet(opts=opts,model_path=None), opts['num_actions'])#model_path=opts['model_sl_path'])
    #print(model_classifier.state_dict())
    #model_classifier.load_state_dict(torch.load('../model/actnet_sl_weight.pth'))
    #model.load_state_dict(torch.load('../model/actnet-rl-v2.pth'))
    #model = model_classifier.actnet
    #model = model.float()
    model_classifier = ActNetClassifier(ActNet(opts=opts,model_path=None), opts['num_actions'])#model_path=opts['model_sl_path'])
    model_classifier.load_state_dict(torch.load('../model/actnet_sl_weight.pth'))
    model = ActNetRL(model_classifier.actnet, opts['num_actions'])
    model.load_state_dict(torch.load('../model/actnet-rl-v2.pth'))
    if opts['gpu']:
        model = model.cuda()
    #model.set_learnable_params(opts['ft_layers'])
    model.train()
    try:
        cx, hx = torch.load(opts['lstm_path'])
        cx, hx = Variable(cx), Variable(hx)
    except IOError:
        print('LSTM state values not found. Initializing as zero valued tensors')
        cx = Variable(torch.zeros(1, 512))
        hx = Variable(torch.zeros(1, 512))
    model.init_hidden(1)

    # Display starting image
    image = Image.open(img_list[0]).convert('RGB')
    savefig = savefig_dir != ''
    if display or savefig: 
        dpi = 80.0
        figsize = (image.size[0]/dpi, image.size[1]/dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0,:2]),gt[0,2],gt[0,3], 
                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        
        rect = plt.Rectangle(tuple(result_bb[0,:2]),result_bb[0,2],result_bb[0,3], 
                linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(0.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir,'0000.jpg'),dpi=dpi)

    
    # testing
    
    bbox_history_all = []
    iou = np.array([])
    #for i in range(len(img_list)):
    for i in range(207):
    #tic = time.time()
    # Load image
        bbox_history = []
        image = Image.open(img_list[i]).convert('RGB')
        print(img_list[i])
        #continue
        image_n = image.copy()
        #exit()
        image_n = np.asarray(image_n)
        if i  in (0,10,30):
            bbox_n = gt[i].copy()
            bbox = torch.from_numpy(bbox_n)
            bbox = Variable(bbox)
            print("GT: box", bbox_n)
        #exit()
        #TODO : save predicted box in result_bb
        state = crop_image(image_n, bbox_n)
        state = state.transpose(2,0,1)
        #print("state: ", state)
        state = Variable(torch.from_numpy(state))
        
        for j in range(opts['max_actions']):
        
            #value, logit, (hx, cx) = model((state.unsqueeze(0).float(), (hx.float(), cx.float())))
            value, logit, (hx, cx) = model(state.unsqueeze(0).float())
            prob, log_prob = F.softmax(logit), F.log_softmax(logit)
            print("Prob:", prob)
            #exit()
            
            
            model.set_hidden((hx, cx))
            
            one_hot_action = torch.zeros(opts['num_actions'])
            action = np.argmax(prob.data.numpy())
            one_hot_action[action] = 1
            print(action)
            bbox, done = get_bbox(one_hot_action, bbox, image_n.shape)

            print("bbox:", bbox)
            bbox_n_copy = bbox_n.copy()
            bbox_history.append(bbox_n_copy)
            if done:
                break
            #exit()
        bbox_history_all.append(bbox_history)
        #print("bbox_his",bbox_history_all[i][0][:2])
        #model()
        #print("his",len(bbox_history_all))
        for k in range(len(bbox_history_all[i])):
        #for k in range(10):
            if display or savefig:
                im.set_data(image)

                if gt is not None:
                    gt_rect.set_xy(gt[i,:2])
                    gt_rect.set_width(gt[i,2])
                    gt_rect.set_height(gt[i,3])

                rect.set_xy(bbox_history_all[i][k][:2])
                rect.set_width(bbox_history_all[i][k][2])
                rect.set_height(bbox_history_all[i][k][3])
                
                if display:
                    plt.pause(.01)
                    plt.draw()
                if savefig:
                    fig.savefig(os.path.join(savefig_dir,'%03d_%d.jpg' %(i,k)),dpi=dpi)

        #print(type( overlap_ratio(bbox_history_all[i][k], gt[i,:])))
        #print(type(iou))
        iou = np.concatenate((iou, overlap_ratio(bbox_history_all[i][k], gt[i,:])))
        print(iou)
        avgiou = iou.mean()
        print("AVG IOU: ", avgiou)
        #exit()


    
    '''
    #for i in range(len(bbox_history_all)):
    for i in range(30):
        # Display
        image = Image.open(img_list[i]).convert('RGB')
        for k in range(len(bbox_history_all[i])):
        #for k in range(10):
            if display or savefig:
                im.set_data(image)

                if gt is not None:
                    gt_rect.set_xy(gt[i,:2])
                    gt_rect.set_width(gt[i,2])
                    gt_rect.set_height(gt[i,3])

                rect.set_xy(bbox_history_all[i][k][:2])
                rect.set_width(bbox_history_all[i][k][2])
                rect.set_height(bbox_history_all[i][k][3])
                
                if display:
                    plt.pause(.01)
                    plt.draw()
                if savefig:
                    fig.savefig(os.path.join(savefig_dir,'%04d.jpg'%(i)),dpi=dpi)
    '''


    #print(bbox_history_all)
    #print(gt)

    #for i in range(0,len(img_list)):
    #    print(i)



    #pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default=None, help='input seq')
    parser.add_argument('-j', '--json', default=None, help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    assert args.seq is not None or args.json is not None
    #
    # generate sequence configuration

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
    #print(gt)
    run_actnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)
    

import os
import numpy as np
import pickle

SEQ_HOME = '../dataset/'
SEQ_LIST_PATH = 'data/bicycle.txt'
OUTPUT_PATH = 'data/bicycle.pkl'

with open(SEQ_LIST_PATH, 'r') as fp:
    seq_list = fp.read().splitlines()

data = {}
for i, seq in enumerate(seq_list):
    img_list = sorted([p for p in os.listdir(SEQ_HOME + seq) if os.path.splitext(p)[1] == '.jpg'])
    gt = np.loadtxt(SEQ_HOME + seq + '/groundtruth.txt', delimiter=',')

    assert len(img_list) == len(gt), "Lengths do not match!"

    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    data[seq] = {'images': img_list, 'gt': gt}

with open(OUTPUT_PATH, 'wb') as fp:
    pickle.dump(data, fp, -1)

import os
import pickle
from gen_dataset_region import GenDatasetRegion
from options import opts
# TO-D0:
# 1. Loading data for SL training must return a train/test split


def load_data(data_path, img_home, arg=None):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    K = len(data)
    print("Data length: {}".format(K))
    dataset = [None] * K

    # TO-DO:
    # - GenDatasetRegion will be different for RL/SL/OL training.
    # - Use arg (str) to determine.
    print(arg)

    for k, (seq_name, seq) in enumerate(data.items()):
        img_list, gt = seq['images'], seq['gt']
        img_dir = os.path.join(img_home, seq_name)
        dataset[k] = GenDatasetRegion(img_dir, img_list, gt, opts, arg)

    return dataset


import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import sys
sys.path.insert(0,'module')
from sample_generator import *

from utils import crop_image, overlap_ratio


class GenDatasetRegion(data.Dataset):
    def __init__(self, img_dir, img_list, gt, opts, mode):

        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
        self.gt = gt
        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        #self.index = np.random.permutation(len(self.img_list))
        self.index = np.arange(len(self.img_list))
        self.pointer = 0

        self.image = Image.open(self.img_list[0]).convert('RGB')
        self.pos_generator = SampleGenerator('gaussian', self.image.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', self.image.size, 1, 1.2, 1.1, True)

        self.mode = mode.lower()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __iter__(self):
        return self

    
    # generate next frame's image and gt bbox
    def next_frame_sl(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0,3,self.crop_size,self.crop_size))
        neg_regions = np.empty((0,3,self.crop_size,self.crop_size))

        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)
            pos_examples = gen_samples(self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)),axis=0)

            n_neg = (self.batch_pos - len(neg_regions)) // (self.batch_frames - i)
            neg_examples = gen_samples(self.neg_generator, bbox, n_neg,
                                       overlap_range=self.overlap_neg)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)),axis=0)
        pos_labels = np.ones(len(pos_regions))
        neg_labels = np.zeros(len(neg_regions))
        pos_regions = torch.from_numpy(pos_regions).float()
        pos_labels = torch.from_numpy(pos_labels).float()
        neg_regions = torch.from_numpy(neg_regions).float()
        neg_labels = torch.from_numpy(neg_labels).float()
        return pos_regions, pos_labels, neg_regions, neg_labels
        #return image, bbox
        
    def next_frame_rl(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0,3,self.crop_size,self.crop_size))
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            bbox_idx = idx[0] - 8
            if bbox_idx<0:
                bbox_idx = 0
            #print("bbox_idx, IDX:", bbox_idx, idx)
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

            #n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)
            #pos_examples = gen_samples(self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
            #pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)),axis=0)
        gt_box = bbox.copy()
        bbox = self.gt[bbox_idx].copy()
        
        #print(gt_box, bbox)
        #pos_regions = torch.from_numpy(pos_regions).float()
        return image, bbox, gt_box

    def next_frame(self):
        if self.mode == 'rl':
            return self.next_frame_rl()
        elif self.mode == 'sl':
            return self.next_frame_sl()
        else:
            raise Exception("mode only supports 'rl' or 'sl', but get{}".format(self.mode))

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(samples):
            
            #regions[i] = self.transform(
            #    crop_image(image, sample, self.crop_size, self.padding, True))
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)
        regions = regions.transpose(0,3,1,2)
        regions = regions.astype('float32') - 128.
        return regions

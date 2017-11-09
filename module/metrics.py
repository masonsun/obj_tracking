import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.options import opts


def overlap_ratio(rect1, rect2):
    """
    Compute overlap ratio based on IoU between two bounding boxes

    :param rect1: 1d array of [x,y,w,h] or 2d array of N*[x,y,w,h]
    :param rect2: same
    :return: IoU
    """

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def reward(iou):
    """
    Reward function determined by IoU

    :param iou: IoU value between 0 and 1, inclusively
    :return: scalar reward
    """
    if iou > opts['iou_criterion']:
        return 1
    return -1


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:, 1]
        neg_loss = -F.log_softmax(neg_score)[:, 0]
        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)
        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.data[0]

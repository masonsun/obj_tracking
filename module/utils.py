import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize

#import sys
#sys.path.insert(0,'/Users/RobinYen/Documents/DQN/actnet/obj_tracking/')
from options import opts


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


def get_reward(iou):
    """
    Reward function determined by IoU
    :param iou: IoU value between 0 and 1, inclusively
    :return: scalar reward
    """
    if iou > opts['iou_criterion']:
        return 1
    return -1


def get_bbox(action, bbox, img_size, alpha=opts['alpha']):
    """
    Given the action, modify the bounding box accordingly

    :param action: one-hot encoded action
    :param bbox: bounding box [x,y,w,h]
    :param alpha: scaling factor
    :return: new bounding box, boolean indicating terminating action
    """
    # Actions
    action_deltas = [
        [-1, 0, 0, 0],   # left
        [+1, 0, 0, 0],   # right
        [0, -1, 0, 0],   # up
        [0, +1, 0, 0],   # down
        [-2, 0, 0, 0],   # shorten width
        [+2, 0, 0, 0],   # elongate width
        [0, -2, 0, 0],   # shorten height
        [0, +2, 0, 0],   # elongate height
        [0, 0, -1, -1],  # smaller
        [0, 0, +1, +1],  # bigger
        [0, 0, 0, 0],    # stop
    ]

    # retrieve index of selected action
    assert action.sum() == 1.0, "Not one-hot encoded action"
    if isinstance(action, torch.Tensor):
        a = action.numpy()      
    else:
        try:
            a = np.asarray(action)
        except AttributeError:
            print("Cannot handle action data type: {}".format(type(action)))
            return bbox
    
    bbox_n = (bbox.data.numpy())
    
    print("current bbox_n:", bbox_n)
    
    bbox_n[0] = bbox_n[0] + 0.5 * bbox_n[2]
    bbox_n[1] = bbox_n[1] + 0.5 * bbox_n[3]

    deltas = alpha * np.asarray([bbox_n[2], bbox_n[3], bbox_n[2], bbox_n[3]])
    
    a = int(np.argmax(a))

    # stop
    if len(action_deltas) - 1 == a:
        return bbox, True

    # apply actions
    # bbox.data += torch.from_numpy((np.asarray(deltas[a]) * alpha))    
    bbox_n += action_deltas[a]*deltas 
    
    # change to original bbox representation 
    bbox_n[0] -= 0.5 * bbox_n[2] 
    bbox_n[1] -= 0.5 * bbox_n[3]
    # edge handling
    bbox_n[2] = max(5, min(img_size[1], bbox_n[2]))
    bbox_n[3] = max(5, min(img_size[0], bbox_n[3]))

    bbox_n[0] = max(bbox_n[0], 1)
    bbox_n[0] = min(bbox_n[0], img_size[1] - bbox_n[2])
    bbox_n[1] = max(bbox_n[1], 1)
    bbox_n[1] = min(bbox_n[1], img_size[0] - bbox_n[3])

    #bbox_n = bbox_n + 10
    #np.add(bbox_n, 10, out = bbox_n)
    #print("delta:", deltas)
    #print("bbox_n:", bbox_n)

    print("new get_bbox : {} action chosen: {} \n".format(bbox, action_deltas[a]))

    return bbox, False


def epsilon_greedy(action, epsilon,  num_actions=opts['num_actions']):
    """
    Select one-hot encoded action using epsilon-greedy

    :param action: action probability vector (torch variable)
    :param epsilon: probability of exploring
    :param num_actions: number of actions
    :return: one-hot encoded action and its index (torch variables)
    """
    # assign probabilities to each action
    
    explore_prob = epsilon / num_actions
    p = np.full(num_actions, explore_prob)
    p[np.argmax(action.data.numpy())] = 1 - epsilon + explore_prob
    print("action probablilty epsilon_greedy:", p)
    #print(explore_prob, epsilon)
    # one-hot encoding of selected action
    one_hot_action = torch.zeros(num_actions)
    index = np.random.choice(np.arange(num_actions), p=p)
    
    #debug 
    #if i == 10:
    #    index = np.int64(10)
    #exit()

    one_hot_action[index] = 1
    return one_hot_action , torch.LongTensor([[index.item()]]) #torch.LongTensor(one_hot_action.long()) #

def fifo_update(x, y):
    """
    Update tensor with new input in a FIFO manner

    :param x: original tensor
    :param y: new input
    :return: row-wise updated tensor
    """
    return torch.cat((y.view(1, -1), x[1:]), 0)


def crop_image(img, bbox, img_size=opts['img_size'], padding=opts['padding'], valid=False):
    """
    Crop image

    :param img: image
    :param bbox: bounding box representation
    :param img_size: width/height of image
    :param padding: amount of padding
    :param valid: boolean to check bbox representation
    :return: cropped image
    """

    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled = imresize(cropped, (img_size, img_size))
    return scaled


def view_image(img):
    """
    Auxiliary function to view an image

    :param img: image represented as Image object or numpy array
    """
    try:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8))
        img.show()
    except TypeError:
        print("Cannot convert image format: {}".format(type(img)))

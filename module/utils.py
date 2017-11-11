import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from training.options import opts


def apply_action_to_bbox(action, bbox, alpha=opts['alpha']):
    """
    Given the action, modify the bounding box accordingly

    :param action: one-hot encoded action
    :param bbox: bounding box [x,y,w,h]
    :param alpha: scaling factor
    :return: new bounding box, boolean indicating terminating action
    """
    # Actions
    deltas = [
        [-1, 0, 0, 0],   # left
        [+1, 0, 0, 0],   # right
        [0, -1, 0, 0],   # up
        [0, +1, 0, 0],   # down
        [0, 0, -1, 0],   # shorten width
        [0, 0, +1, 0],   # elongate width
        [0, 0, 0, -1],   # shorten height
        [0, 0, 0, +1],   # elongate height
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
    # stop
    a = int(np.argmax(a))
    if len(deltas) - 1 == a:
        return bbox, True

    # apply actions
    bbox += (np.asarray(deltas[a]) * alpha)
    return bbox, False


def epsilon_greedy(action, epsilon, num_actions=opts['num_actions']):
    """
    Select one-hot encoded action from action probabilities using epsilon-greedy

    :param action: action probability vector
    :param epsilon: probability of exploring
    :return: one-hot encoded action
    """
    # assign probabilities to each action
    explore_prob = epsilon / num_actions
    p = np.full(num_actions, explore_prob)
    p[np.argmax(action.numpy())] = 1 - epsilon + explore_prob

    # one-hot encoding of selected action
    one_hot_action = torch.zeros(num_actions)
    index = np.random.choice(np.arange(num_actions), p=p)
    one_hot_action[index] = 1
    return one_hot_action


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

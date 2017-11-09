import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from training.options import opts


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

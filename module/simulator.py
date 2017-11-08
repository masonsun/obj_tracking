import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dataset import Dataset
from img_processing import crop_image


class Simulator(nn.Module, Dataset):
    def __init__(self, action, image, bbox, max_actions=20, alpha=0.03):
        super(Simulator, self).__init__()

        self.action = action
        self.image = image
        self.bbox = np.asarray(bbox)

        self.alpha = alpha
        self.cnt_actions = 0
        self.max_actions = max_actions

        self.deltas = [
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

    # Given the action, modify the bounding box accordingly
    def apply_action_to_bbox(self):
        # retrieve index of selected action
        if isinstance(self.action, torch.Tensor):
            a = self.action.numpy()
        else:
            try:
                a = np.asarray(self.action)
            except AttributeError:
                print("Cannot handle action data type: {}".format(type(self.action)))
                return self.bbox
        a = int(np.argmax(a))

        # stop
        self.cnt_actions += 1
        if len(self.deltas) - 1 == a or self.cnt_actions >= self.max_actions:
            return self.bbox

       # apply actions
        self.bbox += (np.asarray(self.deltas[a]) * self.alpha)
        return self.bbox

    # Given a bounding box, return the corresponding patch in the image
    def get_new_patch(self):
        assert len(self.bbox) == 4
        return crop_image(self.image, self.bbox)

    # Auxiliary function to view the original image
    def view_image(self, img=None):
        if img is None:
            img = self.image
        try:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img, dtype=np.uint8))
            img.show()
        except TypeError:
            print("Cannot convert image format: {}".format(type(img)))

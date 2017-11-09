import torch
import torch.nn as nn
from torch.autograd import Variable


class LRN(nn.Module):
    def __init__(self):
        # Local Response Normalization
        super(LRN, self).__init__()

    def forward(self, x):
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x

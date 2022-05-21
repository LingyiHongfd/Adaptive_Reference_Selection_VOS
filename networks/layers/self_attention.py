import torch
import torch.nn as nn
import torch.nn.functional as F
from visdom import Visdom


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Self_Attention(nn.Module):
    def __init__(self,embed_dim):
        super(Self_Attention, self).__init__()
        self.attention_conv = ResBlock(embed_dim + 1, 1)

    def forward(self, feature, mask,show=False):
        concat_fm = torch.cat((feature, mask), dim=1)
        fm = self.attention_conv(concat_fm)
        fm = torch.sigmoid(fm)
        if show:
            viz = Visdom(use_incoming_socket=False)
            for i in range(fm.size()[0]-1):
                viz.image(fm[i+1, :, :, :].squeeze(0))
        return feature * fm

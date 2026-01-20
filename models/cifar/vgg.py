'''
VGG11/13/16/19 in Pytorch.

Implementation FROM: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
'''
import torch
import torch.nn as nn
from collections import OrderedDict  # Add this import

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        layer_idx = 0  # To uniquely identify layers
        for x in cfg:
            if x == 'M':
                layers.append(('maxpool_' + str(layer_idx), nn.MaxPool2d(kernel_size=2, stride=2)))
                layer_idx += 1
            else:
                layers.append(('conv_' + str(layer_idx), nn.Conv2d(in_channels, x, kernel_size=3, padding=1)))
                layer_idx += 1
                layers.append(('bn_' + str(layer_idx), nn.BatchNorm2d(x)))
                layer_idx += 1
                layers.append(('relu_' + str(layer_idx), nn.ReLU(inplace=True)))
                layer_idx += 1
                in_channels = x
        layers.append(('avgpool', nn.AvgPool2d(kernel_size=1, stride=1)))
        return nn.Sequential(OrderedDict(layers))

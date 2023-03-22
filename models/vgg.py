'''
VGG11/13/16/19 in Pytorch.
modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
'''

import torch
import torch.nn as nn
from models.resnet import generate_norm_layer

cfg = {
    'vgg9': [32, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_cls = {
    'vgg9': [4096, 512, 512],  # input_dim, out_dim, out_dim, ...
    'normal': [4096, 4096, 4096]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, norm_name='bn', num_groups=-1):
        super(VGG, self).__init__()
        if vgg_name not in cfg:
            raise RuntimeError("{} is not optional vgg network ".format(vgg_name))

        self.norm_name = norm_name
        self.num_groups = num_groups

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self._make_classifier(vgg_name, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = nn.Sequential()
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.add_module("maxpool%d" % len(layers), nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                mini_layers = nn.Sequential()
                mini_layers.add_module("conv1", nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                if self.norm_name != "nobn":
                    mini_layers.add_module("bn1", generate_norm_layer(num_channels_or_features=x, norm_name=self.norm_name, num_groups=self.num_groups))
                mini_layers.add_module("relu1", nn.ReLU(inplace=True))
                layers.add_module("sub%d" % len(layers), mini_layers)
                in_channels = x
        layers.add_module("avgpool%d" % len(layers), nn.AvgPool2d(kernel_size=1, stride=1))
        return layers

    def _make_classifier(self, vgg_name, num_classes):
        if vgg_name == 'vgg9':
            dim_list = cfg_cls[vgg_name]
        else:
            dim_list = cfg_cls['normal']
        classifier = nn.Sequential()
        for i in range(1, len(dim_list)):
            classifier.add_module(str(len(classifier)), nn.Linear(dim_list[i - 1], dim_list[i]))
            classifier.add_module(str(len(classifier)), nn.ReLU(inplace=True))
        classifier.add_module(str(len(classifier)), nn.Linear(dim_list[-1], num_classes))
        return classifier

def test():
    model = VGG('vgg9', num_classes=11).cuda()
    x = torch.randn(2, 3, 32, 32).cuda()
    y = model(x)
    print(y.size())
    print(model)


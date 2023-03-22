'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def generate_norm_layer(num_channels_or_features, norm_name, num_groups=-1):
    norm_arg = {}
    if norm_name == "bn":
        norm = nn.BatchNorm2d
        norm_arg['num_features'] = num_channels_or_features
    elif norm_name == "gn":
        norm = nn.GroupNorm
        norm_arg['num_groups'] = num_groups
        norm_arg['num_channels'] = num_channels_or_features
    elif norm_name == "ln":
        norm = nn.GroupNorm
        norm_arg['num_groups'] = 1
        norm_arg['num_channels'] = num_channels_or_features
    elif norm_name == "in":
        norm = nn.GroupNorm
        norm_arg['num_groups'] = num_channels_or_features
        norm_arg['num_channels'] = num_channels_or_features
    return norm(**norm_arg)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm_layer, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(num_channels_or_features=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(num_channels_or_features=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.add_module('conv1', nn.Conv2d(in_planes, self.expansion * planes,
                                                        kernel_size=1, stride=stride, bias=False))
            self.shortcut.add_module('bn1',
                                     norm_layer(num_channels_or_features=self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm_layer, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(num_channels_or_features=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(num_channels_or_features=planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(num_channels_or_features=self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.add_module('conv1', nn.Conv2d(in_planes, self.expansion * planes,
                                                        kernel_size=1, stride=stride, bias=False))
            self.shortcut.add_module('bn1', norm_layer(num_channels_or_features=self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, norm_function):
        '''
        :param block: Shallow resnets -> BasicBlock, Deep ones -> Bottleneck
        :param num_blocks: List, indicates each block
        :param num_classes:
        '''
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_function = norm_function
        self.bn1 = self.norm_function(num_channels_or_features=64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.norm_function, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits

    def forward_with_feat(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feature = torch.flatten(out, 1)
        logits = self.classifier(feature)
        return logits, feature

def ResNet18(num_classes, norm_name, num_groups):
    norm_function = partial(generate_norm_layer, norm_name=norm_name, num_groups=num_groups)
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_function=norm_function)


def ResNet34(num_classes, norm_name, num_groups):
    norm_function = partial(generate_norm_layer, norm_name=norm_name, num_groups=num_groups)
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, norm_function=norm_function)


def ResNet50(num_classes, norm_name, num_groups):
    norm_function = partial(generate_norm_layer, norm_name=norm_name, num_groups=num_groups)
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, norm_function=norm_function)


def ResNet101(num_classes, norm_name, num_groups):
    norm_function = partial(generate_norm_layer, norm_name=norm_name, num_groups=num_groups)
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, norm_function=norm_function)

def ResNet152(num_classes, norm_name, num_groups):
    norm_function = partial(generate_norm_layer, norm_name=norm_name, num_groups=num_groups)
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, norm_function=norm_function)


if __name__ == '__main__':
    model = ResNet18(10, 'bn', -1)
    print(model)
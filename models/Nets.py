#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import batch_norm
from torchvision import models
import math

class MLP_emnist(nn.Module):
    def __init__(self, num_classes):
        super(MLP_emnist, self).__init__()
        self.n_cls = 10
        self.fc1 = nn.Linear(1 * 28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)
        return logits
    
    def forward_with_feat(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        feature = F.relu(self.fc2(x))
        logits = self.classifier(feature)
        return logits, feature
class MLP(nn.Module):
    def __init__(self, input_channel, hidden_channel, num_classes):
        super(MLP, self).__init__()
        self.input_channel = input_channel
        self.fc1 = nn.Linear(input_channel, hidden_channel)
        self.fc2 = nn.Linear(hidden_channel, hidden_channel)
        self.classifier = nn.Linear(hidden_channel, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_channel)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)
        return logits
    
    def forward_with_feat(self, x):
        x = x.view(-1, self.input_channel)
        x = F.relu(self.fc1(x))
        feature = F.relu(self.fc2(x))
        logits = self.classifier(feature)
        return logits, feature

class CNNCifar(nn.Module):
    # As same as model of cifar10/cifar100 in FedDyn
    def __init__(self, num_channels, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)
        return logits

    def forward_with_feat(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        feature = F.relu(self.fc2(x))
        logits = self.classifier(feature)
        return logits, feature

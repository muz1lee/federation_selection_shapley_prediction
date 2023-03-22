#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math


class DAM(nn.Module):
    def __init__(self, in_dim, gate_type='relu_tanh'):
        super(DAM, self).__init__()
        self.in_dim = in_dim
        self.gate_type = gate_type
        k = 5
        self.mu = nn.Parameter(torch.arange(self.in_dim).float() / self.in_dim * k, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=False)
        if gate_type != 'relu_tanh':
            self.beta = nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)
        self.register_parameter('mu', self.mu)
        self.tanh = nn.Tanh()
        if self.gate_type == 'relu_tanh':
            self.relu = nn.ReLU()

    def mask(self):
        if self.gate_type == 'relu_tanh':
            mask = self.relu(self.tanh(self.alpha * (self.mu + self.beta)))
        elif self.gate_type == 'tanh':
            mask = self.tanh(self.alpha * (self.mu + self.beta))
        return mask

    def forward(self, x):
        # return x
        return x * self.mask()


class DAM_2d(nn.Module):
    def __init__(self, in_channel, gate_type='relu_tanh'):
        super(DAM_2d, self).__init__()
        self.in_channel = in_channel
        self.gate_type = gate_type
        k = 5
        mu_1d = torch.arange(self.in_channel).float() / self.in_channel * k
        self.mu = nn.Parameter(mu_1d.reshape(-1, self.in_channel, 1, 1), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=False)
        if gate_type != 'relu_tanh':
            self.beta = nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)
        self.register_parameter('mu', self.mu)
        self.tanh = nn.Tanh()
        if self.gate_type == 'relu_tanh':
            self.relu = nn.ReLU()


    def mask(self):
        if self.gate_type == 'relu_tanh':
            mask = self.relu(self.tanh((self.alpha ** 2) * (self.mu + self.beta)))
        elif self.gate_type == 'tanh':
            mask = self.tanh((self.alpha ** 2) * (self.mu + self.beta))

        return mask

    def forward(self, x):
        # return x
        return x * self.mask()



class CNNCifar_order(nn.Module):
    def __init__(self, args):
        super(CNNCifar_order, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.dam1 = DAM_2d(6, gate_type=args.gate_type)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dam2 = DAM_2d(16, gate_type=args.gate_type)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dam_f1 = DAM(120, gate_type=args.gate_type)
        self.fc2 = nn.Linear(120, 100)
        self.dam_f2 = DAM(100, gate_type=args.gate_type)
        self.fc3 = nn.Linear(100, args.num_classes)

        # TEST DAM
        # 这个初始化比默认初始化好一点
        # for module in self.modules():
        #     if isinstance(module, nn.Conv2d):
        #         n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        #         module.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(module, nn.BatchNorm2d):
        #         module.weight.data.fill_(0.5)
        #         module.bias.data.zero_()


        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.dam1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dam2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.dam_f1(F.relu(self.fc1(x)))
        x = self.dam_f2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

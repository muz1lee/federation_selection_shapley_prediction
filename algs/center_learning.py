import os
import sys
import torch
import numpy as np
import copy
from utils import wandb
from models.test import test_center
from utils.log_tools import fprint
from global_tools import save_pkl, append_in_dict, mean_in_dict
from algs.alg_tools import compute_weight_quality, get_optimizer
from torch.nn import CrossEntropyLoss
from utils.train_utils import set_data_loader
from algs.framework import BasicFramework

import time

def VarLoss(model, head_var):
    var = []
    if head_var == 0:
        return torch.tensor([0]).cuda()

    if isinstance(head_var, dict):
        for n, w in model.named_parameters():
            if n in head_var:
                var.append(head_var[n] - w.std().view(1))
    elif isinstance(head_var, float):
        for n, w in model.named_parameters():
            var.append(head_var - w.std().view(1))
    var = torch.cat(var)
    # var /= ep_all #实验vcrfj2p2
    # var = torch.clamp(var, min=0).sum()
    var = torch.norm(var, p=2)
    return var

def VarLoss_relu(model, head_var):
    var = []
    if head_var == 0:
        return torch.tensor([0]).cuda()

    if isinstance(head_var, dict):
        for n, w in model.named_parameters():
            if n in head_var:
                var.append(head_var[n] - w.std().view(1))
    elif isinstance(head_var, float):
        for n, w in model.named_parameters():
            if "weight" in n and "bn" not in n:
                var.append(head_var - w.std().view(1))
    var = torch.cat(var)
    # var /= ep_all #实验vcrfj2p2
    var = torch.clamp(var, min=0).sum()
    # var = torch.norm(var, p=2)
    return var

class CenterLearning(BasicFramework):
    def __init__(self, dataset, model, args, head_var):
        func_info = "{}->{}".format(self.__class__.__name__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        # 接受切分好的数据集
        self.dataset = dataset
        self.data_loader = set_data_loader(dataset, args)
        self.model = model
        self.start_epoch = 1
        self.args = args
        self.base_dir = args.base_dir
        self.lr = self.args.lr
        self.loss_ce = CrossEntropyLoss(reduction='mean')
        
        self.COMPARED_METRIC_NAME = "test_acc"

        
        # var loss
        if args.varloss_type == "norm":
            self.loss_var = VarLoss
        elif args.varloss_type == "relu":
            self.loss_var = VarLoss_relu
        self.lamda = self.args.lamda
        self.head_var = head_var

        self.optimizer = get_optimizer(self.model, optim_name=self.args.optim, lr=self.lr,
                                       weight_decay=self.args.weight_decay, momentum=self.args.momentum)

        self.history = {}  # key is user_id

    def loss_func(self, log_probs, labels, model):
        loss_ce = self.loss_ce(log_probs, labels)
        loss_var = self.loss_var(model, self.head_var)
        loss = loss_ce + self.lamda * loss_var
        return loss, {'loss': loss.item(),'loss_ce':loss_ce.item(), 'loss_var':loss_var.item()}

    def save_results(self, name):
        results = {'args': self.args, 'history': self.history}
        super().save_to_file(name, results)


    def local_update(self, model):

        """
        one epoch training process of center learning
        :arg bs_train, device, local_ep, momentum, optim, weight_decay
        :param model: equipped function [ forward ]
        :return: model's state_dict, train_loss_avg ( average on all of the local eps )
        """
        func_info = "{}->{}".format(CenterLearning.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        model.train()
        loss_results = {}
        for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
            images, labels = images.cuda(), labels.cuda()
            model.zero_grad()
            log_probs = model(images)
            loss, loss_result = self.loss_func(log_probs, labels, model)
            loss.backward()
            self.optimizer.step()
            loss_results =append_in_dict(origin_dict=loss_results ,new_dict=loss_result)

        avg_loss_results = mean_in_dict(loss_results)
        
        return model.state_dict(), avg_loss_results

    def train(self):
        """
            federated training in the communication rounds (self.args.epochs).
        :arg epochs, lr_decay, save_clients, test_freq
        :return:
        """
        func_info = "{}->{}".format(CenterLearning.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            fprint("Round {}, lr: {:.6f}".format(epoch, self.lr))
            old_model = copy.deepcopy(self.model)

            weight_local, train_results = self.local_update(self.model)
            test_results = test_center(self.model, self.data_loader['test'], self.args)
            # weight_quality = compute_weight_quality(self.model.state_dict(), old_model_dict=old_model.state_dict())
            weight_quality =  None
            # print results
            fprint("Epoch: {}  {},  {}".format(epoch, train_results, test_results))
            results_cat = {'epoch':epoch, **train_results, **test_results}
            is_best = False
            if epoch == self.start_epoch or results_cat[self.COMPARED_METRIC_NAME] > self.best_info[self.COMPARED_METRIC_NAME]:
                is_best = True
            self.history = self.record_log(self.history, results_cat, weight_quality, is_best=is_best)

            self.lr *= self.args.lr_decay
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
            self.lamda *= self.args.lamda_decay
            torch.cuda.empty_cache()

if __name__ == '__main__':
    # test code
    pass
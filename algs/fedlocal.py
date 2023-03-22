import os
import sys
import torch
import numpy as np
import copy
from utils import wandb
from torch.utils.data import DataLoader
from models.test import test_center
from utils.log_tools import fprint
from global_tools import save_pkl, append_in_dict
from algs.alg_tools import DatasetSplit, Aggregater, get_optimizer, compute_weight_quality, check_client_class_distribution
from algs.framework import BasicFramework
from torch.nn import CrossEntropyLoss


class FedLocal(BasicFramework):
    def __init__(self, dataset_train, dataset_test, dict_users_train, dict_users_test, model, args):
        func_info = "{}->{}".format(FedLocal.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        # 接受切分好的数据集
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users_train = dict_users_train
        self.dict_users_test = dict_users_test
        self.loader_global_test = DataLoader(self.dataset_test, batch_size=args.bs_test,
                                             shuffle=False)
        self.model = model
        self.start_epoch = 1
        self.args = args
        self.base_dir = args.base_dir
        self.lr = self.args.lr
        self.num_users = self.args.num_users
        self.loss_ce = CrossEntropyLoss(reduction='mean')
        self.COMPARED_METRIC_NAME = "test_acc"
        
        # save data split information
        last_folder = args.base_dir.split('/')[-1]
        if last_folder.isdigit() and last_folder != '1':
            fprint("now_round:%s dont need to save class distribution" % last_folder, level='DEBUG')
            self.data_split = None
        else:
            self.data_split = check_client_class_distribution(targets=dataset_train.targets, dict_users=dict_users_train, run=None, args=args)
        self.history = {}
        self.clients_history = {i: {} for i in range(self.args.num_users)}  # key is user_id
        
    def save_results(self, name):
        results = {'args': self.args, 'history': self.history, 'data_split': self.data_split, "clients_history":self.clients_history}
        super().save_to_file(name, results)

    def local_update(self, model, idx_data, idx_test_data):
        """
        the update process of local client
        :arg bs_train, device, local_ep, momentum, optim, weight_decay
        :param model: equipped function [ forward ]
        :param idx_data: datas' indexes
        :return: model's state_dict, train_loss_avg ( average on all of the local eps )
        """
        func_info = "{}->{}".format(FedLocal.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        model.train()
        loader_local_train = DataLoader(DatasetSplit(self.dataset_train, idx_data), batch_size=self.args.bs_train,
                                        shuffle=True)
        loader_local_test = DataLoader(DatasetSplit(self.dataset_test, idx_test_data), batch_size=self.args.bs_test,
                                       shuffle=False)
        optimizer = get_optimizer(model, optim_name=self.args.optim, lr=self.lr,
                                  weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        local_eps = self.args.local_ep
        client_history = {}
        for _ in range(1, local_eps + 1):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(loader_local_train):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                log_probs = model(images)
                loss_ce = self.loss_ce(log_probs, labels)
                loss = loss_ce
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss = sum(batch_loss) / len(batch_loss)
            test_results = test_center(model, loader_local_test)
            global_test_results = test_center(model, self.loader_global_test, global_test=True)
            fprint("epoch: {} loss: {} {}".format(_, epoch_loss, test_results))

            # weight_quality = {}  # {'var_layer_x', 'mean_layer_x', ...} may be provided by some unrealized function  haha
            # weight_quality = compute_weight_quality(model.state_dict(), old_model_dict=self.model.state_dict())
            weight_quality = None
            # concat parts of dicts except weight_quality.
            results_cat = {'epoch': _, 'train_loss': epoch_loss, **test_results, **global_test_results}
            client_history = self.record_log(client_history, results_cat, weight_quality, glob=False)
            
        torch.cuda.empty_cache()
        return client_history

    def train(self):
        """
            federated training in the communication rounds (self.args.epochs).
        :arg epochs, lr_decay, save_clients, test_freq
        :return:
        """
        func_info = "{}->{}".format(FedLocal.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        for idx in range(self.args.num_users):
            fprint("User {} starts training".format(idx))
            model_local = copy.deepcopy(self.model)
            self.clients_history[idx] = self.local_update(model_local, idx_data=self.dict_users_train[idx],
                                              idx_test_data=self.dict_users_test[idx])

if __name__ == '__main__':
    # test code
    pass

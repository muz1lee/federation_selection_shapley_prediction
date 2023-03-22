import os
import sys
import torch
import numpy as np
import copy
from utils import wandb
from torch.utils.data import DataLoader
from models.test import test_center
from utils.log_tools import fprint
from global_tools import save_pkl, append_in_dict, mean_in_dict, set_random_seed
from algs.alg_tools import DatasetSplit, Aggregater, get_optimizer, get_models_array, load_model_from_params
from torch.nn import CrossEntropyLoss
from algs.fedavg import FedAvg
import torch.nn as nn
import torch.nn.functional as F


class SpreadModel(nn.Module):
    def __init__(self, ws, margin):
        super().__init__()
        self.ws = nn.Parameter(ws)
        self.margin = margin

    def forward(self):
        ws_norm = F.normalize(self.ws, dim=1)
        cos_dis = 0.5 * (1.0 - torch.mm(ws_norm, ws_norm.transpose(0, 1)))

        d_mat = torch.diag(torch.ones(self.ws.shape[0]))
        d_mat = d_mat.to(self.ws.device)

        cos_dis = cos_dis * (1.0 - d_mat)

        indx = ((self.margin - cos_dis) > 0.0).float()
        loss = (((self.margin - cos_dis) * indx) ** 2).mean()
        return loss

class FedAwS(FedAvg):
    def __init__(self, dataset_train, dataset_test, dict_users_train, dict_users_test, model, args, idxs_all_users=None):
        super().__init__(dataset_train, dataset_test, dict_users_train, dict_users_test, model, args)
        self.agger = Aggregater()
        self.COMPARED_METRIC_NAME = 'global_test_loss'
        self.patience_earlystop = args.patience_earlystop
        if idxs_all_users is None:
            idxs_all_users = list(range(self.args.num_users))
        self.idxs_all_users = idxs_all_users

        self.last_params_users = None
        self.tru_batch = args.tru_batch
        self.dist = self.data_split.iloc[:,4:].to_numpy()
        self.dist = self.dist / self.dist.sum(axis=1, keepdims=True)
        self.dist = torch.FloatTensor(self.dist)
        self.aws_margin = args.aws_margin
        self.aws_steps = args.aws_steps
        self.aws_lr = args.aws_lr

    def save_results(self, name):
        results = {'args': self.args, 'clients_history': self.clients_history, 'data_split': self.data_split, 'history': self.history}
        super().save_to_file(name, results)

    def choose_users(self):
        func_info = "{}->{}".format(FedAwS.__qualname__, sys._getframe().f_code.co_name)
        num_users = max(int(self.args.frac * self.args.num_users), 1)
        if self.args.frac == 1.0:
            return self.idxs_all_users, num_users

        idxs_users = np.random.choice(self.idxs_all_users, num_users, replace=False)
        idxs_users = np.sort(idxs_users)

        return idxs_users, num_users

    def update_global_classifier(self, model):
        ws = model.classifier.weight.data
        sm = SpreadModel(ws, margin=self.aws_margin)

        optimizer = torch.optim.SGD(
            sm.parameters(), lr=self.args.aws_lr, momentum=0.9
        )

        for _ in range(self.args.aws_steps):
            loss = sm.forward()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.load_state_dict({"classifier.weight": sm.ws.data}, strict=False)


    def loss_func(self, log_probs, labels):
        loss_ce = self.loss_ce(log_probs, labels)
        loss = loss_ce
        return loss, {'loss': loss.item(),'loss_ce':loss_ce.item()}

    def local_update(self, model, epoch, idx_data, idx_test_data, user_id):
        """
        the update process of local client
        :arg bs_train, device, local_ep, momentum, optim, weight_decay
        :param model: equipped function [ forward ]
        :param epoch: the index of communication round.
        :param idx_data: datas' indexes
        :param idx_test_data: test datas' indexes
        :param user_id: int
        :return: model's state_dict, train_loss_avg ( average on all of the local eps )
        """
        func_info = "{}->{}".format(FedAwS.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')

        loader_local_train = DataLoader(DatasetSplit(self.dataset_train, idx_data), batch_size=self.args.bs_train,
                                        shuffle=True, num_workers=4)
        loader_local_test = DataLoader(DatasetSplit(self.dataset_test, idx_test_data), batch_size=self.args.bs_test,
                                       shuffle=False, num_workers=4)
        #! weight decay set to 0 default, and conduct by the norm function realized by myslef.
        optimizer = get_optimizer(model, optim_name=self.args.optim, lr=self.lr,
                                  weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        epoch_results = {}
        local_eps = self.args.local_ep

        dist = self.dist[user_id]

        batch_cum = 0
        for _ in range(1 + local_eps * (epoch - 1), local_eps + 1 + local_eps * (epoch - 1)):
            model.train()
            loss_results = {}
            for batch_idx, (images, labels) in enumerate(loader_local_train):
                images, labels = images.cuda(), labels.long().cuda()
                model.zero_grad()
                log_probs, feats = model.forward_with_feat(images)

                loss, loss_result = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                loss_results = append_in_dict(origin_dict=loss_results ,new_dict=loss_result)

                batch_cum += 1
                if batch_cum == self.tru_batch:
                    break

            avg_loss_results = mean_in_dict(loss_results)
            epoch_results = append_in_dict(origin_dict=epoch_results ,new_dict=avg_loss_results)
            fprint("epoch: {} train_loss:{}".format(_, avg_loss_results), level='DEBUG')
            if self.args.local_test:
                test_results = test_center(model, loader_local_test)
                global_test_results = test_center(model, self.loader_global_test, global_test=True)
                fprint("epoch: {} train_loss:{} {} {}".format(_, avg_loss_results['loss'], test_results, global_test_results), level='DEBUG')
                # weight_quality = compute_weight_quality(model.state_dict(), old_model_dict=self.model.state_dict())
                weight_quality = None
                # concat parts of dicts except weight_quality.
                results_cat = {'epoch': _, 'train_loss': avg_loss_results['loss'], **test_results, **global_test_results}
                self.clients_history[user_id] = self.record_log(self.clients_history[user_id], results_cat, weight_quality, glob=False)

            if batch_cum == self.tru_batch:
                break
        avg_epoch_results = {**mean_in_dict(epoch_results)}
        return model, avg_epoch_results

    def train(self):
        """
            federated training in the communication rounds (self.args.epochs).
        :arg epochs, lr_decay, save_clients, test_freq
        :return:
        """
        func_info = "{}->{}".format(FedAwS.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        
        current_patience = self.patience_earlystop

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            epoch_results = {}
            idxs_users, num_users = self.choose_users()
            fprint("Round {}, lr: {:.6f}, {}".format(epoch, self.lr, idxs_users), level='DEBUG')
            model_list = []
            for idx in idxs_users:
                set_random_seed(seed=epoch * 1000 + idx) #TRIAL did not consider random of choose users.
                model_local = copy.deepcopy(self.model)
                model_local, local_results = self.local_update(model_local, epoch,
                                                                idx_data=self.dict_users_train[
                                                                    idx], idx_test_data=
                                                                self.dict_users_test[idx],
                                                                user_id=idx)
                epoch_results = append_in_dict(epoch_results, local_results)
                self.agger.increase(model_local.state_dict(), 1)
                # if self.args.save_clients or epoch <= self.start_epoch + 5:
                model_list.append(copy.deepcopy(dict(model_local.named_parameters())))
            model_list.append(copy.deepcopy(dict(self.model.named_parameters())))
            params_array = get_models_array(model_list)
            if self.args.save_clients or epoch <= self.start_epoch + 5:
                np.save(os.path.join(self.base_dir, "models_{}.npy".format(epoch)), params_array)

            # update global weights
            if len(idxs_users) != 0:
                weight_glob = self.agger.aggregate()
                self.agger.clear()
                # copy weight to net_glob
                self.model.load_state_dict(weight_glob)
                self.update_global_classifier(model=self.model)

            avg_epoch_results = mean_in_dict(epoch_results)
            global_test_results = test_center(self.model, self.loader_global_test, global_test=True)
            fprint("Round: {}  {}".format(epoch, global_test_results['global_test_acc']), level='INFO')
            # weight_quality = compute_weight_quality(self.model.state_dict())
            weight_quality = None
            results_cat = {"epoch": epoch, **avg_epoch_results,  **global_test_results}
            is_best = False
            if epoch == self.start_epoch or results_cat[self.COMPARED_METRIC_NAME] < self.best_info[self.COMPARED_METRIC_NAME]:
                is_best = True
                current_patience = self.patience_earlystop
            else:
                current_patience -= 1
            self.history = self.record_log(self.history, results_cat, weight_quality, is_best=is_best)

            self.lr *= self.args.lr_decay
            torch.cuda.empty_cache()

            if len(idxs_users) == 0:
                break
            if current_patience <= 0 and self.patience_earlystop != -1:
                break

if __name__ == '__main__':
    # test code
    pass

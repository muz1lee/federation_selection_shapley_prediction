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


class FedDyn(FedAvg):
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
        self.alpha_coef = args.dyn_alpha
        

    def save_results(self, name):
        results = {'args': self.args, 'clients_history': self.clients_history, 'data_split': self.data_split, 'history': self.history}
        super().save_to_file(name, results)

    def choose_users(self):
        func_info = "{}->{}".format(FedDyn.__qualname__, sys._getframe().f_code.co_name)
        num_users = max(int(self.args.frac * self.args.num_users), 1)
        if self.args.frac == 1.0:
            return self.idxs_all_users, num_users

        idxs_users = np.random.choice(self.idxs_all_users, num_users, replace=False)
        idxs_users = np.sort(idxs_users)

        return idxs_users, num_users

    def loss_func(self, log_probs, labels, model, avg_mdl_param, local_grad_vector, alpha_coef):
        loss_ce = self.loss_ce(log_probs, labels)
        
        local_par_list = None
        for param in model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
            # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
        loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
        loss = loss_ce + loss_algo
        return loss, {'loss': loss.item(),'loss_ce':loss_ce.item(), 'loss_algo':loss_algo.item()}

    def local_update(self, model, alpha_coef, avg_mdl_param, local_grad_vector, epoch, idx_data, idx_test_data, user_id):
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
        func_info = "{}->{}".format(FedDyn.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')

        loader_local_train = DataLoader(DatasetSplit(self.dataset_train, idx_data), batch_size=self.args.bs_train,
                                        shuffle=True, num_workers=4)
        loader_local_test = DataLoader(DatasetSplit(self.dataset_test, idx_test_data), batch_size=self.args.bs_test,
                                       shuffle=False, num_workers=4)
        #! weight decay set to 0 default, and conduct by the norm function realized by myslef.
        optimizer = get_optimizer(model, optim_name=self.args.optim, lr=self.lr,
                                  weight_decay=self.args.weight_decay+alpha_coef, momentum=self.args.momentum)
        n_par = get_models_array([self.model]).shape[1]
        max_norm = 10

        epoch_results = {}
        local_eps = self.args.local_ep

        batch_cum = 0
        for _ in range(1 + local_eps * (epoch - 1), local_eps + 1 + local_eps * (epoch - 1)):
            model.train()
            loss_results = {}
            for batch_idx, (images, labels) in enumerate(loader_local_train):
                images, labels = images.cuda(), labels.long().cuda()
                model.zero_grad()
                log_probs = model(images)
                loss, loss_result = self.loss_func(log_probs, labels,  model, avg_mdl_param, local_grad_vector, alpha_coef)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
                optimizer.step()
                loss_results = append_in_dict(origin_dict=loss_results ,new_dict=loss_result)

                batch_cum += 1
                if batch_cum == self.tru_batch:
                    break

            avg_loss_results = mean_in_dict(loss_results)
            # if self.args.weight_decay != 0:
            #     # Add L2 loss to complete f_i
            #     params = get_models_array([model], n_par)
            #     avg_loss_results[loss] += (alpha_coef+self.args.weight_decay)/2 * np.sum(params * params)
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
        func_info = "{}->{}".format(FedDyn.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        
        current_patience = self.patience_earlystop

        n_par = len(get_models_array([self.model])[0])
        init_par_list=get_models_array([self.model], n_par)[0]
        local_param_list = np.zeros((self.num_users, n_par)).astype('float32')
        clnt_params_list  = np.ones(self.num_users).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par

        cld_model = copy.deepcopy(self.model).cuda()
        cld_mdl_param = get_models_array([cld_model], n_par)[0]

        weight_list = np.asarray([len(self.dict_users_train[i]) for i in range(self.num_users)])
        weight_list = weight_list / np.sum(weight_list) * self.num_users

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            epoch_results = {}
            idxs_users, num_users = self.choose_users()
            fprint("Round {}, lr: {:.6f}, {}".format(epoch, self.lr, idxs_users), level='DEBUG')
            model_list = []
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32).cuda()

            for idx in idxs_users:
                set_random_seed(seed=epoch * 1000 + idx) #TRIAL did not consider random of choose users.
                model_local = copy.deepcopy(cld_model)
                # Warm start from current avg model
                model_local.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
                for params in model_local.parameters():
                    params.requires_grad = True
                
                alpha_coef_adpt = self.alpha_coef / weight_list[idx] # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[idx], dtype=torch.float32).cuda()
                model_local, local_results = self.local_update(model_local, alpha_coef_adpt, cld_mdl_param_tensor, local_param_list_curr, epoch,
                                                                idx_data=self.dict_users_train[
                                                                    idx], idx_test_data=
                                                                self.dict_users_test[idx],
                                                                user_id=idx)
                curr_model_par = get_models_array([model_local], n_par)[0]

                local_param_list[idx] += curr_model_par - cld_mdl_param
                clnt_params_list[idx] = curr_model_par
                
                epoch_results = append_in_dict(epoch_results, local_results)
                # if self.args.save_clients or epoch <= self.start_epoch + 5:
                model_list.append(copy.deepcopy(dict(model_local.named_parameters())))
            
            model_list.append(copy.deepcopy(dict(self.model.named_parameters())))
            params_array = get_models_array(model_list)
            if self.args.save_clients or epoch <= self.start_epoch + 5:
                np.save(os.path.join(self.base_dir, "models_{}.npy".format(epoch)), params_array)

            avg_mdl_param = np.mean(clnt_params_list[idxs_users], axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)
            # update global weights
            if len(idxs_users) != 0:
                # copy weight to net_glob
                self.model = load_model_from_params(self.model, avg_mdl_param)
                cld_model  = load_model_from_params(cld_model, cld_mdl_param)
        
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

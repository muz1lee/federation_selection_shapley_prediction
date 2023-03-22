from utils import wandb
import torch
import copy
import numpy as np
import pandas as pd
from utils.log_tools import fprint
from torch.utils.data import Dataset

def flatten_model(model):
    flat_param = None
    for param in model.parameters():
        if not isinstance(flat_param, torch.Tensor):
            flat_param = param.view(-1)
        else:
            flat_param = torch.cat((flat_param, param.view(-1)), 0)
    return flat_param

def load_model_from_params(model, params):
    dict_param = copy.deepcopy(dict(model.named_parameters()))
    idx = 0
    for name, param in model.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)))
        idx += length
    
    model.load_state_dict(dict_param)
    return model

def get_models_array(model_list, n_par=None):
    """ turn torch parameters of models (exclude bn' stats) into np.array with shape (num of models, parameter size).
    """
    def dict_generater(x):
        if isinstance(x, dict):
            generater = x.items()
        else:
            generater = x.named_parameters()
        return generater

    if n_par==None:
        exp_mdl = model_list[0]
        generater = dict_generater(exp_mdl)
        n_par = 0
        for name, param in generater:
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        generater = dict_generater(mdl)
        for name, param in generater:
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)



def compute_weight_var(weight_dict, remove_bn=True):
    var = []
    var_weight = {}
    for n, w in weight_dict.items():
        if remove_bn and 'bn' in n:
            continue
        if 'weight' in n:
            var.append(w.std().view(1))
            var_weight[n] = w.std()
    var_sum = torch.cat(var).sum()

    return var_sum, var_weight

def get_fed_algo(alg_name):
    alg_name = alg_name.lower()
    if alg_name == "fedlocal":
        return fedlocal.FedLocal
    elif alg_name == "fedavg":
        return fedavg.FedAvg
    elif alg_name == "fedrs":
        return fedrs.FedRS
    elif alg_name == "feddyn":
        return feddyn.FedDyn
    elif alg_name == "moon":
        return moon.Moon
    elif alg_name == "fedaws":
        return fedaws.FedAwS


class DatasetSplit(Dataset):
    """
        数据集和Dataloader之间的接口。
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



# ! [wandb] Deprecated 
class ProcessLogger(object):
    """
        record train/test log information
    """

    def __init__(self, use_wandb=True):
        self.train = {}
        self.test = {}
        self.best_info = {}
        self.use_wandb = use_wandb

    def record_train(self, epoch, info):
        message = ' '.join(['{}: {:.5f}'.format(key, value) for key, value in info.items()])
        fprint('Epoch {}: {}'.format(epoch, message))
        self.train[epoch] = info
        # wandb
        if self.use_wandb:
            wandb.log({**{'epoch': epoch}, **info})

    def record_test(self, epoch, info):
        message = ' '.join(['{}: {:.5f}'.format(key, value) for key, value in info.items()])
        fprint('Test: {}'.format(message))
        self.test[epoch] = info
        # wandb
        if self.use_wandb:
            wandb.log({**{'epoch': epoch}, **info})

    def record_best_info(self, epoch, test_results, model_state=None):
        """
            save best information and best model state.
            inherited and changed for need.
        :arg save_clients, base_dir
        :param test_results:
        :param epoch:
        :param clients_models:
        :return:
        """

        self.best_info["epoch"] = epoch
        wandb.run.summary['best_epoch'] = epoch
        for key, value in test_results.items():
            self.best_info[key] = value
            wandb.run.summary['best_' + key] = value
        
        if model_state is not None:
            self.best_info["model"] = copy.deepcopy(model_state)

        # # [SAVE] according to acc_test, we save the best model.
        # if self.args.save_model:
        #     model_save_path = os.path.join(self.subfolder_models, 'model_{}.pt'.format(epoch))
        #     torch.save(self.best_info, model_save_path)
    
    

class Aggregater(object):
    """
        federated
    """

    def __init__(self):
        self.weight_glob = None
        self.num_coef = []

    def increase(self, weight_local, coefficient=1):
        if self.weight_glob is None:
            self.weight_glob = copy.deepcopy(weight_local)
            for k in self.weight_glob.keys():
                self.weight_glob[k] = weight_local[k] * coefficient
        else:
            for k in self.weight_glob.keys():
                self.weight_glob[k] += weight_local[k] * coefficient
        self.num_coef.append(coefficient)

    def aggregate(self):
        for k in self.weight_glob.keys():
            self.weight_glob[k] = torch.div(self.weight_glob[k], sum(self.num_coef)) # 有问题
        return self.weight_glob

    def clear(self):
        self.weight_glob = None
        self.num_coef = []

class AggregaterGrad(object):
    """
        federated
    """

    def __init__(self, lr_glob=1.0):
        self.grad_agg = None
        self.num_coef = []
        self.lr_glob = lr_glob

    def increase(self, grad_local, coefficient=1):
        if self.grad_agg is None:
            self.grad_agg = copy.deepcopy(grad_local)
            for k in self.grad_agg.keys():
                self.grad_agg[k] = grad_local[k] * coefficient
        else:
            for k in self.grad_agg.keys():
                self.grad_agg[k] += grad_local[k] * coefficient
                
        self.num_coef.append(coefficient)

    def aggregate(self, weights_glob):
        for k in weights_glob.keys():
            # ! 这个sum是按数据集的比例加权
            self.grad_agg[k] = torch.div(self.grad_agg[k], sum(self.num_coef))
            #TODO 对BN并没有处理好，而且引发了新的问题，BN到底该怎么做聚合，还是直接不包含BN的统计信息。
            if 'num' not in k:
                weights_glob[k] += self.grad_agg[k] * self.lr_glob
        return weights_glob

    def clear(self):
        fprint("clear")
        self.grad_agg = None
        self.num_coef = []


    def t(self):
        # test code
        agger = AggregaterGrad(lr_glob=1.0)
        weight_glob = {'w1':10.0, 'w2':5.0}
        weight_local_1 = {'w1':2.0, 'w2':1.0}
        weight_local_2 = {'w1':5.0, 'w2':2.0}
        agger.increase(weight_local_1)
        agger.increase(weight_local_2)
        print(agger.grad_agg) # {'w1': 7.0, 'w2': 3.0}
        weight = agger.aggregate(weight_glob)
        print(weight) # {'w1': tensor(13.5000), 'w2': tensor(6.5000)}

        agger.clear()
        print(agger.grad_agg)
        print(agger.lr_glob)
        print(agger.num_coef)

def weight_norm(model, p=2):
    val = 0
    for n, w in model.named_parameters():
        val += torch.norm(w, p=p)
    return val


def check_client_class_distribution(targets, dict_users, run, args):
    """_summary_

    :param _type_ targets: _description_
    :param _type_ dict_users: _description_
    :param _type_ run: _description_
    :param _type_ args: _description_
    :return _type_: _description_
    """
    # 记录 各客户端：类别分布信息，类别熵信息，拥有类别数目

    client_data_name = [
        args.dataset,
        "{}{}".format(args.split, args.dir_alpha),
        'user{}'.format(args.num_users),
        'ifrac{}'.format(args.iid_frac),
        "s{}".format(args.seed)
    ]
    client_data_name = '_'.join(client_data_name)

    print("创建class_distribution: {}".format(client_data_name))
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    columns = ['user_id', 'data_num', 'classes_num', 'entropy', *map(str, list(set(targets)))]
    
    df = pd.DataFrame(columns=columns)
    results_list = []
    for user_id, idxs in dict_users.items():
        labels = []
        for i in idxs:
            labels.append(targets[i])
        df_user = pd.DataFrame(labels).value_counts()
        result = {'user_id': user_id, 'data_num': df_user.sum(), 'classes_num': len(df_user.index)}
        for key, value in df_user.items():
            result[str(key[0])] = value
        # df = df.append([results], ignore_index=True)
        results_list.append(result)
    df = pd.concat([df, pd.DataFrame.from_records(results_list)])
    df = df.fillna(0)
    for user_id in range(len(df)):
        p = df.iloc[user_id, 4:] / df.iloc[user_id, 1]
        df.iloc[user_id, 3] = (p * (- np.log2(p))).sum()
    return df

def get_optimizer(model, optim_name, lr, weight_decay=0, momentum=0):
    if optim_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

def compute_weight_quality(model_dict, old_model_dict=None):
    """
    for experiment exploration, compute statistical information.
    :param model_dict: model.state_dict()
    :return:
    """
    weight_quality = {}
    for k, v in model_dict.items():
        if 'float' in str(v.dtype):
            weight_quality['std_{}'.format(k)] = v.std().item()
            weight_quality['mean_{}'.format(k)] = v.mean().item()
            # weight_quality['max_{}'.format(k)] = v.max().item()
            # weight_quality['min_{}'.format(k)] = v.min().item()
            weight_quality['norm1_{}'.format(k)] = v.norm(p=1).item()
            if old_model_dict is not None:
                old_v = old_model_dict[k]
                grad = v - old_v
                weight_quality['g-std_{}'.format(k)] = grad.std().item()
                weight_quality['g-mean_{}'.format(k)] = grad.mean().item()
                weight_quality['g-norm1_{}'.format(k)] = grad.norm(p=1).item()
                weight_quality['g-ratio_{}'.format(k)] = old_v.norm(p='fro').item() / (grad.norm(p='fro').item() + 1e-8)
    return weight_quality



from algs import fedlocal, fedavg , fedrs, feddyn, moon, fedaws
if __name__ == '__main__':
    pass
    # --------- Aggregater Test ---------
    # from models.Nets import MLP
    # agger = Aggregater()
    # model = MLP(2,2,2)
    # agger.increase(model.state_dict())
    # agger.increase(model.state_dict())
    # p = agger.aggregate()
    # agger.clear()
    # print(p)

    # --------- ProcessLogger Test ---------
    # logger = ProcessLogger(use_wandb=False)
    # logger.record_train(1, {'loss_avg': 123})
    # logger.record_test(1, {'test_loss': 0.1, 'test_acc': 2})
    # print()


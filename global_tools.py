import random
import numpy as np
import logging
import torch
import os
import wandb

import pickle
import json
import yaml

import sys
import argparse
from itertools import chain, combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy.special import comb as comb_op


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_pkl(path: str, obj, protocol=4):
    """_summary_

    :param str path: _description_
    :param _type_ obj: _description_
    :param int protocol: pickle protocal, defaults to 4, bcz python3.8 HIGHT_PROTOCAL is 5 and python 3.6/3.7 is 4.
    """
    
    if '.pkl' not in path:
        path = path + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pkl(path: str):
    
    if '.pkl' not in path:
        path = path + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file:
        json.dump(data, file)

def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_yaml(file_path: str, data: dict):
    assert file_path.split('.')[-1] == 'yml'
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file)

def save_yaml_all(file_path: str, data: list):
    assert file_path.split('.')[-1] == 'yml'
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump_all(data, file)

def load_yaml_all(file_path: str):
    assert file_path.split('.')[-1] == 'yml'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        data = yaml.load_all(data, Loader=yaml.FullLoader)
    return data


def resolve_args(root: str, args, args_force={}, is_fed=True):
    """load default parameters from _param and exp self.

    :param str root: _description_
    :param _type_ args: _description_
    :param dict args_force: forcely set args except for [alg, dataset], defaults to {}
    :return _type_: _description_
    """

    args_params = vars(args)
    argv_names = obtain_argv_param_names()
    # general.yml 
    file_path = os.path.join(root, "exps/_params/general.yml")
    datas = list(load_yaml_all(file_path))[0]
    ignore_param_names = datas['ignore']
    default_params = datas['params']
    # dataset.yml
    file_path = os.path.join(root, "exps/_params/dataset.yml")
    datas = list(load_yaml_all(file_path))
    for data in datas:
        if args.dataset in data['dataset']:
            print("[dataset.yml] load {}".format(data['dataset']))
            for k,v in data['params'].items():
                default_params[k] = v
    # framework.yml
    file_path = os.path.join(root, "exps/_params/framework.yml")
    datas = list(load_yaml_all(file_path))
    for data in datas:
        if args.alg in data['framework']:
            print("[framework.yml] load {}".format(data['framework']))
            if data['params'] is not None:
                for k,v in data['params'].items():
                    default_params[k] = v
            
    # expx.x_xx/params.yml
    file_path = os.path.join(root, "exps/{}/params.yml".format(args.exp_name))
    if os.path.exists(file_path):
        datas = list(load_yaml_all(file_path))[0]
        exp_ignore_param_names = datas['ignore']
        for p in exp_ignore_param_names:
            if p not in ignore_param_names: ignore_param_names.append(p)
        exp_default_params = datas['params']
        for name, value in exp_default_params.items():
            default_params[name] = value

    special_param_names = []
    # --------------------------------------------------------------------------------------------------
    # according to assigned dataset and alg, acquire default param values for yml file. 
    # to 1) reset args_params and 2) choose special param names.
    for k in args_params.keys():
        if k in default_params:
            if k not in argv_names: # whether params emerged at program start.
                args_params[k] = default_params[k] # case 1.1: arg in default_params && arg not in argv -> set default params.
            else:
                special_param_names.append(k) # case 1.2: arg in default_params && arg in argv -> choose(want to show).
        else:
            if k not in special_param_names: # case 2: arg not in default_params (alg-special) -> choose(want to show).
                special_param_names.append(k)

    # forcely set args except for alg, dataset
    for k,v in args_params.items():
        if k in args_force:
            args_params[k] = v
    
    def abbr(words):
        """_summary_

        :param _type_ s: _description_
        :return _type_: _description_
        """
        split = words.split('_')
        if len(split) == 1:
            name = words[0]
        elif len(split) == 2:
            name = split[0][0] + split[1].capitalize()[:2]
        else:
            print(words)
            raise Exception('args name should not have > 1 _')
        return name

    # ---------------------------------------------------------------------------------------------------
    # after reset arg params, 
    setting_name = []
    if is_fed:
        setting_param_names = ['dataset', 'data_frac', 'num_users','split', 'dir_alpha','frac' ,'clsnum_peruser','epochs','local_ep', 'seed']
        for k in setting_param_names:
            if args_params['split'] == 'dir' and k in ['dir_alpha','clsnum_peruser']:
                continue
            value = args_params[k]
            name = '' if isinstance(args_params[k], str) else abbr(k)
            if value == 'dir':
                value += "%.2f" % args_params['dir_alpha']
            setting_name.append("{}{}".format(name, value))
    else:
        setting_param_names = ['dataset', 'epochs', 'seed']
        for k in setting_param_names:
            value = args_params[k]
            name = '' if isinstance(args_params[k], str) else abbr(k)
            setting_name.append("{}{}".format(name, value))
    print("[setting folder]: ", setting_param_names)
    setting_name = '_'.join(setting_name)

    args = argparse.Namespace(**args_params)
    method_name = []
    for k in special_param_names:
        value = args_params[k]
        if value == "" or k in ignore_param_names or k in setting_param_names:
            continue
        # if type(arg's value) is 'str', name =''. else we abbreviate it.
        name = '' if isinstance(value, str) else abbr(k)
        method_name.append("{}{}".format(name, value))
    method_name = '_'.join(method_name)

    return args, setting_name, method_name


def append_in_dict(origin_dict:dict, new_dict:dict):
    for k, v in new_dict.items():
        if k not in origin_dict:
            origin_dict[k] = []
        origin_dict[k].append(v)
    return origin_dict

def mean_in_dict(data:dict):
    mean_data = {}
    for k, v in data.items():
        mean_data[k] = sum(v) / len(v)
    return mean_data


def obtain_argv_param_names():
    argvs = sys.argv
    param_names = []
    for i in argvs:
        if '--' in i:
            param_names.append(i.replace('--',''))
    return param_names



def powerset(s:list):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def draw_acc_powersets(test_acc_results, num_users, epochs, name="", dpi=100):

    combine_len = []
    for i,j in enumerate(list(powerset(list(range(num_users))))):
        if len(j) == len(combine_len) + 1:
            combine_len.append(i)

    fig = plt.figure(dpi=dpi)
    ax = fig.gca()

    for k,v in test_acc_results.items():
        plt.plot(v, label=str(k))
    plt.legend()
    xtick = np.arange(0, 1025, 128)
    plt.xticks(xtick)
    plt.xlabel('powerset of users')
    plt.ylabel('test_acc ')

    xminorLocator = FixedLocator(combine_len)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.grid(True, which='minor')
    plt.title("test acc of powersets with %d epochs between different collection rounds %s" % (epochs, name))

def draw_sv_users(sv_results, num_users, epochs, dpi=100, name='SV'):
    plt.figure(dpi=dpi)
    total_width, n = 0.8, len(sv_results)
    width = total_width / n
    x = np.arange(num_users)
    x = x - (total_width - width) / 2

    now_width = 0
    for k,v in sv_results.items():
        # plt.plot(v, label=str(k))
        plt.bar(x + now_width,v,width=width, label=str(k))
        now_width += width
    plt.legend(loc="upper right",bbox_to_anchor=(1.2,1))
    plt.title('%s with %d epochs between different collect rounds' % (name, epochs))
    plt.xticks(np.arange(num_users))
    plt.xlabel('User id')
    plt.ylabel('Contribution')

def compute_shapley_value(num_users, test_acc):
    combinations_list = list(powerset(list(range(num_users))))
    combine_2_idx = {v:k for k,v in enumerate(combinations_list)}
    results = np.zeros(num_users)
    for user_id in range(num_users):
        ex_user_combinations = [_ for _ in combinations_list if user_id not in _]
            # ex_user_combinations = [_ for _ in combinations_list if user_id not in _ and len(set(_).difference(set(range(2)))) == 0]
        for ex_user in ex_user_combinations:
            ex_user_add = ex_user + (user_id,)
            ex_user_add = tuple(sorted(list(ex_user_add)))
            # print(ex_user_add, ex_user)
            # print(test_acc[combine_2_idx[ex_user_add],1], '-', test_acc[combine_2_idx[ex_user],1])
            results[user_id] += (test_acc[combine_2_idx[ex_user_add]] - test_acc[combine_2_idx[ex_user]]) / comb_op(num_users-1, len(ex_user))
    return results


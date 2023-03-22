import numpy as np
from torchvision import datasets,transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import pickle

import copy
import math
import random
import argparse
import numpy as np
import torch
import sys
sys.path.append('/data/xuyc/projects/fedvar_inc/')
from global_tools import set_random_seed, save_pkl, load_pkl, powerset
from models.test import test_center
from algs.alg_tools import get_models_array, load_model_from_params
import socket

hostname = socket.gethostname()
if hostname.startswith('xuyc-Server'):
    root_path = '/home/xuyc/datasets/'
elif hostname.startswith('amax'):
    root_path = '/data/xuyc/datasets/'


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help="GPU ID, -1 for CPU")
parser.add_argument('--num', type=int, default=1, help='epoch num')
parser.add_argument('--exp_name', type=str, default="exp1.3_try_emnist/emnist_nUs10_niid-label_dAl0.5_f1.0_cPe2_e3_lEp1_s1", help='project name')
parser.add_argument('--run_name', type=str, default="", help='project name')
arg_1 = parser.parse_args()

num = arg_1.num
exp_name = arg_1.exp_name
run_name = arg_1.run_name
os.environ['CUDA_VISIBLE_DEVICES'] = arg_1.gpu


root = f'/data/xuyc/save/FedVar/{exp_name}/'

path = root + f'{run_name}/nr1/cid-1/models_{num}.npy'
params_array = np.load(path)

from utils.train_utils import get_fed_data, get_model
args_path = root + f'{run_name}/nr1/cid-1/results.pkl'
args = load_pkl(args_path)['args']
setting_dir= root
set_random_seed(seed=args.seed)
# load dataset
dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_data(setting_dir, args.dataset, args.split, args.num_users, args.dir_alpha, args.clsnum_peruser, args.iid_frac, args.imb_alpha, 1.0)
loader_global_test = DataLoader(dataset_test, batch_size=200, num_workers=5,
                                             shuffle=False)
# build model
model = get_model(args.dataset, args.model, norm=args.norm, num_groups=args.num_groups)

# run
num_users = 10
combinations_list = list(powerset(list(range(num_users))))
comb_results = {}
for comb in combinations_list:
# for comb in [combinations_list[-1]]:
    if  len(comb) >= 1:
        model = load_model_from_params(model, params_array[list(comb)].reshape(len(list(comb)),-1).mean(axis=0))
        global_test_results = test_center(model, loader_global_test, global_test=True)
        comb_results[comb] = global_test_results
        print(num, global_test_results)
    else:
        model = load_model_from_params(model, params_array[-1].reshape(1,-1).mean(axis=0))
        global_test_results = test_center(model, loader_global_test, global_test=True)
        comb_results[comb] = global_test_results
        print(num, global_test_results)

save_dir = root + f'{run_name}/nr1/perf'
os.makedirs(save_dir, exist_ok=True)
save_pkl(os.path.join(save_dir,"%d.pkl" % num), comb_results)

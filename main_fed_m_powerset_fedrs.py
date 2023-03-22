#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import os
import re
import numpy as np

import sys
from utils.options import get_basic_parser, add_fl_parser
from utils.train_utils import get_fed_data, get_model, get_weight_initial_std, split_collection

from global_tools import set_random_seed, resolve_args, save_yaml, powerset, save_pkl
from utils.log_tools import generate_log_dir, CatchExcept
from algs.alg_tools import get_fed_algo, load_model_from_params
from algs.alg_tools import get_fed_algo, load_model_from_params
from code_tools import backup_code
import time
# experiment log

if __name__ == '__main__':
    with CatchExcept():
        # parse args
        parser = add_fl_parser(get_basic_parser())
        parser.add_argument('--now_round', type=int, default=-1, help="collection round")
        parser.add_argument('--collect_rounds', type=int, default=5, help="collection rounds")
        parser.add_argument('--reinit', type=int, default=0, help="whether reinit the global model in each collection round")
        parser.add_argument('--combine_id', type=int, default=-1, help="choose which combination of users in powerset [1-2^n]] or use all user by -1")
        parser.add_argument('--tru_batch', type=int, default=-1, help="which batch truncation for fast learning")

        parser.add_argument('--rs_alpha', type=float, default=0.5, help="the alpha of fedrs")
        
        ## set speical args for special methods.
        args = parser.parse_args()
        # ---------------------------------------------------------------
        # Load parameters
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # random seed
        set_random_seed(seed=args.seed)
        args, setting_name, method_name = resolve_args(root='.', args=args)
        print(setting_name, method_name)
        base_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name, method_name)
        args.base_dir = generate_log_dir(path=base_dir, is_use_tb=False, has_timestamp=args.timestamp)
        # # basicly save args and running code.
        print('base_dir: ', args.base_dir)

        if not os.path.exists(args.base_dir + '/args.yml'):
            save_yaml(file_path=args.base_dir + '/args.yml', data=vars(args))
            code_dir = os.path.join(args.base_dir, 'code')
            backup_code(src_dir=os.path.dirname(os.path.abspath(__file__)), to_dir=code_dir)

        # # ---------------------------------------------------------------
        # # load dataset
        setting_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name)
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_data(setting_dir, args.dataset, args.split, args.num_users, args.dir_alpha, args.clsnum_peruser, args.iid_frac, args.imb_alpha, args.data_frac)

        # build model
        global_model = get_model(args.dataset, args.model, norm=args.norm, num_groups=args.num_groups)

        if args.load != "":
            load_path = os.path.join(args.base_dir.replace(f"_e{args.epochs}_", f"_e{args.load}_"), f"nr1/cid-1/models_{args.load}.npy")
            params = np.load(load_path)
            global_model = load_model_from_params(global_model, params[-1])
            start_epoch = eval(args.load)
            print(f"[load pretrained model] from {load_path}")
        else:
            start_epoch = 1

        Algorithm = get_fed_algo(args.alg)
        if args.collect_rounds == 1:
            dict_users_train_rounds = [dict_users_train]
        else:
            dict_users_train_rounds = split_collection(dataset_train.targets, dict_users_train, collection_rounds=args.collect_rounds)

        dict_save_path = os.path.join(args.base_dir, 'dict_users_cr%d.pkl' % (args.collect_rounds))
        save_pkl(dict_save_path, (dict_users_train_rounds, dict_users_train, dict_users_test))

        if args.now_round == -1:
            round_list = range(1, args.collect_rounds+1)
        else:
            round_list = [args.now_round]
        
        idxs_all_users = list(range(args.num_users))
        if args.combine_id > 0:
            combination_list = list(powerset(idxs_all_users))
            idxs_all_users = list(combination_list[args.combine_id-1])

        for now_round in round_list:
            if now_round != -1:
                args.base_dir = generate_log_dir(path=base_dir, ind_sim="nr%d/cid%d" % (now_round, args.combine_id))
            else:
                args.base_dir = generate_log_dir(path=base_dir, ind_sim="nr%d" % (now_round))
                
            framework = Algorithm(dataset_train, dataset_test, dict_users_train_rounds[now_round-1], dict_users_test, global_model, args, idxs_all_users=idxs_all_users)
            framework.set_start_epoch(start_epoch)
            framework.train()
            if args.reinit == 1:
                # build model
                global_model = get_model(args.dataset, args.model, norm=args.norm, num_groups=args.num_groups)
            else:
                global_model = framework.model
            framework.save_results("results")

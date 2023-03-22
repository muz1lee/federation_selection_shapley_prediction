#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch

from utils.options import get_basic_parser
from utils.train_utils import get_center_data, get_model, get_weight_initial_std
import os
import math
from global_tools import fprint, set_random_seed, initial_wandb, load_exp_params, load_pkl, CatchExcept
from code_tools import backup_code
from algs.center_learning import CenterLearning
# experiment log

if __name__ == '__main__':
    with CatchExcept() as _:
        # parse args
        parser = get_basic_parser()
        parser.add_argument('--nid', type=str, default="", help="[NAME]-[INDEX], e.g., a-1.0")
        parser.add_argument('--lamda', type=float, default=0, help="control var loss")
        parser.add_argument('--lamda_decay', type=float, default=0, help="lamda decay")
        parser.add_argument('--head_var', type=float, default=0, help="variance by beating head")
        parser.add_argument('--varloss_type', type=str, default="norm", help="norm, relu")
        parser.add_argument('--var_type', type=str, default="fix", help="init, fix, inifix")
        args = parser.parse_args()
        assert args.alg == "center"

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args, part_hyper_params = load_exp_params(root='.', args=args)
        print("[Default args]: {}".format(args))
        # random seed
        set_random_seed(seed=args.seed)

        run, run_name = initial_wandb(project_name=args.project_name, hyper_params=part_hyper_params, args=args)

        # ---------------------------------------------------------------
        # load dataset
        dataset = get_center_data(args)

        if args.dataset == "cifar100":
            cls_num = dataset['train'].cls_num
        elif args.dataset in ["cifar10", "mnist"]:
            cls_num = 10 #TODO not processed.
        # build model
        model = get_model(args.dataset, args.model, cls_num, norm=args.norm, num_groups=args.num_groups)
        if args.load != "":
            state = torch.load(args.load)
            model.load_state_dict(state['model'])
            start_epoch = state['epoch']
        else:
            start_epoch = 1

        if args.var_type == 'init':
            head_var = {}
            for name, param in model.named_parameters():
                if "weight" in name and "bn" not in name:
                    var = get_weight_initial_std(param)
                    head_var[name] = var
        elif args.var_type == 'fix':
            head_var = args.head_var
        elif args.var_type == 'inifix':
            head_var = {}
            for name, param in model.named_parameters():
                if "weight" in name and "bn" not in name:
                    var = get_weight_initial_std(param)
                    head_var[name] = var
            values = head_var.values()
            max_var, min_var = max(values), min(values)
            for k, v in head_var.items():
                head_var[k] = (v- min_var) / (max_var - min_var + 1e-7) * args.head_var

        Algorithm = CenterLearning

        # save_yaml(file_path=args.base_dir + '/args.yml', data=vars(args))
        code_dir = os.path.join(args.base_dir, 'code')
        backup_code(src_dir=os.path.abspath(__file__), to_dir=code_dir)

        framework = Algorithm(dataset, model, args, head_var)
        framework.set_start_epoch(start_epoch)
        framework.train()
        framework.save_results("results")

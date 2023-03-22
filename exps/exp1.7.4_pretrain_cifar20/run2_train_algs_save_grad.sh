#!/bin/bash

# lr 0.01 for 10 clients
split_list=("niid-label --clsnum_peruser 2" "dir --dir_alpha 10.0")

# fedrs
default_command='python main_fed_m_powerset_fedrs.py --alg fedrs --dataset cifar20 --model cnn --exp_name exp1.7.4_pretrain_cifar20 --lr 0.05 --combine_id -1 --rs_alpha [rs_alpha] --split [split] --epochs 20 --save_clients'

rs_list=(0.6 0.9)
for rs_alpha in "${rs_list[@]}"; do
    for split in "${split_list[@]}"; do
        c=$default_command
        c=${c//\[rs_alpha\]/$rs_alpha}
        c=${c//\[split\]/$split}
        echo $c
    done
done

# feddyn
default_command='python main_fed_m_powerset_feddyn.py --alg feddyn --dataset cifar20 --model cnn --exp_name exp1.7.4_pretrain_cifar20  --lr 0.05 --combine_id -1 --dyn_alpha [dyn_alpha] --split [split] --epochs 20 --save_clients'
dyn_list=(0.01 0.001)
for dyn_alpha in "${dyn_list[@]}"; do
    for split in "${split_list[@]}"; do
        c=$default_command
        c=${c//\[dyn_alpha\]/$dyn_alpha}
        c=${c//\[split\]/$split}
        echo $c
    done
done

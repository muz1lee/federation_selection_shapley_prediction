#!/bin/bash

# lr 0.01 for 10 clients
split_list=("niid-label --clsnum_peruser 2" "dir --dir_alpha 10.0")
exp_dir=$(basename $(dirname $0))

# # Stage 1 
# # cifar20
# default_command="python main_fed_m_powerset.py --alg fedavg --dataset cifar20 --model cnn --exp_name $exp_dir --lr 0.01 --combine_id -1 --split [split] --epochs 20 --save_clients"

# for split in "${split_list[@]}"; do
#     c=$default_command
#     c=${c//\[split\]/$split}
#     echo $c
# done


# # Continue training more 20 rounds
default_command="python main_fed_m_powerset.py --alg fedavg --dataset cifar20 --model cnn --exp_name $exp_dir --lr 0.01 --combine_id -1 --split [split] --epochs 40 --save_clients --load 20"

for split in "${split_list[@]}"; do
    c=$default_command
    c=${c//\[split\]/$split}
    echo $c
done
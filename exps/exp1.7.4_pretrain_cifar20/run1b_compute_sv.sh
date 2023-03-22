#!/bin/bash

default_command='python perf.py --num [num] --exp_name [exp_name] --run_name [run_name]'
exp_dir=$(basename $(dirname $(realpath $0)))

# # Stage 1
# exp_list=(cifar20_dFr1.0_nUs10_dir10.00_f1.0_e20_lEp1_s1 cifar20_dFr1.0_nUs10_niid-label_dAl10.0_f1.0_cPe2_e20_lEp1_s1)
# run_list=(fedavg_cnn_l0.01_r0_tBa-1)
# # num=20
# for num in $(seq 12 18); do
#     for run_name in "${run_list[@]}"; do
#         for exp_name in "${exp_list[@]}"; do
#             c=$default_command
#             c=${c//\[num\]/$num}
#             c=${c//\[exp_name\]/$exp_dir/$exp_name}
#             c=${c//\[run_name\]/$run_name}
#             echo $c
#         done
#     done
# done

# Continue 21-40 epochs
exp_list=(cifar20_dFr1.0_nUs10_dir10.00_f1.0_e40_lEp1_s1 cifar20_dFr1.0_nUs10_niid-label_dAl10.0_f1.0_cPe2_e40_lEp1_s1)
run_list=(fedavg_cnn_l0.01_r0_tBa-1)
for num in $(seq 21 40); do
    for run_name in "${run_list[@]}"; do
        for exp_name in "${exp_list[@]}"; do
            c=$default_command
            c=${c//\[num\]/$num}
            c=${c//\[exp_name\]/$exp_dir/$exp_name}
            c=${c//\[run_name\]/$run_name}
            echo $c
        done
    done
done
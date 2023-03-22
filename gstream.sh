#!/bin/bash

# bash t.sh | xargs -0 bash gstream.sh

get_available_gpu() {
    want_space=$1
    x=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits))
    num_gpu=$((${#x[@]} / 2))
    threshold=6000

    max_id=-1
    max_free_space=-1

    for ((i=0;i<num_gpu;i++)); do
        id_gpu=${x[2*i]//,/}
        free_space=${x[2*i+1]}
        # echo "${id_gpu} ${free_space}"
        if ((free_space>max_free_space)); then    
            max_id=$id_gpu
            max_free_space=$free_space
        fi
    done

    if ((want_space+threshold<max_free_space)); then
        free_block_num=$(((max_free_space-threshold)/want_space))
        if ((free_block_num>3)); then
            free_block_num=3
        fi
        echo $max_id $free_block_num
        return 0
    fi
    echo -1 0
    return 255
}

start_time=$(date +%y/%m/%d-%T)
want_space=2000 #default want space
wait_time=30s   #default wait to ready want space.
arg2=$2
arg3=$3

if [[ -n $arg2 ]]; then
    want_space=$arg2
fi
if [[ -n $arg3 ]]; then
    wait_time=$arg3
fi

echo Default want_space $want_space wait_time $wait_time


# acquire available gpu for the first run
id_gpu=-1
cnt_try=0
while ((id_gpu == -1)); do
    results=($(get_available_gpu ${want_space}))
    id_gpu=${results[0]}
    num_tickets=${results[1]}
    if ((id_gpu == -1)); then
        cnt_try=$((cnt_try+1))
        echo apply $cnt_try times
        sleep 5s
    fi
done
echo Default Run id_gpu $id_gpu num_tickets $num_tickets want_space $want_space
num_tickets=1

pid_array=() # store process' ids
is_set_want_space=0

# conduct real tasks.
tasks=$(bash "$1")
num_tasks=$(echo "${tasks}" | wc -l)
echo num of tasks = $num_tasks
while read line; do
    if [[ -n $line ]]; then
        # $("nohup $line --gpu ${id_gpu} & pid=$!")
        cnt=$((${#pid_array[@]}+1))
        nohup $line --gpu ${id_gpu} >/dev/null 2>&1 & pid=$!
        echo "[$cnt|$num_tasks]: $(date +%y/%m/%d-%T) nohup $line --gpu ${id_gpu} pid $pid"
        pid_array+=($pid)
        # echo ${pid_array[@]}
        
        num_tickets=$((num_tickets-1))
        if ((num_tickets > 0 && is_set_want_space == 1)); then
            continue
        fi
        sleep $wait_time
        if [[ $is_set_want_space == 0 ]]; then
            temp_want_space=$(nvidia-smi | grep ${pid_array[$cnt-1]} | grep -Eo "[0-9]+M")
            if [[ -n $temp_want_space ]]; then
                want_space=${temp_want_space//M/}
                # want_space=$((want_space))
                echo [set want space]=${want_space} is_set_want_space ${is_set_want_space}
                is_set_want_space=1
            fi
        fi
        
        id_gpu=-1
        while ((id_gpu == -1)); do
            results=($(get_available_gpu ${want_space}))
            id_gpu=${results[0]}
            num_tickets=${results[1]}
            echo id_gpu $id_gpu num_tickets $num_tickets want_space $want_space
            if ((id_gpu == -1)); then
                sleep 1m
            fi
        done
    fi
done <<< "$tasks"


echo start time $start_time
# phase: wait for all of runs finished
finished_pid_array=()
# echo ${#finished_pid_array[@]} ${#pid_array[@]} ${pid_array[@]}
while [[ ${#finished_pid_array[@]} < ${#pid_array[@]} ]]; do
    for pid in ${pid_array[@]}; do
        # if pid is not finished
        if [[ ! "${finished_pid_array[@]}" =~ "${pid}" ]]; then
            temp_want_space=$(nvidia-smi | grep ${pid} | grep -Eo "[0-9]+M")
            # if in nvidia-smi, the result is "", add to finished_pid_array 
            if [[ $temp_want_space ]]; then
                finished_pid_array[${#finished_pid_array[@]}]=$pid
                echo $(date +%y/%m/%d-%T) ${#finished_pid_array[@]}/${#pid_array[@]} is finished
            fi
        fi
    done
    if [[ ${#finished_pid_array[@]} == ${#pid_array[@]} ]]; then
        break
    fi
    sleep 1s
done

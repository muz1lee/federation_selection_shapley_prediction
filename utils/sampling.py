#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
from itertools import permutations
import numpy as np
import pandas as pd
import torch
import pdb
from utils.log_tools import fprint
from collections import Counter


def fair_iid(dataset, num_users):
    """
    Sample I.I.D. client data from fairness dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def fair_noniid(train_data, num_users, num_shards=200, num_imgs=300, train=True, rand_set_all=[]):
    """
    Sample non-I.I.D client data from fairness dataset
    :param dataset:
    :param num_users:
    :return:
    """
    assert num_shards % num_users == 0
    shard_per_user = int(num_shards / num_users)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    labels = train_data[1].numpy().reshape(len(train_data[0]), )
    assert num_shards * num_imgs == len(labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    if len(rand_set_all) == 0:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
            for rand in rand_set:
                rand_set_all.append(rand)

            idx_shard = list(set(idx_shard) - rand_set)  # remove shards from possible choices for other users
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    else:  # this only works if the train and test set have the same distribution of labels
        for i in range(num_users):
            rand_set = rand_set_all[i * shard_per_user: (i + 1) * shard_per_user]
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users, rand_set_all


def iid(dataset, num_users, data_frac=1.0):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)
    all_idxs = np.array([i for i in range(len(targets))])
    n_classes = len(set(targets))
    dict_users = {i:[] for i in range(num_users)}

    for c in range(n_classes):
        num_items_per_cls = math.floor(len(targets[targets==c]) / num_users)
        for i in range(num_users):
            ids_cls = all_idxs[np.where(targets[all_idxs]==c)[0]]
            ids_select_cls = np.random.choice(ids_cls, num_items_per_cls, replace=False)
            ids_select_cls = ids_select_cls.tolist()
            dict_users[i] += ids_select_cls
            all_idxs = np.array(list(set(all_idxs) - set(ids_select_cls)))
    for i in dict_users.keys():
        random.shuffle(dict_users[i])

    for i in dict_users.keys():
        data_num = int(len(dict_users[i]) * data_frac)
        dict_users[i] = dict_users[i][:data_num]
    return dict_users

def noniid_label(dataset, num_users, shard_per_user, data_frac=1.0, rand_set_all=[], is_uniform=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param shard_per_user: the number of classes assigned to every user, e.g. 2=each user only have samples from 2 classes.
    :return dict_users: dict{'userid': data}
    :return rand_set_all: list(each user' classes)
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    #
    idxs_dict = {}  # {'label': 'dataset samples' id'}
    targets_tensor = torch.tensor(dataset.targets)
    for i in range(len(dataset)):
        label = targets_tensor[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    for i in idxs_dict.keys():
        data_num = int(len(idxs_dict[i]) * data_frac)
        idxs_dict[i] = idxs_dict[i][:data_num]

    num_classes = len(targets_tensor.unique())
    assert shard_per_user <= num_classes, "The number of classes assigned to each user must <= num_classes"
    shard_per_class = int(shard_per_user * num_users / num_classes)
    # reshape the array of each label to (shard_pre_class, *)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        """
            data of each class need to split into shared_per_class blocks.
            According to data's num and block's num of the class, identify the num of each block of the class.
            If we have more data which can't be modded, put them into former blocks one by one.
        """
        num_leftover = len(x) % shard_per_class
        # get indexes of [len-num_leftover, ..., len]
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x
    # -- assign each user's classes.
    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        if is_uniform is False:
            random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # -- divide and assign
    # give each user his corresponding class data according to rand_set_label.
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))  # pop‘s function is to delete the idx'th element and return it.
        dict_users[i] = np.concatenate(rand_set)

    # -- check whether the assignment results are resonable.
    test = []
    for key, value in dict_users.items():
        x = targets_tensor[value].unique()
        assert (len(x) <= shard_per_user)
        test.append(value)
    test = np.concatenate(test)
    # assert (len(test) == len(dataset))
    # assert (len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all

def noniid_step(targets, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param shard_per_user: the number of classes assigned to every user, e.g. 2=each user only have samples from 2 classes.
    :return dict_users: dict{'userid': data}
    :return rand_set_all: list(each user' classes)
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    #
    idxs_dict = {}  # {'label': 'dataset samples' id'}
    if type(targets) == torch.Tensor:
        targets_array = targets.numpy()
    elif type(targets) == list:
        targets_array = np.array(targets)
    else:
        targets_array = targets
    
    n_classes = len(np.unique(targets_array))
    shard_per_class = int(shard_per_user * num_users / n_classes)
    
    for i in range(len(targets_array)):
        label = targets_array[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    for k in idxs_dict.keys():
        random.shuffle(idxs_dict[k])
        n_samples_1_block = int(len(idxs_dict[k]) / 4)
        if 0<=k<3:
            block_num = 1
        elif 3<=k<8:
            block_num = 2
        elif k==8:
            block_num = 3
        elif k==9:
            block_num = 4
        idxs_dict[k] = list(np.array(idxs_dict[k][:block_num * n_samples_1_block]).reshape(block_num,n_samples_1_block))

    if len(rand_set_all) == 0:
        rand_set_all = np.zeros((num_users,shard_per_class))
        block_pool = np.array([1,1,1,2,2,2,2,2,3,4])
        for k in range(num_users):
            p = block_pool / block_pool.sum()
            select_classes = np.random.choice(list(range(10)), 2, replace=False, p=p)
            for c in select_classes:
                block_pool[c] -= 1
            rand_set_all[k] = select_classes

    # -- divide and assign
    # give each user his corresponding class data according to rand_set_label.
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))  # pop‘s function is to delete the idx'th element and return it.
        dict_users[i] = np.concatenate(rand_set)
    
    return dict_users, rand_set_all

def get_frequncey_classes(x, n_classes):
    frequency_classes = Counter(x)
    frequency_classes = np.array([frequency_classes[i] if i in frequency_classes else 0.0 for i in range(n_classes)], dtype='float')
    frequency_classes /= frequency_classes.sum()
    return frequency_classes

def noniid_dir(targets, num_users, data_frac=1.0, alpha=1.0):
    """
    Sample non-I.I.D client data by dirichlet distribution from dataset 
    :param targets: type{list|np.array|torch.tensor}
    :param num_users:
    :param shard_per_user: the number of every user's data blocks.
    :return dict_users: dict{'userid': data}
    :return rand_set_all: list(each user' classes)
    """
    dict_users = {i:[] for i in range(num_users)}
    if type(targets) == torch.Tensor:
        targets_array = targets.numpy()
    elif type(targets) == list:
        targets_array = np.array(targets)
    else:
        targets_array = targets
    n_classes = len(np.unique(targets_array))
    n_samples = targets_array.shape[0]
    n_samples_per_user = n_samples // num_users
    ids_assigned_samples = []
    frequency_classes = get_frequncey_classes(targets_array, n_classes)

    for u in range(num_users):
        prob_samples = torch.zeros(n_samples)
        dist = np.random.dirichlet(alpha * frequency_classes)
        for c in range(n_classes):
            ids_c = np.where(targets_array==c)[0]
            prob_samples[ids_c] = dist[c]
        prob_samples[ids_assigned_samples] = 0.0
        dict_users[u] = (torch.multinomial(prob_samples, n_samples_per_user, replacement=False)).tolist()
        random.shuffle(dict_users[u])
        ids_assigned_samples += dict_users[u]
    
    dict_frequency_classes = {}
    for u in range(num_users):
        dict_frequency_classes[u] = get_frequncey_classes(targets_array[dict_users[u]], n_classes)

    for i in dict_users.keys():
        data_num = int(len(dict_users[i]) * data_frac)
        dict_users[i] = dict_users[i][:data_num]

    return dict_users, dict_frequency_classes

def imitate_sampling(targets, num_users, dict_frequency_classes):
    dict_users = {i:[] for i in range(num_users)}
    if type(targets) == torch.Tensor:
        targets_array = targets.numpy()
    elif type(targets) == list:
        targets_array = np.array(targets)
    else:
        targets_array = targets
    n_classes = len(np.unique(targets_array))
    n_samples = targets_array.shape[0]
    n_samples_per_user = n_samples // num_users
    ids_assigned_samples = []

    for u in range(num_users):
        dist = dict_frequency_classes[u]
        for c in range(n_classes):
            prob_samples = torch.zeros(n_samples)
            ids_c = np.where(targets_array==c)[0]
            prob_samples[ids_c] = 1.0
            prob_samples[ids_assigned_samples] = 0.0
            n_samples_class = int(dist[c] * n_samples_per_user)
            try:
                if n_samples_class > 0:
                    dict_users[u] += (torch.multinomial(prob_samples, n_samples_class, replacement=False)).tolist()
            except Exception:
                 if prob_samples[ids_c].sum() == 0:
                    raise Exception('all samples in ids_c has been sampled, you should change the random seed. May be caused by `n_samples_class = int(dist[c] * n_samples_per_user)`')
        random.shuffle(dict_users[u])
        ids_assigned_samples += dict_users[u]
    
    return dict_users

def noniid_replace(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param shard_per_user: the classes num assigned to every user.
    :return:
    """
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # -- achieve idxs_dict{'label': 'data'}
    idxs_dict = {}
    targets_tensor = torch.tensor(dataset.targets)
    for i in range(len(dataset)):
        label = targets_tensor[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    # -- sample classes for each user with replacement.
    num_classes = len(targets_tensor.unique())
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # -- divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert (len(targets_tensor[value].unique())) == shard_per_user

    return dict_users, rand_set_all


def noniid_dir_mix_iid(dataset, num_users, iid_frac, rand_set_all=None, alpha=1.0):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param iid_frac: the fraction of iid users in the all of users.
    :param shard_per_user: the number of every user's data blocks.
    :return dict_users: dict{'userid': data}
    :return rand_set_all: list(each user' classes)
    """
    dict_users = {}
    idxs_dict = {}  # {'label': 'dataset samples' id'}
    targets_tensor = torch.tensor(dataset.targets)
    for i in range(len(targets_tensor)):
        label = targets_tensor[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    for vals in idxs_dict.values():
        random.shuffle(vals)

    num_classes = len(targets_tensor.unique())
    num_iid_users = int(iid_frac * num_users)

    if num_iid_users > 0:
        num_data_each_class = {}
        for j in idxs_dict.keys():
            num_data_each_class[j] = int(len(idxs_dict[j]) * iid_frac) // num_iid_users

        for i in range(num_iid_users):
            dict_users[i] = []
            for j in idxs_dict.keys():
                select_ids = np.random.choice(idxs_dict[j], num_data_each_class[j], replace=False)
                idxs_dict[j] = list(set(idxs_dict[j]).difference(set(select_ids)))
                dict_users[i].extend(select_ids)

    num_non_iid_users = num_users - num_iid_users
    if num_non_iid_users > 0:
        num_in_classes = int(len(targets_tensor) * (1 - iid_frac)) // num_classes

        if rand_set_all is None:
            sample_labels = np.random.dirichlet([alpha / num_classes] * num_classes, num_non_iid_users)

            # 归一化保证每个类别维度的概率和为1.
            sample_labels_div = sample_labels / sample_labels.sum(axis=0)
            # 真实样本个数计算, * 前者的每一行的元素会对应乘上后者的元素。
            rand_set_all = sample_labels_div.copy()

        sample_labels_div = rand_set_all

        sample_labels_div = sample_labels_div * num_in_classes
        # each client removes small classes.
        # sample_labels_div[sample_labels_div < sample_labels_div.max(axis=1, keepdims=True) * 0.1] = 0
        sample_labels_div = np.floor(sample_labels_div)
        sample_labels = np.cumsum(sample_labels_div, axis=0).astype(np.int32)

        for user_index in range(0, num_non_iid_users):
            rand_set = []
            for cls_index, cls_set in idxs_dict.items():
                if user_index == 0:
                    tail = sample_labels[user_index, cls_index]

                    rand_set.extend(cls_set[:tail])
                else:
                    head = sample_labels[user_index - 1, cls_index]
                    tail = sample_labels[user_index, cls_index]
                    rand_set.extend(cls_set[head:tail])
            dict_users[user_index + num_iid_users] = rand_set
            random.shuffle(dict_users[user_index + num_iid_users])
    fprint("iid users: {}, non-iid users: {}".format(num_iid_users, num_non_iid_users))
    return dict_users, rand_set_all

def imbalanced_iid(dataset, num_users, alpha=1.0):
    """

    :param dataset:
    :param num_users:
    :param alpha: control the imbalanced level，recommend alpha=[0.05,0.2,0.5], attention: here is different from the alpha of dirichlet sampling in non-IID
    :return:
    """

    dist = np.arange(0, num_users) * alpha + 1
    dist = dist / dist.sum()

    dict_users = {}
    idxs_dict = {}  # {'label': 'dataset samples' id'}
    targets_tensor = torch.tensor(dataset.targets)
    for i in range(len(targets_tensor)):
        label = targets_tensor[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_data_each_class = {}
    for j in idxs_dict.keys():
        num_data_each_class[j] = int(len(idxs_dict[j]))

    for i in range(num_users):
        dict_users[i] = []
        for j in idxs_dict.keys():
            select_ids = np.random.choice(idxs_dict[j], int(num_data_each_class[j] * dist[i]), replace=False)
            idxs_dict[j] = list(set(idxs_dict[j]).difference(set(select_ids)))
            dict_users[i].extend(select_ids)

    return dict_users

def noniid_dir_same_num(dataset, num_users, alpha=1.0):
    """
    Sample non-I.I.D client data from dataset, each client has same number of data.
    :param dataset:
    :param num_users:
    :param shard_per_user: the number of every user's data blocks.
    :return dict_users: dict{'userid': data}
    """


    dict_users = {}
    idxs_dict = {}  # {'label': 'dataset samples' id'}
    targets_tensor = torch.tensor(dataset.targets)
    for i in range(len(targets_tensor)):
        label = targets_tensor[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    # 根据排序后的key值，统计每个类别数据个数。
    idxs_dict_keys = sorted(list(idxs_dict.keys()))
    num_in_classes = []
    for key in idxs_dict_keys:
        random.shuffle(idxs_dict[key])
        num_in_classes.append(len(idxs_dict[key]))
    num_in_classes = np.array(num_in_classes).reshape(1, -1)

    # 迪利克雷分布分布采样
    num_classes = len(targets_tensor.unique())
    sample_labels = np.random.dirichlet([alpha / num_classes] * num_classes, num_users)
    # 归一化保证每个类别维度的概率和为1.
    sample_labels_div = sample_labels / sample_labels.sum(axis=0)
    # 真实样本个数计算, * 前者的每一行的元素会对应乘上后者的元素。
    sample_labels_div = sample_labels_div * num_in_classes
    # each client removes small classes.
    sample_labels_div[sample_labels_div < sample_labels_div.max(axis=1, keepdims=True) * 0.1] = 0
    sample_labels_div = np.floor(sample_labels_div)
    sample_labels = np.cumsum(sample_labels_div, axis=0).astype(np.int32)

    # 从超额的客户端中随机选择N个数出来，全部汇集起来，然后其他不够的客户端从中随机均匀采样
    data_num_each_user = len(targets_tensor) // num_users
    new_sample_set = []
    for user_index in range(num_users):
        rand_set = []
        for cls_index in idxs_dict.keys():
            cls_set = idxs_dict[cls_index]
            if user_index == 0:
                tail = sample_labels[user_index, cls_index]
                rand_set.extend(cls_set[:tail])
                idxs_dict[cls_index] = cls_set[tail:]
            else:
                head = sample_labels[user_index - 1, cls_index]
                tail = sample_labels[user_index, cls_index]
                rand_set.extend(cls_set[:tail - head])
                idxs_dict[cls_index] = cls_set[tail - head:]
        diff_num = len(rand_set) - data_num_each_user
        if diff_num <= 0:
            dict_users[user_index] = rand_set
        else:
            select_ids = np.random.choice(rand_set, diff_num, replace=False)
            rand_set = list(set(rand_set).difference(set(select_ids)))
            dict_users[user_index] = rand_set
            new_sample_set.extend(select_ids)

    for cls_index in idxs_dict.keys():
        cls_set = idxs_dict[cls_index]
        new_sample_set.extend(cls_set)

    for user_index in range(num_users):
        diff_num = data_num_each_user - len(dict_users[user_index])
        if diff_num > 0:
            select_ids = np.random.choice(new_sample_set, diff_num, replace=False)
            new_sample_set = list(set(new_sample_set).difference(set(select_ids)))
            dict_users[user_index].extend(select_ids)
        dict_users[user_index] = np.array(dict_users[user_index])

    if isinstance(targets_tensor, list):
        columns = ['user_id', 'data_num', 'classes_num', 'entropy', *map(str, list(set(targets_tensor)))]
    else:
        columns = ['user_id', 'data_num', 'classes_num', 'entropy', *map(str, list(set(targets_tensor.numpy())))]
    df = pd.DataFrame(columns=columns)
    for user_id, idxs in dict_users.items():
        labels = []
        for i in idxs:
            labels.append(targets_tensor[i])
        df_user = pd.DataFrame(labels).value_counts()
        results = {'user_id': user_id, 'data_num': df_user.sum(), 'classes_num': len(df_user.index)}
        for key, value in df_user.items():
            results[str(key[0])] = value
        df = df.append([results], ignore_index=True)
    df = df.fillna(0)
    for user_id in range(len(df)):
        p = df.iloc[user_id, 4:] / df.iloc[user_id, 1]
        df.iloc[user_id, 3] = (p * (- np.log2(p))).sum()

    new_indexes = df.sort_values(by="entropy", ascending=True)['user_id'].values
    dict_users_new = {}
    for i in range(len(new_indexes)):
        old_index = new_indexes[i]
        dict_users_new[i] = dict_users[old_index]
    dict_users = dict_users_new

    return dict_users


if __name__ == '__main__':
    # 
    num_users = 100
    alpha = 1.0
    targets = np.arange(10).reshape(1,-1).repeat(5000, axis=0).reshape(-1)
    dict_users, dict_frequency_classes = noniid_dir(targets, num_users, alpha)
    dict_users_test = imitate_dict_users(targets, num_users, dict_frequency_classes)
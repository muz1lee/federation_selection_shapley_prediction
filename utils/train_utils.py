import os
import pickle
import copy
import math
import numpy as np
import torch
import socket
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from global_tools import load_pkl, save_pkl, load_json
from utils.dataset import Emnist
from utils.sampling import iid, imitate_sampling, noniid_label, noniid_dir, noniid_dir_mix_iid, imbalanced_iid, noniid_dir_same_num, noniid_step

from models.Nets import MLP, CNNCifar
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.vgg import VGG
from models.Nets import MLP_emnist

# learning problem -> model type(experiment methods) -> dataset/basic model

hostname = socket.gethostname()
if hostname.startswith('xuyc-Server'):
    root_path = '/home/xuyc/datasets/'
elif hostname.startswith('amax'):
    root_path = '/data/xuyc/datasets/'

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def weight_init(model, std_lin=0.01):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)


def set_data_loader(datasets: dict, args):
    data_loader = {}
    for name, dataset in datasets.items():
        if name == 'train':
            data_loader[name] = DataLoader(dataset, batch_size=args.bs_train, shuffle=True)
        elif name in ['test', 'valid']:
            data_loader[name] = DataLoader(dataset, batch_size=args.bs_test, shuffle=False)
        elif name == 'gtest':
            data_loader[name] = DataLoader(dataset, batch_size=args.bs_test, shuffle=False)
    return data_loader


class Cifar100Wrapper(Dataset):
    def __init__(self, root_path:str, transform, train:bool=True, download:bool=True, chose_cls:list=None, data_ratio:float=1.0, merged_cls_num:int=100):
        """
            quantity
        """
        self.dataset = datasets.CIFAR100(root_path, train=train, download=download, transform=transform)
        idxs_dict = {}  # {'label': 'dataset samples' id'}
        targets = self.dataset.targets
        for i in range(len(targets)):
            label = targets[i]
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)
        
        # dataset.classes  can get all fine class name.
        self.target_fine  = self.dataset.targets
        info = self.unpickle(os.path.join(root_path,"cifar-100-python/train"))
        self.targets_coarse = [i for i in info[b'coarse_labels']]
        info_meta = self.unpickle(os.path.join(root_path,"cifar-100-python/meta"))
        self.fine_class_names = self.dataset.classes
        self.coarse_class_names = [i.decode() for i in info_meta[b'coarse_label_names']]
        del info
        del info_meta

        self.coarse_to_fine_idx = {}
        self.fine_to_coarse_idx = {}

        for i in range(len(self.target_fine)):
            if self.targets_coarse[i] not in self.coarse_to_fine_idx:
                self.coarse_to_fine_idx[self.targets_coarse[i]] = set()
            self.coarse_to_fine_idx[self.targets_coarse[i]].add(self.target_fine[i])
            self.fine_to_coarse_idx[self.target_fine[i]] = self.targets_coarse[i]

        self.coarse_to_fine_idx = {k: list(self.coarse_to_fine_idx[k]) for k in sorted(self.coarse_to_fine_idx)}
        self.coarse_to_fine_name = {}
        for k,v in self.coarse_to_fine_idx.items():
            self.coarse_to_fine_name[self.coarse_class_names[k]] = []
            for i in v:
                self.coarse_to_fine_name[self.coarse_class_names[k]].append(self.fine_class_names[i])


        self.indexs = []
        self.chose_cls = chose_cls
        if chose_cls is None:
            chose_cls = list(idxs_dict.keys())
        for key, value in idxs_dict.items():
            if key not in chose_cls:
                    continue
            else:
                if train is True:
                    num = len(value)
                    chose_num = math.ceil(num * data_ratio)
                    select_ids = np.random.choice(value, chose_num, replace=False)
                else:
                    select_ids = value
                self.indexs.extend(select_ids)

        # if shuffle:
            # random.shuffle(self.indexs)
        

        self.proj_index = {}
        self.cls_num = merged_cls_num
        if self.cls_num == 20:
            self.proj_index = self.fine_to_coarse_idx
        elif self.cls_num == 2:
            creature_idx = ([0,1,7,8] + list(range(11,17)))
            fine_to_2_idx = {i:0 if i in creature_idx else 1 for i in range(20)}
            self.proj_index = {k:fine_to_2_idx[v] for k,v in self.fine_to_coarse_idx.items()}

        if self.chose_cls is not None:
            self.proj_index = {v:k for k,v in enumerate(self.chose_cls)}


        self.targets = [self.proj_index[self.dataset[i][1]] for i in self.indexs]

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        feature, label = self.dataset[self.indexs[idx]]
        if self.cls_num != 100 or self.chose_cls is not None:
            label = self.proj_index[label]
        return (feature, label)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

def convert_exp_to_param(nid:str="a-1.0"):
    """ convert exploration experiments name into concrete parameters.

    :param str split: "[NAME]-[INDEX]", defaults to "a-1.0"
    """
    
    _ = load_json(os.path.join(root_path,"cifar-100-python/class_idx_convert.json"))
    coarse_to_fine_idx, fine_to_coarse_idx = _["coarse_to_fine_idx"], _["fine_to_coarse_idx"]
    coarse_to_fine_idx = {int(k):v for k,v in coarse_to_fine_idx.items()}
    fine_to_coarse_idx = {int(k):v for k,v in fine_to_coarse_idx.items()}

    result = {}
    name, index = nid.split("-")
    if name == "a":
        # ind_to_ratio = {1:0.1, 2:0.4, 3:1.0}
        index = float(index)
        assert index <= 1.0, "data_ratio only support <=1.0"
        result = {**result, "data_ratio": index}
    elif name == "b":
        chose_cls = []
        index = int(index)
        assert index <= 20, "chose_cls only support 20 coarse classes"
        for i in range(index):
            chose_cls.extend(coarse_to_fine_idx[i])
        result = {**result, "chose_cls": chose_cls}
    elif name == "c":
        index = int(index)
        assert index in [2, 20, 100], "merged_cls_num only support [2, 20, 100]"
        result = {**result, "merged_cls_num": index}
    # elif name == "d":
    #     indexs = index.split("|")

    #     if index < 6:
    #         chose_cls = 
    #     result = {**result, "chose_cls": chose_cls}

    return result

def get_center_data(dataset, nid=None):
    """center training exploration

    :param _type_ args: _description_
    :return _type_: _description_
    """
    dataset_train, dataset_test = None, None
    if dataset == 'mnist':
        dataset_train = datasets.MNIST(root_path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(root_path, train=False, download=True, transform=trans_mnist)
    elif dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(root_path, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(root_path, train=False, download=True, transform=trans_cifar10_val)
    elif dataset == 'cifar100':
        split_param = convert_exp_to_param(nid=nid)
        dataset_train = Cifar100Wrapper(root_path, transform=trans_cifar100_train, train=True, download=True, **split_param)
        dataset_test = Cifar100Wrapper(root_path, transform=trans_cifar100_val, train=False, download=True, **split_param)
        # dataset_gtest = datasets.CIFAR100(root_path, transform=trans_cifar100_val, train=False, download=True)
    else:
        exit('Error: unrecognized dataset')

    return {'train': dataset_train, 'test': dataset_test}
    # 'gtest': dataset_gtest}

def get_normal_data(dataset):
    """

    :param args: dataset
    :return:
    """
    dataset_train, dataset_test = None, None
    if dataset == 'mnist':
        dataset_train = datasets.MNIST(root_path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(root_path, train=False, download=True, transform=trans_mnist)
    elif dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(root_path, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(root_path, train=False, download=True, transform=trans_cifar10_val)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(root_path, train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100(root_path, train=False, download=True, transform=trans_cifar100_val)
    else:
        exit('Error: unrecognized dataset')

    return {'train': dataset_train, 'test': dataset_test}


def get_fed_data(setting_path, dataset, split, num_users, dir_alpha, clsnum_peruser, iid_frac, imb_alpha, data_frac):
    """

    :param args: dataset/split/num_users/shard_per_user/dir_alpha/
    :param mode: nn/fed
    :return:
    """
    dataset_train, dataset_test = None, None
    dict_save_path = os.path.join(setting_path, 'dict_users.pkl')

    if dataset == 'mnist':
        dataset_train = datasets.MNIST(root_path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(root_path, train=False, download=True, transform=trans_mnist)
    elif dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(root_path, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(root_path, train=False, download=True, transform=trans_cifar10_val)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(root_path, train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100(root_path, train=False, download=True, transform=trans_cifar100_val)
    elif dataset == 'cifar20':
        split_param = convert_exp_to_param(nid='b-2')
        dataset_train = Cifar100Wrapper(root_path, transform=trans_cifar100_train, train=True, download=True, **split_param)
        dataset_test = Cifar100Wrapper(root_path, transform=trans_cifar100_val, train=False, download=True, **split_param)
    elif dataset == 'emnist':
        dataset_train = Emnist(root_path, train=True)
        dataset_test = Emnist(root_path, train=False)
    else:
        exit('Error: unrecognized dataset')

    if os.path.exists(dict_save_path):
        dict_users_train, dict_users_test = load_pkl(dict_save_path)
        print('***Load split of users from: %s' % dict_save_path)
    else:
        # sample users
        if split == 'iid':
            dict_users_train = iid(dataset_train, num_users, data_frac)
            dict_users_test = iid(dataset_test, num_users)
        elif split == 'niid-label':
            dict_users_train, rand_set_all = noniid_label(dataset_train, num_users, shard_per_user=clsnum_peruser, data_frac=data_frac, is_uniform=False)
            dict_users_test, _ = noniid_label(dataset_test, num_users, shard_per_user=clsnum_peruser, rand_set_all=rand_set_all, is_uniform=False)
            # dict_users_test = iid(dataset_test, num_users)
        # elif split == 'step':
        #     dict_users_train, rand_set_all = noniid_step(dataset_train.targets, num_users, shard_per_user=clsnum_peruser)
        #     dict_users_test, rand_set_all = noniid_step(dataset_test.targets, num_users, shard_per_user=clsnum_peruser, rand_set_all=rand_set_all)
        elif split == 'dir':
            dict_users_train, pre_dict_frequency_classes = noniid_dir(dataset_train.targets, num_users, data_frac, alpha=dir_alpha)
            dict_users_test = imitate_sampling(dataset_test.targets, num_users,dict_frequency_classes=pre_dict_frequency_classes)
        elif split == 'mix':
            dict_users_train, rand_set_all = noniid_dir_mix_iid(dataset_train, num_users, iid_frac=iid_frac,
                                                                alpha=dir_alpha)
            dict_users_test, rand_set_all = noniid_dir_mix_iid(dataset_test, num_users, iid_frac=iid_frac,
                                                            rand_set_all=rand_set_all, alpha=dir_alpha)
        elif split == 'imb-iid':
            dict_users_train = imbalanced_iid(dataset_train, num_users, alpha=imb_alpha)
            dict_users_test = imbalanced_iid(dataset_test, num_users, alpha=imb_alpha)
        elif split == 'same-dir':
            dict_users_train = noniid_dir_same_num(dataset_train, num_users, alpha=dir_alpha)
            dict_users_test = iid(dataset_test, num_users)
        # save users
        
        save_pkl(dict_save_path, (dict_users_train, dict_users_test))

    return dataset_train, dataset_test, dict_users_train, dict_users_test


def get_weight_initial_std(weight):
    fan = nn.init._calculate_correct_fan(weight, "fan_out")
    gain = nn.init.calculate_gain("relu")
    std = gain / math.sqrt(fan)
    return std

def get_model(dataset, model, num_classes:int=None, norm='bn', num_groups=32):
    """

    :param args: dataset/device/model
    :return:
    """
    # according to model type, decide import.

    if dataset in ["mnist", "cifar10", "emnist"]:
        num_classes = 10
    elif dataset == "cifar100":
        if num_classes is None:
            num_classes = 100
    elif dataset == "cifar20":
        num_classes = 10
    if dataset == "mnist":
        num_channels = 1
    else:
        num_channels = 3

    if model == 'cnn':
        if dataset in ['cifar10', 'cifar100', 'cifar20']:
            model = CNNCifar(num_channels, num_classes)
    elif model == 'mlp':
        if dataset == 'mnist':
            model = MLP(input_channel=784, hidden_channel=200, num_classes=10)
        elif dataset == 'emnist':
            model = MLP_emnist(num_classes=10)
    elif 'vgg' in model:
        model = VGG(model, num_classes, norm_name=norm, num_groups=num_groups)

    elif model == 'resnet18' and dataset in ['cifar10', 'cifar100']:
        model = ResNet18(num_classes, norm_name=norm, num_groups=num_groups)
    elif model == 'resnet34' and dataset in ['cifar10', 'cifar100']:
        model = ResNet34(num_classes, norm_name=norm, num_groups=num_groups)
    elif model == 'resnet50' and dataset in ['cifar10', 'cifar100']:
        model = ResNet50(num_classes, norm_name=norm, num_groups=num_groups)
    elif model == 'resnet101' and dataset in ['cifar10', 'cifar100']:
        model = ResNet101(num_classes, norm_name=norm, num_groups=num_groups)
    elif model == 'resnet152' and dataset in ['cifar10', 'cifar100']:
        model = ResNet152(num_classes, norm_name=norm, num_groups=num_groups)
    else:
        exit('Error: unrecognized model')

    print(model)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        model = model.cuda()
    

    return model

def split_collection(targets, dict_users, collection_rounds=1):
    # targets = np.repeat(np.expand_dims(np.repeat(np.arange(10), 1000),0), 10, axis=0).reshape(-1)
    # n_targets = len(targets) // 10
    # dict_users = {i: list(range(n_targets * i, n_targets * (i + 1))) for i in range(10)}
    # dict_users_split=split_collection(targets, dict_users, collection_rounds=10)
    # for k,v in dict_users_split[-1].items():
    #    print(set(dict_users_split[-1][k]) == set(dict_users[k]))
    dict_users = copy.deepcopy(dict_users)
    n_classes = len(set(targets))
    dict_users_split = []
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)
    # new collect number in the current round
    n_per_cls_user = math.floor(len(targets) / (len(dict_users) * n_classes * collection_rounds))

    for rod in range(1, collection_rounds+1):
        dict_users_round = {}
        for user, id_data in dict_users.items():
            dict_users_round[user] = []
            id_data = np.array(id_data)
            for c in range(n_classes):
                ids_cls = id_data[np.where(targets[id_data]==c)[0]]
                ids_select_cls = np.random.choice(ids_cls, n_per_cls_user, replace=False)
                dict_users_round[user] += ids_select_cls.tolist()
                id_data = np.array(list(set(id_data).difference(set(ids_select_cls))))
            dict_users[user] = id_data
            if rod > 1:
                dict_users_round[user] = dict_users_split[-1][user] + dict_users_round[user]
        dict_users_split.append(dict_users_round)
    return dict_users_split

# 改了iid的bug

def set_wandb(use_wandb=True):
    global wandb
    if use_wandb:
        import wandb as t
        wandb = t
    else:
        from utils import wandb as t
        wandb = t


wandb = None

"""
Reference:
FedML: A Research Library and Benchmark for Federated Machine Learning
@misc{he2020fedml,
      title={FedML: A Research Library and Benchmark for Federated Machine Learning}, 
      author={Chaoyang He and Songze Li and Jinhyun So and Xiao Zeng and Mi Zhang and Hongyi Wang and Xiaoyang Wang and Praneeth Vepakomma and Abhishek Singh and Hang Qiu and Xinghua Zhu and Jianzong Wang and Li Shen and Peilin Zhao and Yan Kang and Yang Liu and Ramesh Raskar and Qiang Yang and Murali Annavaram and Salman Avestimehr},
      year={2020},
      eprint={2007.13518},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""
import json
import logging
import os

import numpy as np
import torch
from torchvision import transforms
import logging

import numpy as np
import torch.utils.data as data
from PIL import Image

import torchvision


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def load_partition_data_FashionMNIST_by_device_id(batch_size,
                                           device_id,
                                           train_path="FashionMNIST_mobile",
                                           test_path="FashionMNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_FashionMNIST(batch_size, train_path, test_path)


def partition_data_dataset(X_train,y_train, n_nets, alpha):
    min_size = 0
    K = 10
    N = y_train.shape[0]
    print("N = " + str(N))
    net_dataidx_map = {}

    while min_size < 10:
        #print(min_size)
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):#K个类别
            idx_k = np.where(y_train == k)[0]
            np.random.seed(k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets)) 
            np.random.shuffle(idx_k)
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map


def partition_data(dataset, datadir, partition, n_nets, alpha):
    print("*********partition data***************")
    X_train, y_train, X_test, y_test = load_FashionMNIST_data(datadir)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        test_total_num= n_test
        idxs = np.random.permutation(total_num)
        idxs_test= np.random.permutation(test_total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_test={i: batch_idxs_test[i] for i in range(n_nets)}

    elif partition == "hetero":#在此处分割数据
        net_dataidx_map_train=partition_data_dataset(X_train,y_train,n_nets,alpha)
        net_dataidx_map_test=partition_data_dataset(X_test,y_test,n_nets,alpha)
    else:
        raise Exception("partition arg error")
    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test




def _data_transforms_FashionMNIST():
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.expand((3, 28, 28))),
        transforms.Normalize([0.1307,0.1307,0.1307],[0.3081,0.3081,0.3081]),
    ])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.expand((3, 28, 28))),
        transforms.Normalize([0.1307,0.1307,0.1307],[0.3081,0.3081,0.3081]),
    ])

    return train_transform, valid_transform


class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        FashionMNIST_dataobj = torchvision.datasets.FashionMNIST(root=self.root, train=self.train, transform=self.transform, download=self.download)
        data = None
        target = None
        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = FashionMNIST_dataobj.train_data
            data = FashionMNIST_dataobj.data
            target = np.array(FashionMNIST_dataobj.targets)
        else:
            data = FashionMNIST_dataobj.data
            target = np.array(FashionMNIST_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img=img.reshape(1,28,28).type(torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    return get_dataloader_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train,dataidxs_test)


def get_dataloader_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    dl_obj = FashionMNIST_truncated

    transform_train, transform_test = _data_transforms_FashionMNIST()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def load_FashionMNIST_data(datadir):
    train_transform, test_transform = _data_transforms_FashionMNIST()
    train_data=FashionMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    test_data = FashionMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    
    X_train, y_train = train_data.data, np.array(train_data.target)
    X_test, y_test = test_data.data, np.array(test_data.target)

    return (X_train, y_train, X_test, y_test)

def load_partition_data_FashionMNIST(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num_train = len(np.unique(y_train))
    class_num_test = len(np.unique(y_test))
    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    print("train_dl_global number = " + str(len(train_data_global)))
    print("test_dl_global number = " + str(len(test_data_global)))

    data_local_num_dict_train = dict()
    data_local_num_dict_test = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs_train = net_dataidx_map_train[client_idx]
        dataidxs_test =  net_dataidx_map_test[client_idx]


        local_data_num_train = len(dataidxs_train)
        local_data_num_test = len(dataidxs_test)


        data_local_num_dict_train[client_idx] = local_data_num_train
        data_local_num_dict_test[client_idx] = local_data_num_test

        print("client_idx = %d, train_local_sample_number = %d" % (client_idx, local_data_num_train))
        print("client_idx = %d, test_local_sample_number = %d" % (client_idx, local_data_num_test))
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs_train,dataidxs_test)


        print("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

        
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict_train, data_local_num_dict_test,train_data_local_dict, test_data_local_dict, class_num_train,class_num_test




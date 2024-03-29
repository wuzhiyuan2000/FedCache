import argparse
import logging
import os
import sys
import numpy as np
import argparse
from sys import argv
import torch
import random
from fashionmnist_data_loader import load_partition_data_FashionMNIST
from resnet_client import resnet20, resnet16, resnet8
from FedCache import FedCache_standalone_API


def add_args(parser):
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers hetero/homo')
    parser.add_argument('--model_setting', type=str, default='hetero', metavar='N',
                        help='how to set on-device models on clients hetero/homo')
    parser.add_argument('--wd', type=float, default=5e-4, 
                        help='weight decay parameter;')
    parser.add_argument('--comm_round', type=int, default=1000,
                        help='how many round of communications we shoud use (default: 1000)')
    parser.add_argument('--alpha', default=1.5, type=float, 
                        help='Input the relative weight: default (1.5)')    
    parser.add_argument('--sel', type=int, default=1, metavar='EP',
                        help='one out of every how many clients is selected to conduct testing  (default: 1)')
    parser.add_argument('--interval', type=int, default=1, metavar='EP',
                        help='how many communication round intervals to conduct testing  (default: 1)')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--client_number', type=int, default=20, metavar='NN',
                        help='number of clients')
    parser.add_argument('--partition_alpha', type=float, default=1.0, metavar='PA',
                        help='partition alpha (default: 1.0)')
    parser.add_argument('--class_num', type=int, default=10,
                        help='class_num')
    parser.add_argument('--R', type=int, default=16,
                        help='how many other samples are associated with each sample')
    parser.add_argument('--T', type=float, default=1.0,
                        help='distrillation temperature (default: 1.0)')
    parser.add_argument('--dataset', type=str, default='fashionmnist', metavar='N',
                        help='dataset used for training')
    args = parser.parse_args()
    args.client_number_per_round=args.client_number
    args.client_num_in_total=args.client_number
    return args


def load_data(args, dataset_name):
    data_loader = load_partition_data_FashionMNIST
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num_train, class_num_test = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_number, args.batch_size)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test]
    return dataset


def create_client_model(args, n_classes,index):
    if args.model_setting=='hetero':
        if index%3==0:
            return resnet8(n_classes)
        elif index%3==1:
            return resnet16(n_classes)
        else:
            return resnet20(n_classes)
    elif args.model_setting=='homo':
        return resnet20(n_classes)
    else:
        raise Exception("model setting exception")

def create_client_models(args, n_classes):
    random.seed(123)
    client_models=[]
    for _ in range(args.client_number):
        client_models.append(create_client_model(args,n_classes,_))
    return client_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(5))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test] = dataset
    client_models=create_client_models(args,class_num_train)
    api=FedCache_standalone_API(client_models,train_data_local_num_dict,test_data_local_num_dict, train_data_local_dict, test_data_local_dict, args,test_data_global)
    api.do_fedcache_stand_alone(client_models,train_data_local_num_dict, test_data_local_num_dict,train_data_local_dict, test_data_local_dict, args)
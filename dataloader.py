#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
import random
import torch
import pandas as pd
import scipy.io as io
import torch.utils.data as data_utils
from torchvision import datasets, transforms

 
def to_categorical(y, nr_class):
    '''
    generate one-hot label
    '''
    y_list = [0] * nr_class
    for i in y:
        y_list[i] = 1
    y_cate = np.array(y_list)

    return y_cate


def load_data_mat(mat_path, nr_fea, nr_class, normalize=False):
    '''
    load the dataset in mat format 
    '''
    data_mat = io.loadmat(mat_path)
    data = data_mat['data']
    all_ins_fea = []
    all_ins_fea_tmp = np.empty((0, nr_fea))
    ins_num, bag_idx_of_ins = [], []
    bag_lab, dummy_ins_lab = [], []
    partial_bag_lab = []
    partial_dummy_ins_lab = np.empty((0, nr_class))
    partial_dummy_ins_lab_processed = np.empty((0, nr_class))
    partial_bag_lab, partial_bag_lab_processed = np.empty((0, nr_class)), np.empty((0, nr_class))
    bag_cnt = 1
    for i in range(data.shape[0]):
        all_ins_fea = np.vstack((all_ins_fea_tmp, data[i, 0]))
        all_ins_fea_tmp = all_ins_fea
        ins_num_tmp = data[i, 0].shape[0]
        ins_num.append(ins_num_tmp)
        bag_idx_of_ins_tmp = [bag_cnt] * ins_num_tmp
        bag_idx_of_ins = bag_idx_of_ins + bag_idx_of_ins_tmp
        bag_cnt += 1
        # the ground-truth labels of bags
        bag_lab_tmp = list(data[i, 2].flatten() - 1)
        bag_lab = bag_lab + bag_lab_tmp
        dummy_ins_lab_tmp = [bag_lab_tmp] * ins_num_tmp
        dummy_ins_lab = dummy_ins_lab + dummy_ins_lab_tmp
        # the partial labels of bags
        partial_bag_lab_tmp = list(data[i, 1].flatten() - 1)
        partial_bag_lab_tmp = to_categorical(partial_bag_lab_tmp, nr_class)
        partial_bag_lab_tmp = np.expand_dims(partial_bag_lab_tmp, axis=0)
        partial_bag_lab = np.vstack((partial_bag_lab, partial_bag_lab_tmp))
        partial_dummy_ins_lab_tmp = partial_bag_lab_tmp.repeat(ins_num_tmp, axis=0)
        partial_dummy_ins_lab = np.vstack((partial_dummy_ins_lab, partial_dummy_ins_lab_tmp))

    bag_idx_of_ins = np.array(bag_idx_of_ins)
    bag_idx_of_ins = np.expand_dims(bag_idx_of_ins, axis=1)
    bag_lab = np.array(bag_lab)
    dummy_ins_lab = np.array(dummy_ins_lab)
    lab_inx_fea = np.hstack((dummy_ins_lab, bag_idx_of_ins, all_ins_fea))
    nr_partial_lab_per_ins = np.expand_dims(np.sum(partial_dummy_ins_lab, 1), axis=1)
    partial_dummy_ins_lab_processed = partial_dummy_ins_lab / nr_partial_lab_per_ins
    nr_partial_lab_per_bag = np.expand_dims(np.sum(partial_bag_lab, 1), axis=1)
    partial_bag_lab_processed = partial_bag_lab / nr_partial_lab_per_bag

    if normalize:
        data_mean, data_std = np.mean(all_ins_fea, 0), np.std(all_ins_fea, 0)
        data_min, data_max = np.min(all_ins_fea, 0), np.max(all_ins_fea, 0)
        all_ins_fea_norm = (all_ins_fea - data_mean) / data_std
        all_ins_fea = all_ins_fea_norm

    all_ins_fea = torch.from_numpy(all_ins_fea)
    bag_idx_of_ins = torch.from_numpy(bag_idx_of_ins)
    dummy_ins_lab = torch.from_numpy(dummy_ins_lab)
    bag_lab = torch.from_numpy(bag_lab)
    partial_bag_lab = torch.from_numpy(partial_bag_lab)
    partial_bag_lab_processed = torch.from_numpy(partial_bag_lab_processed)

    return all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed


def load_idx_mat(idx_file):
    '''
    load the index in mat format
    '''
    idx = io.loadmat(idx_file)
    idx_tr_np = idx['trainIndex']
    idx_te_np = idx['testIndex']
    idx_tr = list(np.array(idx_tr_np).flatten())
    idx_te = list(np.array(idx_te_np).flatten())
    random.shuffle(idx_tr)
    random.shuffle(idx_te)

    return idx_tr, idx_te


class MIPLDataloader(data_utils.Dataset):
    def __init__(self, all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr, idx_te,
                 args, seed=1, train=True, normalize=False):
        self.all_ins_fea = all_ins_fea
        self.bag_idx_of_ins = bag_idx_of_ins
        self.dummy_ins_lab = dummy_ins_lab
        self.bag_lab = bag_lab
        self.partial_bag_lab_processed = partial_bag_lab_processed
        self.idx_tr = idx_tr
        self.idx_te = idx_te
        self.train = train
        self.args = args
        self.nr_fea = args.nr_fea
        self.normalize = normalize

        if self.train:
            self.train_bags_list, self.train_ins_lab_list, self.train_partial_bag_lab_list, \
            self.train_true_bag_lab_list = self._create_bags()
        else:
            self.test_bags_list, self.test_ins_lab_list, self.test_partial_bag_lab_list, \
            self.test_true_bag_lab_list = self._create_bags()

    def _create_bags(self):
        bags_list, ins_lab_list, partial_bag_lab_list, true_bag_lab_list = [], [], [], []
        if self.train:
            for i in self.idx_tr:
                bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i
                bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
                bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]
                ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]
                partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]
                partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0)
                true_bag_lab = self.bag_lab[i - 1]
                if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
                    bag = bag.reshape(bag.shape[0], 1, 28, 28)
                bags_list.append(bag)
                ins_lab_list.append(ins_lab)
                partial_bag_lab_list.append(partial_bag_lab)
                true_bag_lab_list.append(true_bag_lab)
        else:
            for i in self.idx_te:
                bag_idx_of_ins_a_bag = self.bag_idx_of_ins == i
                bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
                bag = self.all_ins_fea[bag_idx_of_ins_a_bag, :]
                ins_lab = self.dummy_ins_lab[bag_idx_of_ins_a_bag]
                partial_bag_lab = self.partial_bag_lab_processed[i - 1, :]
                partial_bag_lab = np.expand_dims(partial_bag_lab, axis=0)
                true_bag_lab = self.bag_lab[i - 1]
                if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
                    bag = bag.reshape(bag.shape[0], 1, 28, 28)
                bags_list.append(bag)
                ins_lab_list.append(ins_lab)
                partial_bag_lab_list.append(partial_bag_lab)
                true_bag_lab_list.append(true_bag_lab)

        return bags_list, ins_lab_list, partial_bag_lab_list, true_bag_lab_list

    def __len__(self):
        if self.train:
            return len(self.train_ins_lab_list)
        else:
            return len(self.test_ins_lab_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            partial_bag_label = self.train_partial_bag_lab_list[index]
            true_bag_label = self.train_true_bag_lab_list[index]
        else:
            bag = self.test_bags_list[index]
            partial_bag_label = self.test_partial_bag_lab_list[index]
            true_bag_label = self.test_true_bag_lab_list[index]

        return bag, partial_bag_label, true_bag_label, index

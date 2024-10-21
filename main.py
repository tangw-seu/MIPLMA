#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from __future__ import print_function
import argparse
import random
import os
import sys
import time
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import scipy.io as io
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from dataloader import *
from model import *
from utils import *

# training settings
parser = argparse.ArgumentParser(description='Multi-Instance Partial-Label Learning with Margin Adjustment')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R', help='weight decay')
parser.add_argument('--w_lambda', type=float, default=0.01, metavar='L', help='weight of the loss function')
parser.add_argument('--L', type=int, default=128, metavar='S', help='the embedded space dimension')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')
parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
parser.add_argument('--index', type=str, default='index', help='index path')
parser.add_argument('--ds', type=str, default='MNIST_MIPL', help='MNIST_MIPL, FMNIST_MIPL, ...')
parser.add_argument('--ds_suffix', type=str, default='1', help='the specific type of the data set')
parser.add_argument('--bs_tr', type=int, default=1, help='batch size for training')
parser.add_argument('--bs_te', type=int, default=1, help='batch size for testing')
parser.add_argument('--nr_fea', type=int, default=784, help='feature dimension of an instance')
parser.add_argument('--nr_class', type=int, default=5, help='classes of bag')
parser.add_argument('--normalize', type=str2bool, default=False, help='normalize the dataset, True or False')
parser.add_argument('--init_tau', type=float, default=5.0, metavar='LR', help='initialized tau')
parser.add_argument('--min_tau', type=float, default=0.1, metavar='LR', help='minimum tau')
parser.add_argument('--decay_tau', type=float, default=0.05, metavar='LR', help='decay')
parser.add_argument('--mar_scale', type=float, default=0.5, metavar='L', help='the scale of margin loss')
parser.add_argument('--smoke_test', action='store_true', default=False, help='smoke_test, True or False')
args = parser.parse_args()

seed_everything(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cuda:
    print('\nGPU is available!')

all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
             'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']
if args.smoke_test:
    all_folds = ['index1.mat']

def evaluate(loader, model):
    '''
    model testing
    '''
    model.eval()
    all_true_bag_lab = []
    all_pred_bag_lab = []
    all_pred_bag_prob = np.empty((0, args.nr_class))
    for data, _, true_bag_lab, _ in loader:
        data = data.to(device)
        true_bag_lab = true_bag_lab.to(device)
        data = data.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        output = model.evaluate_objective(data, args)
        all_pred_bag_prob = np.vstack((all_pred_bag_prob, output.detach().cpu().numpy()))
        _, pred_bag_lab = torch.max(output.data, 1)
        all_true_bag_lab.append(true_bag_lab.item())
        all_pred_bag_lab.append(pred_bag_lab.item())
    all_true_bag_lab = np.array(all_true_bag_lab)
    all_pred_bag_lab = np.array(all_pred_bag_lab)
    acc = accuracy_score(all_true_bag_lab, all_pred_bag_lab)

    return acc


def train(epoch):
    '''
     model training
    '''
    model.train()
    train_loss = 0.
    margins_loss_epoch = torch.tensor([0.], requires_grad=True).to(device)
    for batch_idx, (data, partial_bag_lab, true_bag_lab, index) in enumerate(train_loader):
        if args.cuda:
            data, partial_bag_lab, true_bag_lab = data.cuda(), partial_bag_lab.cuda(), true_bag_lab.cuda()
        data, partial_bag_lab, true_bag_lab = Variable(data), Variable(partial_bag_lab), Variable(true_bag_lab)
        data = data.to(torch.float32)
        partial_bag_lab = partial_bag_lab.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, new_partial_bag_lab, margin_losses_per_epoch_new = model.calculate_objective(data, partial_bag_lab, margins_loss_epoch, args)
        train_loss += loss.item()
        lamda = alpha_list[epoch - 1]
        new_partial_bag_lab = new_partial_bag_lab.cpu().detach().numpy()
        partial_bag_lab = partial_bag_lab.cpu().detach().numpy()
        new_label = lamda * partial_bag_lab + (1. - lamda) * new_partial_bag_lab
        new_label = np.squeeze(new_label, axis=0)
        train_loader.dataset.train_partial_bag_lab_list[index] = new_label
        # backward pass
        loss.backward()
        optimizer.step()
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    if epoch == 1 or (epoch) % 10 == 0:
        print('Epoch: {}, Train loss: {:.4f}.'.format(epoch, train_loss))

    return model, train_loss


def adjust_alpha(epochs):
    '''
    the weighting factor in Equation (8)
    $\alpha^{(t)} = \frac{T-t}{T}$
    '''
    alpha_list = [1.0] * epochs
    for ep in range(epochs):
        alpha_list[ep] = (epochs - ep) / (epochs)
    return alpha_list


if __name__ == "__main__":
    time_s = time.time()
    alpha_list = adjust_alpha(args.epochs)
    num_trial = 1
    num_fold = len(all_folds)
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_path = os.path.join(args.data_path, args.ds)
    index_path = os.path.join(data_path, args.index)
    if args.ds != "CRC_MIPL":
        mat_name = args.ds + "_r" + args.ds_suffix + ".mat"
    else:
        mat_name = args.ds + "_" + args.ds_suffix + ".mat"
    print(' - ', mat_name)
    mat_path = os.path.join(data_path, mat_name)
    ds_name = mat_name[0:-4]
    all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed = load_data_mat(
        mat_path, args.nr_fea, args.nr_class, normalize=args.normalize)

    accuracy = np.empty((num_trial, num_fold))
    for trial_i in range(num_trial):
        for fold_i in range(num_fold):
            print('\n---------------- time: %d, fold: %d ----------------' % (trial_i + 1, fold_i + 1))
            idx_file = index_path + '/' + all_folds[fold_i]
            # load the index and dataset
            idx_tr, idx_te = load_idx_mat(idx_file)
            train_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr,
                               idx_te, args, seed=args.seed, train=True, normalize=args.normalize),
                batch_size=args.bs_tr, shuffle=True, **loader_kwargs)
            test_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr,
                               idx_te, args, seed=args.seed, train=False, normalize=args.normalize),
                batch_size=args.bs_tr, shuffle=False, **loader_kwargs)

            # ---------------- init model ----------------
            model = GatedAttention(args)
            if args.cuda:
                model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.reg)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # -------------- start training ---------------
            for epoch in range(1, args.epochs + 1):
                model, loss = train(epoch)
                model.decay_temperature()
                lr_scheduler.step()

            # -------------- start testing ---------------
            test_accuracy = evaluate(test_loader, model)
            print('test_acc: {:.3f}'.format(test_accuracy))
            accuracy[trial_i, fold_i] = test_accuracy
    print('The mean and std of accuracy at %d times %d folds: %f, %f' % (
    num_trial, num_fold, np.around(np.mean(accuracy), 3), np.around(np.std(accuracy), 3)))

    time_e = time.time()
    print('\nRunning time is', time_e - time_s, 'seconds.')
    print('Training is finished.')

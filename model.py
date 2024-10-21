#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatedAttentionLayerV(nn.Module):
    '''
    $\text{tanh}\left(\boldsymbol{W}_{1}^\top \boldsymbol{H}_{i,j}\right)$ in Equation (2)
    '''

    def __init__(self, dim=512):
        super(GatedAttentionLayerV, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_V, b_V):
        out = F.linear(features, W_V, b_V)
        out_tanh = torch.tanh(out)

        return out_tanh


class GatedAttentionLayerU(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{2}^\top \boldsymbol{H}_{i,j} \right)$ in Equation (2)
    '''

    def __init__(self, dim=512):
        super(GatedAttentionLayerU, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_U, b_U):
        out = F.linear(features, W_U, b_U)
        out_sigmoid = torch.sigmoid(out)

        return out_sigmoid


class GatedAttention(nn.Module):
    def __init__(self, args):
        super(GatedAttention, self).__init__()
        self.args = args
        if self.args.ds in ["CRC_MIPL"]:
            self.L = 512
        else:
            self.L = 128
        self.D = 128
        self.K = 1
        self.nr_fea = self.args.nr_fea
        self.tau = args.init_tau
        self.decay_tau = args.decay_tau
        self.min_tau = args.min_tau

        if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(50 * 4 * 4, self.L),
                nn.Dropout(),
                nn.ReLU()
            )
        else:   # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.args.nr_fea, self.L),
                nn.Dropout(), 
                nn.ReLU()
            )
        # Equation (2):
        self.att_layer_V = GatedAttentionLayerV(self.L)
        self.att_layer_U = GatedAttentionLayerU(self.L)
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, self.args.nr_class)
        self.attention_weights = nn.Sequential(
            nn.Linear(self.args.nr_class, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        ) 

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.args.nr_class),
            nn.Sigmoid()
        )


    def decay_temperature(self):
        '''
        Equation (3)
        '''
        self.tau *= (1. - self.decay_tau)
        self.tau = max(self.tau, self.min_tau)


    def forward(self, X, args):
        if self.args.ds in ["MNIST_MIPL", "FMNIST_MIPL"]:
            X = X.squeeze(0)
            H = self.feature_extractor_part1(X)
            H = H.view(-1, 50 * 4 * 4)
            H = self.feature_extractor_part2(H)
        else:   # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets
            X = X.float()
            X = X.view(-1, args.nr_fea)
            H = self.feature_extractor_part2(X)  
        A_V = self.att_layer_V(H, self.linear_V.weight, self.linear_V.bias)
        A_U = self.att_layer_U(H, self.linear_V.weight, self.linear_V.bias)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        # Equation (4):
        A = (A - A.mean()) / (torch.std(A.detach()) + 1e-8)
        A = A / math.sqrt(self.L)
        A = F.softmax(A / self.tau, dim=1)
        Z = torch.mm(A, H)  # Equation (5)
        Y_logits = self.classifier(Z)

        return Y_logits, A
    

    def full_loss(self, A, prediction, target, margin_losses_per_epoch, args):
        '''
        the total loss function in Equation (9)
        '''
        # the dynamic disambiguation loss in Equation (6):
        Y_candiate = torch.zeros(target.shape).to(device)
        Y_candiate[target > 0] = 1
        prediction_can = prediction * Y_candiate
        new_prediction = prediction_can / prediction_can.sum(dim=1).repeat(prediction_can.size(1), 1).transpose(0, 1)
        d_loss = - torch.sum(target * torch.log(prediction))
        # d_loss = - target * torch.log(prediction)

        # the margin-compliant loss in Equation (10):
        prediction_non = prediction - prediction_can  
        can_p_top1 = torch.max(prediction_can)
        non_p_top1 = torch.max(prediction_non)
        margin_loss = args.mar_scale * torch.pow(1. - can_p_top1 + non_p_top1, 1.).reshape(-1).to(device)
        margin_losses_per_epoch = torch.cat((margin_losses_per_epoch, margin_loss))
        margin_loss_mean = margin_losses_per_epoch.mean(dim=0).requires_grad_(True)  
        margin_loss_std = margin_losses_per_epoch.std(dim=0, unbiased=True).requires_grad_(True)
        m_loss = margin_loss_mean / (1. - margin_loss_std) 

        loss =  d_loss + args.w_lambda * m_loss      # Equation (11)
        # loss =  torch.sum(d_loss) + args.w_lambda * m_loss      # Equation (11)

        return new_prediction, loss, margin_losses_per_epoch
    

    def calculate_objective(self, X, Y, margin_losses_per_epoch, args):
        '''
        calculate the full loss
        '''
        Y = Y.reshape(-1)
        Y_logits, A = self.forward(X, args)
        Y_logits = torch.clamp(Y_logits, min=1e-5, max=1.-1e-5)
        Y_prob = F.softmax(Y_logits, dim=1)
        new_prob, loss, margin_losses_per_epoch = self.full_loss(A, Y_prob, Y, margin_losses_per_epoch, args)

        return loss, new_prob, margin_losses_per_epoch


    def evaluate_objective(self, X, args):
        '''
        model testing
        '''
        Y_logits, _ = self.forward(X, args)
        Y_prob = F.softmax(Y_logits, dim=1)

        return Y_prob

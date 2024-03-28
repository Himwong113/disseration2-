#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True, mode ='cls '):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    loss =0
    if mode =='cls':
        print('enter classification mode')
        gold = gold.contiguous().view(-1)
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            n_class = pred.size(1)
            #print(gold)
            eps = 0.2
            one_hot = torch.eye(n_class, device=gold.device)[gold]
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            #print(f'one hot shape ={one_hot}')
            log_prb = F.softmax(pred, dim=1)
            #print(f'log pred ={log_prb}')
            loss = F.cross_entropy(log_prb ,one_hot, reduction='mean')
    elif(mode=='seg'):
        #print('enter partseg mode')
        if smoothing:
            eps = 0.2
            n_class = pred.size(-1)
            gold = gold.unsqueeze(-1)
            #print(f'gold ={gold.shape}')
            one_hot = torch.zeros_like(pred).scatter(1, gold, 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            #print(f'one hot ={one_hot.shape}')

            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            n_class = pred.size(1)
            # print(gold)
            eps = 0.2
            one_hot = torch.eye(n_class, device=gold.device)[gold]
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            # print(f'one hot shape ={one_hot}')
            log_prb = F.softmax(pred, dim=1)
            # print(f'log pred ={log_prb}')
            loss = F.cross_entropy(log_prb, one_hot, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
if __name__ == '__main__':
    gold = torch.randint(0,49,(32,2048))
    pred = torch.rand(32, 2048,50)
    print(gold.shape)
    print(f'squeeze = {gold.unsqueeze(-1).shape} ')
    print(pred.shape)

    loss = cal_loss(pred,gold,mode='seg')
    print(loss)
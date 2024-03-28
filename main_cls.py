#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model_cls import DGCNNCaps
import tqdm
#from model import PointNet , DGCNN
from DGCNN import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import visdom
from LatentspaceCapsule import LatentCapsLayer
import pandas as pd

def losshistory(log,epoch,name):
    #'checkpoints/%s/models/model.pt' % args.exp_name
    csv_file_path = f"checkpoints/{args.exp_name}/models/{name}_log_{epoch}.csv"

    # Create a DataFrame with an "iteration" column and "loss" column

    df = pd.DataFrame(log)

    if os.path.isfile(csv_file_path):
        # The file exists, so you can load it
        loaded_df = pd.read_csv(csv_file_path)

        # Concatenate the new data with the loaded data, preserving the index
        df = pd.concat([loaded_df, df], ignore_index=True)
        df.to_csv(csv_file_path, index=False)  # Save to the same file
    else:
        # The file doesn't exist, so save your DataFrame to the CSV file
        df.to_csv(csv_file_path, index=False)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py checkpoints'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model =='dgccaps':
        model = DGCNNCaps(args).to(device)

    else:
        raise Exception("Not implemented")
    #print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    fold_results = {'train_loss': [], 'train_accuracy': [], 'Avg_Train_Accuracy':[]}
    test_results = { 'test_loss': [], 'test_accuracy': [], 'Avg_Test_Accuracy': []}

    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        train_loss_temp = 0
        for data, label in tqdm.tqdm( train_loader):
            #print(f'data shape = {data.shape}')# batch 1024 3
            data, label = data.to(device), label.to(device).squeeze()
            #print(label.shape)
            data = data.permute(0, 2, 1)  # batch  3 102
            #print(f'data shape = {data.shape}')
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)




            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_loss_temp = loss.item()

            print(f'logit = {preds},label ={label} ')
            print(f'train loss is = {loss.item()}')
            if loss.item()<0:
                break


        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        print(f'train ture ={train_true},train ={train_pred}')
        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        test_loss_temp = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            test_loss_temp=loss.item()
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc_test = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc_test)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.pt' % args.exp_name)
        test_loss_temp =  test_loss*1.0/count



        fold_results['train_loss'].append(train_loss_temp)
        fold_results['train_accuracy'].append( train_acc)
        fold_results['Avg_Train_Accuracy'].append(avg_per_class_acc)

        test_results['test_loss'].append(test_loss_temp)
        test_results['test_accuracy'].append(test_acc)
        test_results['Avg_Test_Accuracy'].append(avg_per_class_acc_test)




        vis.scatter( torch.tensor([epoch]),torch.tensor([train_loss_temp ]), win='Train Loss', update='append',
                 name=f'Epoch_{epoch}', opts={'title': 'Train Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})
        vis.scatter(torch.tensor([epoch]), torch.tensor([test_loss_temp]), win='Test Loss', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'})

        vis.scatter( torch.tensor([epoch]),torch.tensor([train_acc]), win='Train Accuracy', update='append',
                 name=f'Epoch_{epoch}', opts={'title': 'Train Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})
        vis.scatter(torch.tensor([epoch]), torch.tensor([test_acc]), win='Test Accuracy', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Test Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})


        vis.scatter(torch.tensor([epoch]), torch.tensor([avg_per_class_acc]), win='Test  Accuracy',
                    update='append',name=f'Epoch_{epoch}', opts={'title': 'Avg Train Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})
        vis.scatter(torch.tensor([epoch]), torch.tensor([avg_per_class_acc_test]), win='Test  Accuracy', update='append',
                    name=f'Epoch_{epoch}', opts={'title': 'Avg Test Accuracy', 'xlabel': 'Epoch', 'ylabel': '%'})

        losshistory(fold_results,epoch,'train')
        losshistory(test_results, epoch,'test')


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    criterion = cal_loss
    test_loss =0
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        loss = criterion(logits, label)
        test_loss += loss.item()

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)



    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn','dgccaps'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()
    vis = visdom.Visdom(env=args.exp_name)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:44:19 2022

@author: clem
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models import WideResNet
from dataloaders import load_cifar
from solver import Solver
from utils import save_checkpoint
import pickle
import os

from tqdm import trange

#path = '/home/clem/Documents/Thesis/Datasets'
#checkpoint_path = '/home/clem/Documents/Thesis/checkpoints'

path = '/zhome/fa/5/117117/Thesis/Datasets'
checkpoint_path = '/zhome/fa/5/117117/Thesis/checkpoints'

dataset_choice = 'cifar10'
seed = 12
epochs = 400
batch_nb = 256
num_workers = 0

# Params WideResNet:
depth = 16          # minimum 10
widen_factor = 4
lr = 1e-5
weight_decay = 0

model_params = f'WideResNet-{depth}-{widen_factor}_MAP_lr_{lr}_btch_{batch_nb}_epochs_{epochs}_wd_{weight_decay}'

checkpoint_path = os.path.join(checkpoint_path, model_params)

assert dataset_choice == 'cifar10' or dataset_choice == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f'Start Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
#if not os.path.isdir(args.checkpoint):
#   mkdir_p(args.checkpoint)

# Data
train_loader, test_loader, num_classes = load_cifar(dataset_choice, path, batch_nb, num_workers)

# Model
model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, dropRate=0.0)

model = torch.nn.DataParallel(model).to(device)
cudnn.benchmark = True

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

solver = Solver(model, optimizer, criterion, device)

best_acc = 0  # best test accuracy

# Train and val
pbar = trange(0, epochs)
pbar.set_description(f'[Epoch: 0; LR: {lr:.4f}; ValAcc: N/A]')

metrics = {'train_losses': list(), 'test_losses': list(), 'train_accs': list(), 'test_accs': list()}

for epoch in pbar:
    #print(f'epoch: {epoch}')
    #print(f'Prior train Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
    train_loss, train_acc = solver.train(train_loader, epoch)
    #print(f'Post train Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
    test_loss, test_acc = solver.test(test_loader, epoch)
    
    metrics['train_losses'].append(train_loss)
    metrics['test_losses'].append(test_loss)
    metrics['train_accs'].append(train_acc)
    metrics['test_accs'].append(test_acc)

    # Telemetry
    pbar.set_description(f'[Epoch: {epoch+1}; LR: {lr:.4f}; ValAcc: {test_acc:.1f}]\n')

    #print('Got out of test, preceeding model saving')

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)

metrics_path = os.path.join('Run_metrics', model_params)
os.makedirs(metrics_path)
f = open(os.path.join(metrics_path,"metrics.pkl"),"wb")
pickle.dump(metrics,f)
f.close()

print('Best acc:')
print(best_acc)

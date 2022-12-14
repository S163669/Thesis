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
opt_choice = 'SGD'      # if Adam then Adam optimizer is used else SGD with Nesterov
torch.manual_seed = 12
epochs = 100
batch_size = 128
num_workers = 0
data_aug = True    # Use data augmentation
data_norm = False  # Normalize data

# Params WideResNet:
depth = 16          # minimum 10
widen_factor = 4
lr = 1e-1
weight_decay = 5e-4

if opt_choice == 'Adam':
    if dataset_choice == 'f-mnist':
        model_params = f'LeNet_MAP_Adam_lr_{lr}_btch_{batch_size}_epochs_{epochs}_wd_{weight_decay}_new_data_prep'
    else:
        model_params = f'WideResNet-{depth}-{widen_factor}_MAP_Adam_lr_{lr}_btch_{batch_size}_epochs_{epochs}_wd_{weight_decay}_new_data_prep'
else:
    lr_min = 1e-6
    if dataset_choice == 'f-mnist':
        model_params = f'LeNet_MAP_SGDNesterov_lr_{lr}_lr_min_{lr_min}_btch_{batch_size}_epochs_{epochs}_wd_{weight_decay}_new_data_prep'
    else:
        model_params = f'WideResNet-{depth}-{widen_factor}_MAP_SGDNesterov_lr_{lr}_lr_min_{lr_min}_btch_{batch_size}_epochs_{epochs}_wd_{weight_decay}_new_data_prep'

checkpoint_path = os.path.join(checkpoint_path + f'/{dataset_choice}', model_params)

assert dataset_choice == 'cifar10' or dataset_choice == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f'Start Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
#if not os.path.isdir(args.checkpoint):
#   mkdir_p(args.checkpoint)

# DIFFERENCE HERE: BATCH SIZE IS SET TO 128 BUT IN PAPER 512
train_loader, val_loader, test_loader, num_classes = load_cifar(dataset_choice, path, batch_size, num_workers, batch_size_val=batch_size,
                                                                val_size=2000, data_augmentation=data_aug, normalize=data_norm)

# Model# DIFFERENCE HERE: BATCH SIZE IS SET TO 128 BUT IN PAPER 512
model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, dropRate=0.0)

# DIIFFERENCE HERE: THE TWO CONSECUTIVE LINES ARE NOT USED IN PAPER CODE (REMEMBER TO INCLUDE TO DEVICE IF REMOVED)
model = torch.nn.DataParallel(model).to(device)
cudnn.benchmark = True

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
# DIFFERENCE HERE: nn.CrossEntropyLoss() USED INSTEAD OF F.cross_entropy
criterion = nn.CrossEntropyLoss()

if opt_choice == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    n_steps = epochs * len(train_loader)
    # DIFFERENCE HERE, NO eta_min IN PAPER CODE
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
# DIFFERENCE HERE IN SOLVE.TRAIN amp.GradScaler() IS USED IN PAPER
solver = Solver(model, optimizer, criterion, device, scheduler=scheduler)

best_acc = 0  # best test accuracy

# Train and val
pbar = trange(0, epochs)
pbar.set_description(f'[Epoch: 0; LR: {lr:.4f}; ValAcc: N/A]')

metrics = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': [] ,'test_acc': float}

for epoch in pbar:
    #print(f'epoch: {epoch}')
    #print(f'Prior train Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
    train_loss, train_acc = solver.train(train_loader)
    #print(f'Post train Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
    val_loss, val_acc = solver.test(val_loader)
    
    metrics['train_losses'].append(train_loss)
    metrics['val_losses'].append(val_loss)
    metrics['train_accs'].append(train_acc)
    metrics['val_accs'].append(val_acc)

    # Telemetry
    pbar.set_description(f'[Epoch: {epoch+1}; LR: {lr:.4f}; ValAcc: {val_acc:.2f}]\n')

    #print('Got out of test, preceeding model saving')

    # save model
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)

_ , test_acc = solver.test(test_loader) 
metrics['test_acc'] = test_acc
metrics_path = os.path.join('Run_metrics' + f'/{dataset_choice}', model_params)

if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)
    
f = open(os.path.join(metrics_path,"metrics.pkl"),"wb")
pickle.dump(metrics,f)
f.close()

print(f'Best acc: {best_acc}')
print(f'Test acc final: {test_acc}')

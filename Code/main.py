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

from tqdm import trange

torch.cuda.is_available()

#path = '/home/clem/Documents/Thesis/Datasets'
#checkpoint_path = '/home/clem/Documents/Thesis/checkpoints'

path = '/zhome/fa/5/117117/Thesis/Datasets'
checkpoint_path = '/zhome/fa/5/117117/Thesis/checkpoints'

dataset_choice = 'cifar10'
seed = 12
epochs = 2
batch_nb = 256
num_workers = 0

# Params WideResNet:
depth = 16          # minimum 10
widen_factor = 4
lr = 1e-4
weight_decay = 0


assert dataset_choice == 'cifar10' or dataset_choice == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#if not os.path.isdir(args.checkpoint):
#   mkdir_p(args.checkpoint)

# Data
train_loader, test_loader, num_classes = load_cifar(dataset_choice, path, batch_nb, num_workers)

# Model
model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, dropRate=0.0)

#model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

solver = Solver(model, optimizer, criterion, device)

best_acc = 0  # best test accuracy

# Train and val
pbar = trange(0, epochs)
pbar.set_description(f'[Epoch: 0; LR: {lr:.4f}; ValAcc: N/A]')

for epoch in pbar:
    print(f'epoch: {epoch}')
    train_loss, train_acc = solver.train(train_loader, epoch)
    test_loss, test_acc = solver.test(test_loader, epoch)

    # Telemetry
    pbar.set_description(f'[Epoch: {epoch+1}; LR: {lr:.4f}; ValAcc: {test_acc:.1f}]')

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

print('Best acc:')
print(best_acc)

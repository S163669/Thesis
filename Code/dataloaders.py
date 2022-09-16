#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:25:43 2022

@author: clem
"""

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def load_cifar(dataset_choice, path, batch_nb, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if dataset_choice == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataset_choice = datasets.CIFAR100
        num_classes = 100
    
    trainset = dataloader(root=path, train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=batch_nb, shuffle=True, num_workers=num_workers)
    
    testset = dataloader(root=path, train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=batch_nb, shuffle=False, num_workers=num_workers)
    
    return(train_loader, test_loader, num_classes)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:25:43 2022

@author: clem
inspired from: https://github.com/wiseodd/pytorch-classification/blob/master/cifar.py
https://github.com/wiseodd/laplace-redux/blob/cd208c44149dbb093e9107cf0d6fde7c9f106027/utils/data_utils.py#L164

https://github.com/runame/laplace-redux/blob/main/utils/data_utils.py

Exchanged order of transforms.RandomHorizontalFlip() and transforms.RandomCrop(32, padding=4) in new version + 
changed numbers for normalization to the ones from last link
"""

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch

def load_cifar(dataset_choice, path, batch_size="fullbatch", num_workers=0, batch_size_val="fullbatch", val_size=2000, data_augmentation=True, normalize=True):
    #batch_size_val was set to 512
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    transform_base = [transforms.ToTensor()]
    if normalize:
        transform_base.append(transforms.Normalize(mean, std))
    if data_augmentation:
        train_transforms = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),] + transform_base
    else:
        train_transforms = transform_base
    
    transform_train = transforms.Compose(train_transforms)
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_test = transforms.Compose(transform_base)
    if dataset_choice == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
    
    trainset = dataloader(root=path, train=True, download=True, transform=transform_train)
    
    if batch_size == "fullbatch":
        train_loader = data.DataLoader(trainset, batch_size=trainset.__len__(), shuffle=True, num_workers=num_workers)
    else:
        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    valset = dataloader(root=path, train=False, download=False, transform=transform_test)
    
    if val_size:
        
        val_loader, test_loader = val_test_split(valset, batch_size=batch_size_val, val_size=val_size, num_workers=num_workers)
        return(train_loader, val_loader, test_loader, num_classes)
    
    elif batch_size == "fullbatch":
        test_loader = data.DataLoader(valset, batch_size=valset.__len__(), shuffle=True, num_workers=num_workers)
        
    else:
        test_loader = data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #test_loader = data.DataLoader(valset, batch_size=valset.__len__(), shuffle=True, num_workers=num_workers)
    return(train_loader, test_loader, num_classes)


def val_test_split(dataset, val_size=5000, batch_size=512, num_workers=0, pin_memory=False):
    
    # Split into val and test sets
    test_size = len(dataset) - val_size
    dataset_val, dataset_test = data.random_split(
        dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42))
    
    if batch_size == "fullbatch":
        
        val_loader = data.DataLoader(dataset_val, batch_size=dataset_val.__len__(), shuffle=False,
                                       num_workers=num_workers, pin_memory=pin_memory)
    
        test_loader = data.DataLoader(dataset_test, batch_size=dataset_test.__len__(), shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory)
        
    else:
        
        val_loader = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin_memory)
    
        test_loader = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory)
        
    return val_loader, test_loader
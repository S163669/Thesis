#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:34:37 2022

@author: clem
"""

import torch

class Solver():
    
    def __init__(self, model, optimizer, criterion, device):
        
        self.device = device
        self.model=model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        
    def train(self, trainloader, epoch):
        # switch to train mode
        
        losses = list()
        self.model.train()
        nb_obs = 0
        true_class = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            preds = torch.argmax(outputs, dim=1)
            true_class += torch.sum(preds == targets)
            nb_obs += len(targets)
            
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.cpu())
        
        mean_loss = torch.mean(torch.stack(losses))
        accuracy = true_class/nb_obs
        
        return mean_loss, accuracy

    def test(self, testloader, epoch):
        print('entering test')
        losses = list()
        nb_obs = 0
        true_class = 0
        
        # switch to evaluate mode
        self.model.eval()

        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            losses.append(loss.cpu())
            
            preds = torch.argmax(outputs, dim=1)
            true_class += torch.sum(preds == targets)
            nb_obs += len(targets)

        mean_loss = torch.mean(torch.stack(losses))
        accuracy = true_class/nb_obs
        
        return mean_loss, accuracy
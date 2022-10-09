#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:34:37 2022

@author: clem
"""

import torch

class Solver():
    
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        
        self.device = device
        self.model=model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.scheduler = scheduler
        
    def train(self, trainloader):
        # switch to train mode
        losses = []
        self.model.train()
        nb_obs = 0
        true_class = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            preds = torch.argmax(outputs, dim=1)
            true_class += torch.sum(preds == targets).item()
            nb_obs += len(targets)
            
            #print(f'Post batch train Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
            
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            losses.append(loss.cpu().item())
            
            #print(f'Post backprop batch train Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')
            
        mean_loss = sum(losses)/len(losses)
        accuracy = true_class/nb_obs
        
        return mean_loss, accuracy

    def test(self, testloader):
        #print('entering test')
        losses = []
        nb_obs = 0
        true_class = 0
        
        with torch.no_grad():
            # switch to evaluate mode
            self.model.eval()
    
            for batch_idx, (inputs, targets) in enumerate(testloader):
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
    
                # compute output
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss.cpu().item())
                
                preds = torch.argmax(outputs, dim=1)
                true_class += torch.sum(preds == targets).item()
                nb_obs += len(targets)
                #print(f'Post batch test Memory: {torch.cuda.mem_get_info(0)[0]/1048576} MB / {torch.cuda.mem_get_info(0)[1]/1048576} MB \n')

        mean_loss = sum(losses)/len(losses)
        accuracy = true_class/nb_obs
        
        return mean_loss, accuracy
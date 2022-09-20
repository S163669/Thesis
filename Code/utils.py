#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:13:12 2022

@author: clem
"""

import os
import torch
import shutil
import matplotlib.pyplot as plt

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pt'):
    
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pt'))
        
        
def plot_metrics(metrics, lst, group_train_test=True, savefig=True):
    
    epochs = range(1,len(metrics[lst[0]])+1)
    
    if group_train_test:
        itr = len(lst)//2
        
    for i in itr:
        
        plt.figure()
        plt.plot(epochs, metrics[lst[2*i]], label=f'{lst[2*i]}')
        plt.plot(epochs, metrics[lst[2*i+1]], label=f'{lst[2*i+1]}')
        plt.xlabel(epochs)
        if lst[2*i][:-3] == 'acc':
            plt.ylabel('Accuracy')
        else:
            plt.ylabel('Loss')
        plt.legend()
        if savefig:
            plt.savefig('loss_plot.pdf', bbox_inches='tight', format='pdf')
        else:
            plt.show()
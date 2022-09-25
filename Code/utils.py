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
    
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pt'))
        
        
def plot_metrics(metrics, lst, title, savefig=True):
    
    epochs = list(range(1,len(metrics[lst[0]])+1))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    fig.suptitle(f'{title}')
    ax1.plot(epochs, metrics[lst[0]], label=f'{lst[0]}')
    ax1.plot(epochs, metrics[lst[1]], label=f'{lst[1]}')
    ax2.plot(epochs, metrics[lst[2]], label=f'{lst[2]}')
    ax2.plot(epochs, metrics[lst[3]], label=f'{lst[3]}')
    ax1.set(xlabel='epochs', ylabel='Loss')
    ax2.set(xlabel='epochs', ylabel='Accuracy')
    ax1.legend()
    ax2.legend()
    
    if savefig:
        plt.savefig(f'/home/clem/Documents/Thesis/Figures/{title}.pdf', bbox_inches='tight', format='pdf')
    else:
        plt.show()

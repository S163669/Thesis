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
        
def plot_prior_vs_posterior_weights_pred(hmc_samples, data, labels, num_classes):
    
    hmc_samples = hmc_samples['prior_samples']
    pad_1 = torch.nn.ConstantPad1d((0,1), 1)
    data = pad_1(data)
    
    nb_samples = hmc_samples.size()[0]
    dim = data.size()[1]*num_classes
    #coefs_mean = mean*torch.ones(len(var))
    coefs_mean = torch.zeros(dim)
    prior_samples = torch.distributions.normal.Normal(coefs_mean, ((1/40)**1/2)*torch.ones(dim)).sample(torch.Size([nb_samples]))  #Precision of 40 in paper
    
    plt.figure()
    plt.title('Mean coefficient value prior samples vs hmc samples')
    plt.plot(range(dim), prior_samples.mean(0), '.', label='prior samples mean')
    plt.plot(range(dim), hmc_samples.mean(0), '.', alpha=0.3, label='hmc samples mean')
    plt.xlabel('coefficient number')
    plt.ylabel('coefficient mean value')
    plt.legend()
    
    if not os.path.exists('./Figures'):
        os.makedirs('./Figures')
    
    plt.savefig('Figures/hmc_coefs_vs_prior_coeffs.pdf', bbox_inches='tight', format='pdf')
    
    prior_probs = []
    hmc_probs = []
    
    for i in range(nb_samples):
        
        prior_prob = torch.softmax(data @ prior_samples[i].reshape(-1,10), dim=1)
        hmc_prob = torch.softmax(data @ hmc_samples[i].reshape(-1,10), dim=1)
        
        prior_probs.append(prior_prob[range(len(labels)), labels])
        hmc_probs.append(hmc_prob[range(len(labels)), labels])
    
    prior_prob_mean = torch.mean(torch.stack(prior_probs), dim=0)
    hmc_prob_mean = torch.mean(torch.stack(hmc_probs), dim=0)

    plt.figure()
    plt.title('Probability mean of last layer weight from prior vs from hmc')
    plt.plot(torch.linspace(0,1,100),torch.linspace(0,1,100), label='reference line') 
    plt.plot(prior_prob_mean, hmc_prob_mean, '.', label='data samples')
    plt.xlabel('prior weights probability mean')
    plt.ylabel('hmc weihgts probability mean)')
    plt.legend()
    plt.savefig('Figures/prob_mean_hmc_coeffs_vs_prior_coeffs.pdf', bbox_inches='tight', format='pdf')
    
    print(f'Probability mean for prior weight samples {torch.mean(prior_prob_mean).item()}')
    print(f'Probability mean for hmc weight samples {torch.mean(hmc_prob_mean).item()}')
    
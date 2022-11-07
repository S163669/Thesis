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
import pyro
import pyro.distributions as dist
from netcal.metrics import ECE
import torch.nn.functional as F

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
    


def predict(testloader, device, model, using_laplace=False, link_approx='mc', n_samples=20):

    preds = []    
    targets = []
    
    with torch.no_grad():
        for inputs, ys in testloader:
            
            inputs = inputs.to(device)
    
            if using_laplace:
                preds.append(model(inputs, link_approx=link_approx, n_samples=n_samples))
            else:
                model.eval()
                preds.append(torch.softmax(model(inputs), dim=-1))
            
            targets.append(ys)
    
    targets = torch.cat(targets, dim=0)
    preds = torch.cat(preds).cpu()
    
    return preds, targets


def get_act_Lm1(model, data_loader, device, integrated_in_model=True):
    
    if not integrated_in_model:
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        model.module.bn1.register_forward_hook(get_activation('bn1'))
        
    data = []
    ys = []
    
    with torch.no_grad():
        model.eval()
        for x, y in data_loader:
            
            if integrated_in_model:  
                out = model.get_lhl_act(x.to(device))
                data.append(out)
            else:    
                output = model(x.to(device))
                data.append(activation['bn1'])
            ys.append(y) 
    
    data = torch.cat(data, dim=0)
    ys = torch.cat(ys)
    
    if not integrated_in_model:
        data = F.avg_pool2d(data, 8)
        data = data.view(-1, 64*4).cpu()
    
    return data, ys
    

def model_hmc(data, num_classes, labels, prec):
    
    pad_1 = torch.nn.ConstantPad1d((0,1), 1)
    data = pad_1(data)
    
    dim = data.size()[1]*num_classes
    coefs_mean = torch.zeros(dim)
    coefs = pyro.sample('ll_weights', dist.Normal(coefs_mean, ((1/prec)**1/2)*torch.ones(dim)))  #Precision of 40 in paper

    y = pyro.sample('y', dist.Categorical(logits=data @ coefs.reshape(-1,10)), obs=labels)
    
    return y

def predict_Lm1(coeffs, x, y):
        
    pad_1 = torch.nn.ConstantPad1d((0,1), 1)
    # activation of layer L-1 with padded 1
    x = pad_1(x)
    # (observations x number of classes)
    class_prob = torch.softmax(x @ coeffs.reshape(-1,10), dim=1)
    y_pred = torch.argmax(class_prob, 1)
    
    acc = sum(y_pred == y)/len(y)
    
    #probs_true = class_prob[range(len(y)),y]
    
    nll_hmc = -torch.distributions.Categorical(class_prob).log_prob(y).mean()
    
    return acc.item(), class_prob, nll_hmc


def metrics_ll_weight_samples(samples, x, y):
    
    accs = []
    sum_class_probs = 0
    
    for coeff in samples:
        
        acc, class_probs, _  = predict_Lm1(coeff, x, y)
        accs.append(acc)
        # sum of (observation x number of classes) for each HMC sample
        sum_class_probs += class_probs
        
    # Calculating 1/S*Sum_1^Ŝ(p(y* | f(x*)))) for each x* and taking log to get log(p(y* |x*, D)) 
    # then taking negative of the mean for each x*.
    nll = -torch.mean(torch.log(sum_class_probs[range(len(y)),y]/len(samples))).item()
    ece = ECE(bins=15).measure(class_probs.numpy(), y.numpy())
    
    return sum(accs)/len(accs), ece, nll
        
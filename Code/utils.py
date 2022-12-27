#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:13:12 2022

@author: clem
"""

import os
import math
import torch
import shutil
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from netcal.metrics import ECE
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import pdist
import pickle
import re
from dataloaders import load_cifar

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
                out = model.module.get_lhl_act(x.to(device))
                data.append(out)
            else:
                output = model(x.to(device))
                data.append(activation['bn1'])
            ys.append(y) 
    
    data = torch.cat(data, dim=0)
    ys = torch.cat(ys, dim=0)
    
    if not integrated_in_model:
        data = F.avg_pool2d(data, 8)
        data = data.view(-1, 64*4)
    
    return data.cpu(), ys
    

def model_hmc(data, num_classes, labels, prec):
    
    #pad_1 = torch.nn.ConstantPad1d((0,1), 1)
    #data = pad_1(data)
    num_features = data.shape[1]
    dim = (num_features + 1)*10
    #dim = data.size()[1]*num_classes
    coefs_mean = torch.zeros(dim)
    # Added to_event(1) to make samples dependent.
    coefs = pyro.sample('ll_weights', dist.Normal(coefs_mean, math.sqrt(1/prec)).to_event(1))  #Precision of 40 in paper
    
    act_w = coefs[:num_features*num_classes].reshape(num_classes, num_features)
    bias_w = coefs[num_features*num_classes:]
    
    y = pyro.sample('y', dist.Categorical(logits=data @ act_w.T + bias_w), obs=labels)
    
    return y


def predict_Lm1(coefs, x, y, num_classes):
    
    #pad_1 = torch.nn.ConstantPad1d((0,1), 1)
    # activation of layer L-1 with padded 1
    #x = pad_1(x)
    # (observations x number of classes)
    #class_prob = torch.softmax(x @ coefs.reshape(-1,10), dim=1)
    num_features = x.shape[1]
    # The below is matching the way pytorch.nn.utils.parameters_to_vector is arranging the parameters
    act_w = coefs[:num_features*num_classes].reshape(num_classes, num_features)
    bias_w = coefs[num_features*num_classes:]
    
    class_prob = torch.softmax(x @ act_w.T + bias_w, dim=1)
    
    y_pred = torch.argmax(class_prob, 1)
    
    acc = sum(y_pred == y)/len(y)
    
    #probs_true = class_prob[range(len(y)),y]
    
    nll_hmc = -torch.distributions.Categorical(class_prob).log_prob(y).mean()
    
    return acc.item(), class_prob, nll_hmc


def multisample_ll_prediction(samples, x, y, num_classes):
    
    sum_class_probs = 0
    
    for coeff in samples:
        
        _ , class_probs, _  = predict_Lm1(coeff, x, y, num_classes)
        # sum of (observation x number of classes) for each HMC sample
        sum_class_probs += class_probs
        
    final_class_probs = sum_class_probs/len(samples)
    y_pred = torch.argmax(final_class_probs, 1)
    
    acc = sum(y_pred == y).item()/len(y)
    
    return final_class_probs, acc
    
        
def metrics_ll_weight_samples(samples, x, y, num_classes):
    
        final_class_probs, acc = multisample_ll_prediction(samples, x, y, num_classes)
        
        # Calculating 1/S*Sum_1^Ŝ(p(y* | f(x*)))) for each x* and taking log to get log(p(y* |x*, D)) 
        # then taking negative of the mean for each x*.
        nll = -torch.mean(torch.log(final_class_probs[range(len(y)),y])).item()
        ece = ECE(bins=15).measure(final_class_probs.numpy(), y.numpy())
        
        return acc, ece, nll


def mmd_rbf(X, Y):
    """
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2)).
    Source: https://github.com/jindongwang/transferlearning
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    # Median heuristic --- use some hold-out samples
    all_samples = np.concatenate([X[:50], Y[:50]], 0)
    pdists = pdist(all_samples)
    sigma = np.median(pdists)
    gamma=1/(sigma**2)

    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


def hmc_mmd_uppermatrix(lst):
    
    mmd_distances = []
    n = len(lst)
    
    for i in range(n):
        for j in range((i+1),n):
            mmd_distances.append(mmd_rbf(lst[i], lst[j]))
            
    return mmd_distances


def get_mmds(dataset_choice, model_choice):
    
    with open(f'./Run_metrics/{dataset_choice}/{model_choice}/results_metrics.pkl', 'rb') as f:
        results = pickle.load(f)
    
    hmc_samples = torch.load(f'./Run_metrics/{dataset_choice}/{model_choice}/hmc_samples')['ll_weights'].cpu().numpy()
    la_samples = torch.load(f'./Run_metrics/{dataset_choice}/{model_choice}/la_samples').cpu().numpy()
    map_sample = torch.load(f'./checkpoints/{dataset_choice}/{model_choice}/checkpoint.pt')['state_dict']
    
    #params = ((k.partition('module.')[2], map_sample[k]) for k in map_sample.keys())
    
    map_sample = torch.load(f'./Run_metrics/{dataset_choice}/{model_choice}/map_sample').cpu().numpy()
    map_sample = np.repeat(map_sample.reshape(1,-1), 600, axis=0)
    
    mmd_map = mmd_rbf(map_sample, hmc_samples)
    mmd_la = mmd_rbf(la_samples, hmc_samples)
    
    results['map']['mmd'] = mmd_map
    results['la']['mmd'] = mmd_la
    
    print(f"MMD MAP: {mmd_map}")
    print(f"MMD LA: {mmd_la}")
    
    samples = [filename for filename in os.listdir(f'./Run_metrics/{dataset_choice}/{model_choice}') if filename.startswith('refined_posterior_samples')]
    for sample in samples:
        
        nf_samples = torch.load(f'./Run_metrics/{dataset_choice}/{model_choice}/{sample}').cpu().numpy()
        mmd_ref_nf = mmd_rbf(nf_samples, hmc_samples)
        
        m = re.search('refined_posterior_samples_(.*)_(\d*)', sample)
        
        results[f'ref_nf_{m.group(1)}_{m.group(2)}']['mmd'] = mmd_ref_nf
        print(f"MMD NF refined {m.group(1)}-{m.group(2)}: {mmd_ref_nf}")
        
    with open(f'./Run_metrics/{dataset_choice}/{model_choice}/results_metrics.pkl', 'wb') as f:
        pickle.dump(results, f)


def compute_performance(dataset, base_model):
    
    runs = [filename for filename in os.listdir(f'./Run_metrics/{dataset}') if filename.startswith(base_model) and len(filename) <= len(base_model) + 3]
    nb_runs = len(runs)
    
    all_runs = 0
    list_hmc_samples = []
    
    for run in runs:
        
        with open(f'./Run_metrics/{dataset}/{run}/results_metrics.pkl', 'rb') as f:
            results = pickle.load(f)
            
            if not all_runs:
                
                keys = list(results.keys())
                all_runs = dict(zip(keys, [{}]*len(keys)))
                
                for key in keys:
                    
                    keys_in = list(results[key].keys())
                    all_runs[key] = dict(zip(keys_in, [[v] for v in results[key].values()]))
                    
            else:
                
                for key in all_runs.keys():
                    
                    for key_in in all_runs[key].keys():
                        
                        all_runs[key][key_in].append(results[key][key_in])
            
            list_hmc_samples.append(torch.load(f'./Run_metrics/{dataset}/{run}/hmc_samples')['ll_weights'].cpu().numpy())
    
    # Get all possible mmds between hmc samples of each run to have it as a baseline
    all_runs['hmc']['mmd'] = hmc_mmd_uppermatrix(list_hmc_samples)
    
    with open(f'./Run_metrics/{dataset}/{base_model}_all_results_metrics.pkl', 'wb') as f:
        pickle.dump(all_runs, f)
    
    with open(f'./Run_metrics/{dataset}/{base_model}_summary.txt', 'w') as f:
        
        for key in keys:
            f.write(f'{key}\n')
            f.write(f'{list(all_runs[key].keys())}\n')
            for key_in in all_runs[key].keys():
                f.write(f'{np.round(np.mean(all_runs[key][key_in]), 4)} $\pm$ {np.round(np.std(all_runs[key][key_in])/np.sqrt(nb_runs), 4)}\t')
            f.write('\n')

def string_num_sort(string):
    return list(map(int, re.findall(r'\d+', string)))[0]

def get_ece_baseline(dataset, repetitions=10):
    
    basepath = '/zhome/fa/5/117117/Thesis/'
    _ , _ , test_loader, num_classes = load_cifar(dataset, basepath + 'Datasets', 0, val_size=2000, data_augmentation=False)
    
    _, y = next(iter(test_loader))
    d = torch.distributions.dirichlet.Dirichlet(torch.ones(num_classes,))
    
    eces = [ECE(bins=15).measure(d.sample(y.size()).numpy(), y.numpy()) for _ in range(repetitions)]   
    
    return np.mean(eces), np.std(eces)/np.sqrt(repetitions), eces
    
    
def plot_flow_performance(dataset, base_model):
    
    if not os.path.isfile(f'./Run_metrics/{dataset}/{base_model}_all_results_metrics.pkl'):
        compute_performance(dataset, base_model)
    
    if not os.path.exists(f'./Figures/{base_model}/'):
        os.makedirs(f'./Figures/{base_model}/')
    
    with open(f'./Run_metrics/{dataset}/{base_model}_all_results_metrics.pkl', 'rb') as f:
        results = pickle.load(f)
        
        dic = {}
        for key in results:
            
            m = re.search('ref_nf_(.*)_(\d*)', key)
            if m:
                if m.group(1) not in dic.keys():
                
                    dic[m.group(1)] = {'flow_lens': [int(m.group(2))], 'samples': [m.group(0)]}
            
                else:
                
                    dic[m.group(1)]['flow_lens'].append(int(m.group(2)))
                    dic[m.group(1)]['samples'].append(m.group(0))
                    
        
        
        for metric in results['map'].keys():
            
            plt.figure()
            
            for flow_type in dic.keys():
                
                dic[flow_type]['flow_lens'].sort()
                dic[flow_type]['samples'].sort(key=string_num_sort)
                
                y = [results[nf][metric] for nf in dic[flow_type]['samples']]
                y = np.asarray(y)
                x = dic[flow_type]['flow_lens']
                
                plt.errorbar(x, np.mean(y, axis=1), yerr=np.std(y, axis=1)/np.sqrt(len(x)), marker='.', label=flow_type)
            
            base_list = ['map', 'la', 'hmc']
                
            for base in base_list:
                plt.errorbar(x, [np.mean(results[base][metric])]*len(x), yerr=[np.std(results[base][metric])/np.sqrt(len(x))]*len(x), label=base, alpha=0.5, marker='.', linestyle='--')
            
            plt.xlabel('Length of normalizing flow')
            plt.ylabel(metric.upper()+'.')
            plt.legend()
            plt.show()
            plt.savefig(f'./Figures/{base_model}/{metric.upper()}_vs_flow_len.pdf', bbox_inches='tight', format='pdf')
        
            
def plot_swag_validation(data_norm):
    
    basepath = '/zhome/fa/5/117117/Thesis/'
    with open(basepath + 'Run_metrics/swag_grid_search_results_norm={data_norm}.pkl', 'rb') as f:
        results = pickle.load(f)
        
    lst = list(results.keys())
    lst.remove('opt_params')
    lst.remove('best_nll')
    lst.sort(key=string_num_sort)
    
    style = {'0.01': '-',
             '0.001': '--',
             '0.0001':'-.',
             '1e-05': ':',
             '25': 'C5',
             '50': 'C0',
             '100': 'C1',
             '200': 'C2',
             '400': 'C3',
             '600': 'C4'}
    
    metrics = 0
    dic = {}
    for key in lst:
        
        m = re.search('epoch_(\d*)_K_(\d*)_lr_(\d*\.\d*)', key)
        if m:
            # if the K (rank) isn't present in dictionary.
            if m.group(2) not in dic.keys():
            
                dic[m.group(2)] ={m.group(3): {'epochs': [int(m.group(1))], 'acc': [results[m.group(0)]['acc']],
                                               'ece': [results[m.group(0)]['ece']], 'nll': [results[m.group(0)]['nll']]}}
                
                if not metrics:
                    metrics = list(results[m.group(0)].keys())
                
            # if K (rank) is present but not the learning rate.
            elif m.group(3) not in dic[m.group(2)].keys():
                
                dic[m.group(2)][m.group(3)] = {'epochs': [int(m.group(1))], 'acc': [results[m.group(0)]['acc']],
                                               'ece': [results[m.group(0)]['ece']], 'nll': [results[m.group(0)]['nll']]}
            # if K and learning rate are all present already.    
            else:
                dic[m.group(2)][m.group(3)]['epochs'].append(int(m.group(1)))
                dic[m.group(2)][m.group(3)]['acc'].append(results[m.group(0)]['acc'])
                dic[m.group(2)][m.group(3)]['ece'].append(results[m.group(0)]['ece'])
                dic[m.group(2)][m.group(3)]['nll'].append(results[m.group(0)]['nll'])
                
    for metric in metrics:
        
        plt.figure()
        
        for K in dic.keys():
            for lr in dic[K].keys():
                plt.plot(dic[K][lr]['epochs'], dic[K][lr][metric], color=style[K], linestyle=style[lr], label=f'K={K}, lr={lr}')
        
        plt.scatter(results['opt_params']['epoch'], results[f"epoch_{results['opt_params']['epoch']}_K_{results['opt_params']['K']}_lr_{results['opt_params']['lr']}"][metric],
                 marker='*', color=style[f"{results['opt_params']['K']}"], edgecolor='b', label='optimum')
        plt.xlabel('number of epochs')
        plt.ylabel(metric.upper()+'.')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'./Figures/Swag/{metric}_vs_epoch-datanorm={data_norm}.pdf', bbox_inches='tight', format='pdf')
        plt.show()
                    
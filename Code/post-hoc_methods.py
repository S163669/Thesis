#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:25:13 2022

@author: clem
"""

from laplace import Laplace
import torch
from models import WideResNet
from dataloaders import load_cifar
from netcal.metrics import ECE
from utils import predict, plot_prior_vs_posterior_weights_pred, model_hmc, get_activation, metrics_hmc_samples
import pyro
import torch.nn.functional as F


#basepath = '/home/clem/Documents/Thesis/'
basepath = '/zhome/fa/5/117117/Thesis/'
do_map = True
do_laplace = True
do_hmc = True

dataset_choice = 'cifar10'
torch.manual_seed = 12
batch_nb = 128
num_workers = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WideResNet(depth=16, num_classes=10, widen_factor=4, dropRate=0.0)
model = torch.nn.DataParallel(model).to(device)

#checkpoint_bestmodel = torch.load('/home/clem/Documents/Thesis/checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_16_epochs_150_wd_0.0005/model_best.pt')
checkpoint_bestmodel = torch.load(basepath + 'checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_128_epochs_100_wd_0.0005_new_data_prep_5/checkpoint.pt')

model.load_state_dict(checkpoint_bestmodel['state_dict'])

train_loader, val_loader, test_loader, num_classes = load_cifar(dataset_choice, basepath + 'Datasets', batch_nb, num_workers, batch_size_val=batch_nb, val_size=2000)

if do_map:
    
    probs_map, targets_map = predict(test_loader, device, model)
    ece_map = ECE(bins=15).measure(probs_map.numpy(), targets_map.numpy())
    acc_map = sum(torch.argmax(probs_map, 1)==targets_map)/len(targets_map)
    nll_map = -torch.distributions.Categorical(probs_map).log_prob(targets_map).mean()
    print(f'[MAP] Acc.: {acc_map}; ECE: {ece_map:.1%}; NLL: {nll_map}')


if do_laplace:
    la = Laplace(model, 'classification',
                 subset_of_weights='last_layer',
                 hessian_structure='full')
    
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik', link_approx='mc')
    
    probs_laplace, targets = predict(test_loader, device, la, using_laplace=True)
    acc_laplace = (probs_laplace.numpy().argmax(-1) == targets.numpy()).astype(int).mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -torch.distributions.Categorical(probs_laplace).log_prob(targets).mean()
    
    la_samples = la.sample(600)
    torch.save(la_samples, './Run_metrics/la_samples')

    print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')
    #print(f"Mean LA samples {la_samples.mean(0)}")

if do_hmc:
    activation = {}
    
    model.module.bn1.register_forward_hook(get_activation('bn1'))
    
    acts = []
    ys = []
    
    with torch.no_grad():
        model.eval()
        for x, y in train_loader:
    
            output = model(x.to(device))
            acts.append(activation['bn1'])
            ys.append(y) 
    
    acts = torch.cat(acts, dim=0)
    ys = torch.cat(ys)
    
    data = F.avg_pool2d(acts, 8)
    data = data.view(-1, 64*4).cpu()
    
    nuts_kernel = pyro.infer.NUTS(model_hmc)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=600, warmup_steps=300)
    
    mcmc.run(data, num_classes, ys)

    hmc_samples = mcmc.get_samples(600)
    torch.save(hmc_samples, './Run_metrics/hmc_samples')
    
    data_test = torch.load(basepath + '/Datasets/output_data_L-1_test')
    metrics_hmc_samples(hmc_samples['ll_weights'], data_test)

    #print(f"Mean HMC samples {hmc_samples['ll_weights'].mean(0)}")

#plot_prior_vs_posterior_weights_pred(hmc_samples, data, ys, num_classes)








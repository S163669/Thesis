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
from utils import predict, plot_prior_vs_posterior_weights_pred, model_hmc, metrics_hmc_samples, get_act_Lm1
import pyro
from models import Normalizing_flow
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
import matplotlib.pyplot as plt

#basepath = '/home/clem/Documents/Thesis/'
basepath = '/zhome/fa/5/117117/Thesis/'
do_map = True
do_laplace = True
do_hmc = True
do_posterior_refinemenent = True
make_plots = True

precs_prior_hmc = [1,2,4,8,16,32,64]

dataset_choice = 'cifar10'
torch.manual_seed = 12
batch_nb = 128
num_workers = 0

if do_posterior_refinemenent and (not do_laplace or not do_hmc):
    do_laplace, do_hmc = True, True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WideResNet(depth=16, num_classes=10, widen_factor=4, dropRate=0.0)
model = torch.nn.DataParallel(model).to(device)

#checkpoint_bestmodel = torch.load('/home/clem/Documents/Thesis/checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_16_epochs_150_wd_0.0005/model_best.pt')
checkpoint_bestmodel = torch.load(basepath + 'checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_128_epochs_100_wd_0.0005_new_data_prep_5/checkpoint.pt')

model.load_state_dict(checkpoint_bestmodel['state_dict'])
model.eval()

train_loader, val_loader, test_loader, num_classes = load_cifar(dataset_choice, basepath + 'Datasets', batch_nb, num_workers, batch_size_val=batch_nb, val_size=2000)

if do_map:
    
    probs_map, targets_map = predict(test_loader, device, model)
    ece_map = ECE(bins=15).measure(probs_map.numpy(), targets_map.numpy())
    acc_map = sum(torch.argmax(probs_map, 1)==targets_map)/len(targets_map)
    nll_map = -torch.distributions.Categorical(probs_map).log_prob(targets_map).mean()
    print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.4}')


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

    print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.4}')
    #print(f"Mean LA samples {la_samples.mean(0)}")
    
    posterior_params = {'mean': la.mean, 'covariance_m': la.posterior_covariance}
    torch.save(posterior_params, './Run_metrics/la_approx_posterior')


if do_hmc:
    
    act_train, y_train = get_act_Lm1(model, train_loader, device)
    act_val, y_val = get_act_Lm1(model, val_loader, device)
    act_test, y_test = get_act_Lm1(model, test_loader, device)
    
    best_nll_hmc = -torch.log(torch.tensor(0)).item()
    
    for prec in precs_prior_hmc:
        
        nuts_kernel = pyro.infer.NUTS(model_hmc)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=600, warmup_steps=300)
    
        mcmc.run(act_train, num_classes, y_train, prec)

        hmc_samples = mcmc.get_samples(600)
        
        acc_hmc, ece_hmc, nll_hmc = metrics_hmc_samples(hmc_samples['ll_weights'], act_val, y_val)
        
        if nll_hmc < best_nll_hmc:
            
            best_acc_hmc, best_ece_hmc, best_nll_hmc = acc_hmc, ece_hmc, nll_hmc
            best_prec = prec
            torch.save(hmc_samples, './Run_metrics/hmc_samples')
            
    
        print(f'[HMC] validation: Prec: {prec} -> Acc.: {acc_hmc:.1%}; ECE: {ece_hmc:.1%}; NLL: {nll_hmc:.4}')
    
    print(f'[HMC] BEST validation: Opt-Prec {best_prec};  Acc.: {best_acc_hmc:.1%}; ECE: {best_ece_hmc:.1%}; NLL: {best_nll_hmc:.4}')
    
    
    hmc_samples = torch.load('./Run_metrics/hmc_samples')
    acc_hmc, ece_hmc, nll_hmc = metrics_hmc_samples(hmc_samples['ll_weights'], act_test, y_test)
    
    print(f'[HMC] Best on test: Acc.: {acc_hmc:.1%}; ECE: {ece_hmc:.1%}; NLL: {nll_hmc:.4}')
    
    #print(f"Mean HMC samples {hmc_samples['ll_weights'].mean(0)}")
    #plot_prior_vs_posterior_weights_pred(hmc_samples, data, ys, num_classes)
    
if do_posterior_refinemenent and do_laplace and do_hmc:
    
    lr_min = 1e-6
    n_epochs = 20
    
    dim = hmc_samples['ll_weights'][0].shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam({'lr' : 0.001})
    #n_steps = epochs * len(train_loader)
    params_scheduler = {'optimizer': optimizer, 'optim_args': {'T_max': n_epochs, 'eta_min': lr_min}}
    scheduler = optim.CosineAnnealingLR(params_scheduler)
    nf = Normalizing_flow(dim, 'radial', 1, device, posterior_params)

    svi = SVI(nf.model, nf.guide, scheduler, loss=Trace_ELBO())

    losses = []
    for epoch in range(n_epochs):
        
        loss = svi.step(act_train, y_train)
        scheduler.step()
        losses.append(loss)
    
    refined_posterior_samples = nf.sample(600)
    torch.save(hmc_samples, './Run_metrics/hmc_samples')
    
    if make_plots:
        
        plt.figure()
        plt.plot(list(range(n_epochs)), losses, label='loss')
        plt.xlabel('epochs')
        plt.ylabel('losses')
        plt.title('Training loss of normalizing flow per epoch')
        plt.legend()
        plt.savefig(basepath + 'Figures/Training_loss_nfs.pdf', bbox_inches='tight', format='pdf')        
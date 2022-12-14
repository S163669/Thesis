#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:25:13 2022

@author: clem
"""

from laplace import Laplace
import torch
from models import WideResNet, Swag
from dataloaders import load_cifar
from netcal.metrics import ECE
from utils import predict, plot_prior_vs_posterior_weights_pred, model_hmc, metrics_ll_weight_samples, get_act_Lm1
import pyro
from models import Normalizing_flow
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
import os
import pickle

basepath = '/home/clem/Documents/Thesis/'
#basepath = '/zhome/fa/5/117117/Thesis/'
do_map = False
do_laplace = False
do_hmc = False
do_swag = True
K = 100                 #rank for covariance matrix approximation of swag
do_posterior_refinemenent = False
flow_types = ['radial', 'planar', 'spline']
make_plots = False
save_results = True

precs_prior_hmc = [30, 35, 40, 45, 50, 55]
flow_lens = [1, 5, 10, 30]

dataset_choice = 'cifar10'
data_norm = True
model_choice = 'WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_128_epochs_100_wd_0.0005_new_data_prep_5'
torch.manual_seed = 12
batch_nb = 128
num_workers = 0

train_loader, val_loader, test_loader, num_classes = load_cifar(dataset_choice, basepath + 'Datasets', batch_nb, num_workers,
                                                                batch_size_val=batch_nb, val_size=2000, data_augmentation=False, normalize=data_norm)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WideResNet(depth=16, num_classes=num_classes, widen_factor=4, dropRate=0.0)
model = torch.nn.DataParallel(model).to(device)

#checkpoint_bestmodel = torch.load('/home/clem/Documents/Thesis/checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_16_epochs_150_wd_0.0005/model_best.pt')
checkpoint_bestmodel = torch.load(basepath + f'checkpoints/{dataset_choice}/{model_choice}/checkpoint.pt')

model.load_state_dict(checkpoint_bestmodel['state_dict'])
model.eval()

metrics_path = f'./Run_metrics/{dataset_choice}/{model_choice}/'
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)
    results = {}
else:
    try:
        with open(metrics_path + 'results_metrics.pkl', 'rb') as f:
            results = pickle.load(f)
    except:
        results = {}


if do_map:
    
    probs_map, targets_map = predict(test_loader, device, model)
    ece_map = ECE(bins=15).measure(probs_map.numpy(), targets_map.numpy())
    acc_map = (sum(torch.argmax(probs_map, 1)==targets_map)/len(targets_map)).item()
    nll_map = -torch.distributions.Categorical(probs_map).log_prob(targets_map).mean().item()
    map_sample = torch.nn.utils.parameters_to_vector(model.parameters()).detach()[-(256+1)*num_classes:]
    if save_results:
        torch.save(map_sample, metrics_path + 'map_sample')
    
    results['map'] = {'acc': acc_map, 'ece': ece_map, 'nll': nll_map}
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
    nll_laplace = -torch.distributions.Categorical(probs_laplace).log_prob(targets).mean().item()
    
    la_samples = la.sample(600)
    
    results['la'] = {'acc': acc_laplace, 'ece': ece_laplace, 'nll': nll_laplace}
    print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.4}')
    
    posterior_params = {'mean': la.mean, 'covariance_m': la.posterior_covariance}
    
    if save_results:
        torch.save(la_samples, metrics_path + 'la_samples')
        torch.save(posterior_params, metrics_path + 'la_approx_posterior')


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
        
        acc_hmc, ece_hmc, nll_hmc = metrics_ll_weight_samples(hmc_samples['ll_weights'], act_val, y_val, num_classes)
        
        if nll_hmc < best_nll_hmc:
            
            best_acc_hmc, best_ece_hmc, best_nll_hmc = acc_hmc, ece_hmc, nll_hmc
            best_prec = prec
            hmc_samples['precision'] = prec 
            if save_results:
                torch.save(hmc_samples, metrics_path + 'hmc_samples')
            
    
        print(f'[HMC] validation: Prec: {prec} -> Acc.: {acc_hmc:.1%}; ECE: {ece_hmc:.1%}; NLL: {nll_hmc:.4}')
    
    print(f'[HMC] BEST validation: Opt-Prec {best_prec};  Acc.: {best_acc_hmc:.1%}; ECE: {best_ece_hmc:.1%}; NLL: {best_nll_hmc:.4}')
    
    if save_results:
        hmc_samples = torch.load(metrics_path + 'hmc_samples')
    acc_hmc, ece_hmc, nll_hmc = metrics_ll_weight_samples(hmc_samples['ll_weights'], act_test, y_test, num_classes)
    
    results['hmc'] = {'acc': acc_hmc, 'ece': ece_hmc, 'nll': nll_hmc}
    print(f'[HMC] Best on test: Acc.: {acc_hmc:.1%}; ECE: {ece_hmc:.1%}; NLL: {nll_hmc:.4}')
    
    #print(f"Mean HMC samples {hmc_samples['ll_weights'].mean(0)}")
    #plot_prior_vs_posterior_weights_pred(hmc_samples, data, ys, num_classes)
    

if do_swag:
    
    lr_swag = 1e-3
    epochs_swag = 100
    c = 1               # Add weights in running average every c'th iteration
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.module.fc.weight.requires_grad = True
    model.module.fc.bias.requires_grad = True
    
    nb_params = sum(p.numel() for p in filter(lambda l: l.requires_grad, model.parameters()))
    
    swag = Swag(model, nb_params, device, K=K, swag=True)
    
    swag.fit_swag(train_loader, lr_swag, epochs_swag, c)
    
    probs_swag, targets, swag_samples = swag.swag_inference(test_loader, S=600)
    
    acc_swag = (probs_swag.numpy().argmax(-1) == targets.numpy()).astype(int).mean()
    ece_swag = ECE(bins=15).measure(probs_swag.numpy(), targets.numpy())
    nll_swag = -torch.distributions.Categorical(probs_swag).log_prob(targets).mean().item()
    
    results['swag'] = {'acc': acc_swag, 'ece': ece_swag, 'nll': nll_swag}
    print(f'[Swag] Acc.: {acc_swag:.1%}; ECE: {ece_swag:.1%}; NLL: {nll_swag:.4}')
    
    dist_params = swag.get_distribution_parameters()
    
    if save_results:
        torch.save(swag_samples, metrics_path + 'swag_samples')
        torch.save(dist_params, metrics_path + 'swag_approx_posterior')
        
    model.load_state_dict(checkpoint_bestmodel['state_dict'])   # Reloading map weights as part of them have been changed in SGD of SWAG
    
if do_posterior_refinemenent:
    
    n_epochs = 20
    
    if 'act_train' not in locals():
        act_train, y_train = get_act_Lm1(model, train_loader, device)
        act_test, y_test = get_act_Lm1(model, test_loader, device)
        
    if 'posterior_params' not in locals():
        posterior_params = torch.load(metrics_path + 'la_approx_posterior')
    
    if 'best_prec' not in locals():
        hmc_samples = torch.load(metrics_path + 'hmc_samples')
        best_prec = hmc_samples['precision']
    
    N = act_train.shape[0]
    dim = (act_train.shape[1] + 1)*num_classes   # +1 for bias *10 for number of weights per hidden unit    
    
    train_loader_act = data.DataLoader(data.TensorDataset(act_train.cpu(), y_train.cpu()), batch_size=128, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for flow_type in flow_types:
        for flow_len in flow_lens:
        
            #optimizer = optim.Adam({'lr' : 0.001})
            optimizer = torch.optim.Adam
            n_steps = n_epochs * len(train_loader_act)
            params_scheduler = {'optimizer': optimizer, 'optim_args': {'lr': 1e-3, 'weight_decay': 0}, 'T_max': n_steps}
            scheduler = optim.CosineAnnealingLR(params_scheduler)
            nf = Normalizing_flow(dim, flow_type, flow_len, device, posterior_params, num_classes, best_prec, N)
        
            svi = SVI(nf.model, nf.guide, optim=scheduler, loss=Trace_ELBO())
        
            losses = []
            for epoch in range(n_epochs):
                print(f'epoch: {epoch+1}')
                epoch_loss = 0
                
                for x, y in train_loader_act:
                    loss = svi.step(x.to(device), y.to(device))
                    scheduler.step()
                    epoch_loss += loss
                    
                print(f'loss: {epoch_loss}')
                losses.append(epoch_loss)
            
            refined_posterior_samples = nf.sample(600)
            if save_results:
                torch.save(refined_posterior_samples, metrics_path + f'refined_posterior_samples_{flow_type}_{flow_len}')
            
            acc_refp, ece_refp, nll_refp = metrics_ll_weight_samples(refined_posterior_samples.cpu(), act_test, y_test, num_classes)
            
            results[f'ref_nf_{flow_type}_{flow_len}'] = {'acc': acc_refp, 'ece': ece_refp, 'nll': nll_refp}
            print(f'[Refined posterior nf_len: {flow_type}-{flow_len}] Best on test: Acc.: {acc_refp:.1%}; ECE: {ece_refp:.1%}; NLL: {nll_refp:.4}')
        
    if make_plots:
        
        plt.figure()
        plt.plot(list(range(n_epochs)), losses, label='loss')
        plt.xlabel('epochs')
        plt.ylabel('losses')
        plt.title('Training loss of normalizing flow per epoch')
        plt.legend()
        if save_results:
            plt.savefig(basepath + 'Figures/Training_loss_nfs.pdf', bbox_inches='tight', format='pdf')
            

    
if save_results:        
    with open(metrics_path + 'results_metrics.pkl', 'wb') as f:
        pickle.dump(results, f)
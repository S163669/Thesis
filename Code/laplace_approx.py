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


def predict(testloader, device, model, using_laplace=False):

    preds = []    
    targets = []

    for inputs, ys in testloader:
        
        inputs = inputs.to(device)

        if using_laplace:
            preds.append(model(inputs))
        else:
            preds.append(torch.softmax(model(inputs), dim=-1))
        
        targets.append(ys)
    
    targets = torch.cat(targets, dim=0)
    preds = torch.cat(preds).cpu()
    
    return preds, targets

path = '/home/clem/Documents/Thesis/Datasets'

#path = '/zhome/fa/5/117117/Thesis/Datasets'

dataset_choice = 'cifar10'
torch.manual_seed = 12
batch_nb = 16
num_workers = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WideResNet(depth=16, num_classes=10, widen_factor=4, dropRate=0.0)
model = torch.nn.DataParallel(model).to(device)

#checkpoint_bestmodel = torch.load('/home/clem/Documents/Thesis/checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_16_epochs_150_wd_0.0005/model_best.pt')
checkpoint_bestmodel = torch.load('/home/clem/Documents/Thesis/checkpoints/WideResNet-16-4_MAP_Adam_lr_0.0001_btch_256_epochs_400_wd_0.001/model_best.pt')


model.load_state_dict(checkpoint_bestmodel['state_dict'])

la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='full')

train_loader, test_loader, num_classes = load_cifar(dataset_choice, path, batch_nb, num_workers, val_size=0)

la.fit(train_loader)
la.optimize_prior_precision(method='marglik')

probs_laplace, targets = predict(test_loader, device, la, using_laplace=True)
acc_laplace = (probs_laplace.numpy().argmax(-1) == targets.numpy()).astype(int).mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -torch.distributions.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')






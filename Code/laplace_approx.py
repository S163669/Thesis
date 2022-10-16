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
#import pyro
#import pyro.distributions as dist
#import torch.nn.functional as F


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


basepath = '/home/clem/Documents/Thesis/'
#basepath = '/zhome/fa/5/117117/Thesis/'

dataset_choice = 'cifar10'
torch.manual_seed = 12
batch_nb = 16
num_workers = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WideResNet(depth=16, num_classes=10, widen_factor=4, dropRate=0.0)
model = torch.nn.DataParallel(model).to(device)

#checkpoint_bestmodel = torch.load('/home/clem/Documents/Thesis/checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_16_epochs_150_wd_0.0005/model_best.pt')
checkpoint_bestmodel = torch.load(basepath + 'checkpoints/WideResNet-16-4_MAP_SGDNesterov_lr_0.1_lr_min_1e-06_btch_128_epochs_100_wd_0.0005_new_data_prep_5/model_best.pt')


model.load_state_dict(checkpoint_bestmodel['state_dict'])

la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='full')

train_loader, test_loader, num_classes = load_cifar(dataset_choice, basepath + 'Datasets', batch_nb, num_workers, val_size=0)

la.fit(train_loader)
la.optimize_prior_precision(method='marglik')

probs_laplace, targets = predict(test_loader, device, la, using_laplace=True)
acc_laplace = (probs_laplace.numpy().argmax(-1) == targets.numpy()).astype(int).mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -torch.distributions.Categorical(probs_laplace).log_prob(targets).mean()


print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

import torch
import pyro
import pyro.distributions as dist
import torch.nn.functional as F

def model_hmc(data, num_classes, labels):
    
    pad_1 = torch.nn.ConstantPad1d((0,1), 1)
    data = pad_1(data)
    
    dim = data.size()[1]*num_classes
    #coefs_mean = mean*torch.ones(len(var))
    coefs_mean = torch.zeros(dim)
    coefs = pyro.sample('prior_samples', dist.Normal(coefs_mean, ((1/40)**1/2)*torch.ones(dim)))  #Precision of 40 in paper

    #y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
    #y = torch.argmax(torch.softmax(data @ coefs.reshape(-1,10), dim=1), dim=1)
    #y = pyro.sample('y', dist.Categorical(logits=data @ coefs.reshape(-1,10)), obs=labels)
    y = torch.softmax(data @ coefs.reshape(-1,10), dim=1)
    y = y[range(len(y)), labels]
    return y
    

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model.module.bn1.register_forward_hook(get_activation('bn1'))

acts = []
ys = []

with torch.no_grad():

    for x, y in train_loader:
        
        model.eval()
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

la_samples = la.sample(500)
hmc_samples = mcmc.get_samples(500)

print(f"Mean LA samples {la_samples.mean(0)}")
print(f"Mean HMC samples {hmc_samples['prior_samples'].mean(0)}")







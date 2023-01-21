#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:16:57 2022

@author: clem
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
import pyro
from utils import predict_Lm1, predict, get_act_Lm1, multisample_ll_prediction

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def get_lhl_act(self, x):
        
        out = self.conv1(x)
        
        out = self.block1(out)
        
        out = self.block2(out)
        
        out = self.block3(out)
        
        out = self.relu(self.bn1(out))
        
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        
        return out
    
    def forward(self, x):
        
        return self.fc(self.get_lhl_act(x))

def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


class Normalizing_flow(nn.Module):
    
    def __init__(self, input_dim, nf_type, flow_len, device, base_dist_params, num_classes, prior_prec, N, diag=False):
        
        super(Normalizing_flow, self).__init__()
        
        self.input_dim = input_dim
        self.nf_type = nf_type
        self.flow_len = flow_len
        self.device = device
        self.num_classes = num_classes
        self.prior_prec = prior_prec
        self.N = N
        
        if not diag: 
            #self.base_dist = dist.MultivariateNormal(base_dist_params['mean'].to(device), base_dist_params['covariance_m'].to(device))
            # The first line was changed into the second due to the way positive definitess is checked in pyro
            # being more imprecise than in PyTorch and resulting in errors in certain cases.
            self.base_dist = dist.MultivariateNormal(base_dist_params['mean'].to(device), scale_tril=torch.linalg.cholesky_ex(base_dist_params['covariance_m'].to(device)[0]))
        else:
            self.base_dist = dist.Normal(base_dist_params['mean'].to(device), base_dist_params['std'].to(device))
        
        if nf_type == 'planar':
        
            trans = dist.transforms.Planar
            
        elif nf_type == 'spline':
            
            trans = dist.transforms.Spline
        
        elif nf_type == 'radial':
            
            trans = dist.transforms.Radial
            
        else:
            
            print('The inputed flow type is not valid, the type will be set to radial by default.')
            trans = dist.transforms.Radial
            
        self.transforms = [trans(input_dim) for _ in range(flow_len)]
        
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.transforms)
        
    
    def guide(self, x=None, p=None):
        """
        p isn't used but must be provided due to the way the pyro framework is built.
        This part represents the variational distribution called guide in pyro vocabulary
        """
        
        #N = len(x) if x is not None else None
        pyro.module("nf", nn.ModuleList(self.transforms).to(self.device))
        pyro.sample("weights", self.flow_dist)
                
    def model(self, x, y):
        """
        This part represents the target distribution.
        p(x,z), but x is not required if there is a true density function (p_z in this case)
        
        1. Sample Z ~ p_z
        2. Score it's likelihood against p_z
        """
        
        #ws = pyro.sample("weights", self.flow_dist)
        ws = pyro.sample("weights", dist.Normal(torch.zeros(self.input_dim).to(self.device), math.sqrt(1/(self.prior_prec))).to_event(1))
        _, class_probs, _  = predict_Lm1(ws, x, y, self.num_classes)
        with pyro.plate("data", size=self.N, subsample_size=len(y.squeeze())):
            pyro.sample("obs", dist.Categorical(probs=class_probs), obs=y)
    
    
    def sample(self, num_samples):
        
        return self.flow_dist.sample(torch.Size([num_samples]))
    
    def log_prob(self, z):
        """
        Returns log q(z|x) for z (assuming no x is required)
        """
        return self.flow_dist.log_prob(z)
    
    
class Swag():
    
    def __init__(self, model, no_params, device, K, swag=True):
        
        self.model = model
        self.no_params = no_params
        self.device = device
        self.swa_avg_m1 = torch.nn.utils.parameters_to_vector(filter(lambda l: l.requires_grad, self.model.parameters())).detach().to(self.device)
        self.swag = swag
        
        if swag:
            self.swa_avg_m2 = torch.square(self.swa_avg_m1)
            self.Ds = torch.zeros((no_params, K)).to(self.device)
            self.K = K
            self.D_it = 0
        else:
            self.swa_avg_m2 = None
            self.Ds = None
            self.swa_diag = None

    def fit_swag(self, train_loader, lr, epochs, c, diag=False):
        
        self.diag = diag
        optimizer = torch.optim.SGD(filter(lambda l: l.requires_grad, self.model.parameters()), lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        
        for epoch in range(1, epochs+1):
            loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
    
                # compute output
                outputs = self.model(inputs)
                batch_loss = criterion(outputs, targets)
                
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss.cpu()
                
            if not epoch%10 or epoch == 1:
                print(f'[SWAG SGD] epoch {epoch} loss {loss}')
            
            # Add a point every c'th iteration
            if epoch % c == 0:
                with torch.no_grad():
                    n = epoch // c

                    layer = torch.nn.utils.parameters_to_vector(filter(lambda l: l.requires_grad, self.model.parameters()))
                    self.swa_avg_m1 = (n * self.swa_avg_m1 + layer) / (n + 1)

                    if self.swag:
                        self.swa_avg_m2 = (n * self.swa_avg_m2 + torch.square(layer)) / (n + 1)

                        if epoch > epochs - self.K*c and not diag:
                            self.Ds[:, self.D_it] = layer - self.swa_avg_m1
                            self.D_it += 1
     
    
        if self.swag:
            self.swa_diag = self.swa_avg_m2 - torch.square(self.swa_avg_m1)
            
    def sample(self, S=1):
        
        samples = []
        
        z1 = torch.normal(0, 1, size=(S, self.no_params)).to(self.device)
    
        if self.diag:
            samples = self.swa_avg_m1 + torch.sqrt(self.swa_diag) * z1
        
        else:
            z2 = torch.normal(0, 1, size=(S, self.K)).to(self.device)
            # Here self.Ds @ z2 is changed into z2 @ self.Ds.T to do computations for more than one sample at the time.
            samples = self.swa_avg_m1 + torch.sqrt(self.swa_diag) * z1/math.sqrt(2) + (z2@self.Ds.T)/math.sqrt(2*(self.K-1))
            
        return samples
            
            
    def swag_inference(self, data_loader, S=600, last_layer=False):
    
        with torch.no_grad():

            if torch.sum(self.swa_diag <= 0):
                print(f"{torch.sum(self.swa_diag <= 0)} ({torch.sum(self.swa_diag <= 0)/self.no_params*100:.4f}%) non-positive values encountered in the diagonal matrix!")
                self.swa_diag[self.swa_diag < 0] = 1e-10

            self.model.eval()
            samples = self.sample(S)
            
            # The distinction is made here as precomputing the activation makes the computation ~30 times faster
            if last_layer:
                
                act, y = get_act_Lm1(self.model, data_loader, self.device)

                final_probs, acc = multisample_ll_prediction(samples.cpu(), act, y, 10)
                
            else:
                sum_probs = 0
                for sample in samples:
    
                    torch.nn.utils.vector_to_parameters(sample, filter(lambda l: l.requires_grad, self.model.parameters()))
                    
                    probs, y = predict(data_loader, self.device, self.model)
                    sum_probs += probs
    
                final_probs = sum_probs/S
                
                y_pred = torch.argmax(final_probs, 1)
                acc = sum(y_pred == y).item()/len(y)/math.sqrt(2)
                
        return final_probs, y, acc, samples
    
    
    def get_distribution_parameters(self):
        
        if self.diag:
            if torch.sum(self.swa_diag <= 0):
                print(f"{torch.sum(self.swa_diag <= 0)} ({torch.sum(self.swa_diag <= 0)/self.no_params*100:.4f}%) non-positive values encountered in the diagonal matrix!")
                self.swa_diag[self.swa_diag < 0] = 1e-10
            swag_params = {'mean': self.swa_avg_m1, 'std': torch.sqrt(self.swa_diag)}
        
        else:    
            cov_mat = 1/2*torch.diag(self.swa_diag) + (self.Ds @ self.Ds.T)/(2*(self.K-1))
            mask = cov_mat[range(self.no_params),range(self.no_params)] <= 0
        
            # Enforcing positive definitness of matrix
            if torch.sum(mask):
                print(f"{torch.sum(mask)} ({torch.sum(mask)/self.no_params*100:.4f}%) non-positive values encountered in the matrix diagonal!")
                cov_mat[range(self.no_params),range(self.no_params)][mask] = 1e-10
                print("the non-positive elements were set to 1e-16")
            
            # Including parameters for diagonal SWAG, for them to come from the same run
            if torch.sum(self.swa_diag <= 0):
                self.swa_diag[self.swa_diag < 0] = 1e-10
        
            swag_params = {'mean': self.swa_avg_m1, 'covariance_m': cov_mat, 'std': torch.sqrt(self.swa_diag)}
        
        return swag_params
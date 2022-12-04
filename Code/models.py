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
from utils import predict_Lm1

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
    
    def __init__(self, input_dim, nf_type, flow_len, device, base_dist_params, num_classes, prior_prec, N):
        
        super(Normalizing_flow, self).__init__()
        
        self.input_dim = input_dim
        self.nf_type = nf_type
        self.flow_len = flow_len
        self.device = device
        self.num_classes = num_classes
        self.prior_prec = prior_prec
        self.N = N
        
        self.base_dist = dist.MultivariateNormal(base_dist_params['mean'].to(device), base_dist_params['covariance_m'].to(device))
        
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
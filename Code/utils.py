#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:13:12 2022

@author: clem
"""

import os
import torch
import shutil

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pt'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pt'))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:09:39 2024

@author: kanchan
"""

import torch
import math
from torch import nn
import torch.nn.functional as F
from operator import mul
from functools import reduce

from local_attention.local_attention import LocalAttention
from transformer import MultiHead

def spatial_attention(self,x):
    b,h,w,t,d = x.shape
    spatial_feat = x.premute(1,2,3,0,4).reshape(h,w,t,b*d)
    spatial_model=LocalAttention(dim = 100,                # dimension of each head (you need to pass this in for relative positional encoding)
              window_size = 8,       
              causal = True,          
              look_backward = 1,       
              look_forward = 0,       
              dropout = 0.1,           
              exact_windowsize = False #  
              ).cuda()
    temporal_model = MultiHead(d_key=100, d_value=100, n_heads=4, drop_ratio=0.1,causal=False).cuda()
    spatial_attention = spatial_model(spatial_feat,spatial_feat,spatial_feat)
    spatial_output = spatial_attention.rehape(h,w,t,b,d).permute(3,0,1,2,4)
    spatial_output = spatial_output.reshape(0,3,2,1,4).reshape(b,t,h*w*d)
    temporal_output = temporal_Model(spatial_output,spatial_output,spatial_output)
    return temporal_output
    
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 01:39:03 2022

@author: Tinky
"""
import torch, numpy as np
import torch.nn as nn

class MLPCustom(nn.Module):
    def __init__(self, 
                 input_dim, 
                 h_sizes_list,
                 d_prob_list, 
                 output_dim, 
                 non_linearity=None, 
                 dropout=True,
                 batch_norm=False, 
                 momentum_batch=None,  
                 vocab_size=None, 
                 embed_dim=None, 
                 mode='normal'):

        super().__init__()
        
        self.mode = mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_sizes_list = h_sizes_list
        self.d_prob_list = d_prob_list
        self.batch_norm = batch_norm
        self.momentum_batch = momentum_batch
        self.embedding = None
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.non_linearity = non_linearity
    
        model_layers = []
        if self.mode == 'image':
            model_layers.append(nn.Flatten())
            in_dim = self.input_dim
        elif self.mode =='language':
            self.embedding = nn.EmbeddingBag(self.vocab_size, self.embed_dim)
            in_dim = self.embed_dim
        else:
            in_dim = self.input_dim
            
        # hidden layers, droput, non_linearity, batchnorm layers
        for k, hidden_size in enumerate(self.h_sizes_list):
            # hidden_layer
            model_layers.append(nn.Linear(in_dim, hidden_size))
            # Activation function
            model_layers.append(self.non_linearity)
            # Dropout Layer
            if dropout:
                model_layers.append(nn.Dropout(p=self.d_prob_list[k]))
            # Batch_Norm Layer
            if self.batch_norm:
              model_layers.append(nn.BatchNorm1d(hidden_size, momentum = self.momentum_batch))
            in_dim = hidden_size
        
        # output layer  
        if len(self.h_sizes_list)>0:
            model_layers.append(nn.Linear(self.h_sizes_list[-1], self.output_dim))
        else:
            model_layers.append(nn.Linear(in_dim, self.output_dim))
        
        self.module_list = nn.ModuleList(model_layers)
    
    def forward(self, x, offsets=None):
        if self.mode == 'language':
            x = self.embedding(x, offsets)
        else:
            pass
        
        for layer in self.module_list:
            x = layer(x)
        return x

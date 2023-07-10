# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:30:49 2023

@author: lingg
"""
import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from time import time
# from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import torch
import os
import nn_training_classification as nnfunc
import nn_models as nnm
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import WeightedRandomSampler




hyperparameter = SimpleNamespace(seed = 42, 
                                 train_frac = 0.75,
                                 epochs = 40,
                                 input_dim = 32*32*3,
                                 output_dim = 2,
                                 h_sizes = [100]*5, 
                                 dprob = [0.3]*5,
                                 non_linearity = nn.ReLU(),
                                 batch_norm = True,
                                 batch_size= 62,
                                 learning_rate=0.005,
                                 dataset="CIFAR10",
                                 architecture="MLP",
                                 log_interval = 1,
                                 log_batch = True,
                                 file_model = Path(os.getcwd())/r'models/NN_best.pt',
                                 grad_clipping = False, 
                                 early_stopping = True,
                                 max_norm = 1,
                                 momentum = 0,
                                 patience = 5,
                                 # scheduler_factor = 0.5,
                                 # scheduler_patience = 0,
                                 weight_decay = 0.00, 
                                 device = torch.device('cpu'),
                                 max_lr = 0.15,
                                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))



x = []
y = []
x = torch.tensor(x)
y = torch.tensor(y)
sample_weights = [0.1, 0.9]
 
train_val_set = nnfunc.CustomDataset(x, y)
trainset, valset = nnfunc.split_dataset(train_val_set, hyperparameter.train_frac, seed=hyperparameter.seed)

sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples = len(trainset), replacement=True)  


train_loader = torch.utils.data.DataLoader(trainset, batch_size=hyperparameter.batch_size, shuffle = True, sampler=sampler)
val_loader = torch.utils.data.DataLoader(valset, batch_size=hyperparameter.batch_size, shuffle = False)

model = nnm.MLPCustom(input_dim=x[1], 
                      h_sizes_list=hyperparameter.h_sizes,
                      d_prob_list=hyperparameter.dprob, 
                      output_dim=hyperparameter.output_dim, 
                      non_linearity=hyperparameter.non_linearity, 
                      dropout=True,
                      batch_norm=False, 
                      momentum_batch=hyperparameter.momentum,  
                      vocab_size=None, 
                      embed_dim=None, 
                      mode='normal')
model.to(hyperparameter.device)
model.apply(nnfunc.init_weights)
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), 
                            lr = hyperparameter.learning_rate, 
                            momentum = hyperparameter.momentum,
                            weight_decay = hyperparameter.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=hyperparameter.max_lr, 
                                                total_steps=(len(train_loader)*hyperparameter.epochs),  
                                                three_phase=True)
batch_ct_train, batch_ct_valid = 0, 0
train_loss_history, train_acc_history, valid_loss_history, valid_acc_history = nnfunc.train_loop(train_loader, 
                                                                                                  val_loader, 
                                                                                                  model, 
                                                                                                  optimizer,
                                                                                                  loss_function,
                                                                                                  hyperparameter.epochs,
                                                                                                  hyperparameter.device,
                                                                                                  hyperparameter.patience,
                                                                                                  hyperparameter.early_stopping,
                                                                                                  hyperparameter.file_model,
                                                                                                  hyperparameter.grad_clipping,
                                                                                                  hyperparameter.max_norm,
                                                                                                  scheduler)
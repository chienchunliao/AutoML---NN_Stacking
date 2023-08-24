# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 01:39:03 2022

@author: Tinky
"""
import torch, numpy as np
import torch.nn as nn
from time import time
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import copy
import nn_training_classification as nnfunc
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, r2_score

#%%
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

#%%
class autoML():
    def __init__(self, classification=True):
        self.ml_models_best = None
        self.ml_cv_score = None
        self.classification = classification
        
    def _get_multi_pred(self, x, model_list, proba=True):
        preds_init = []
        for model in model_list:
            if proba:
                preds_init.append(torch.tensor(model.predict_proba(x)[:,1]))
            else:
                preds_init.append(torch.tensor(model.predict(x)[:,1]))
        x_stack = torch.stack(preds_init, dim=1)
        x_stack = x_stack.to(torch.float32)
        return x_stack

    def fit(self, 
            models, 
            x_train, 
            y_train, 
            scoring='r2', 
            n_jobs=-1, 
            n_best=1, 
            train_frac=3/4, 
            seed=42, 
            batch_size=32,
            optimizer=None,
            scheduler=None,
            classification=False):
        
        global hyperparameter
        
        self.x_train = x_train
        self.y_train = y_train
        self.ml_models = models
        
        ml_result_list = []
        columns = ['model name', 
                   'hyperparameter', 
                   'cv score', 
                   'time used', 
                   'model']
        
        ## Train 1st layer ML models
        for model in self.ml_models:
            t_0 = time()
            model_name = model.name
            parameters = model.parameters
            model_cv = GridSearchCV(model.model, 
                                    parameters,
                                    n_jobs=n_jobs, 
                                    scoring=scoring)
            model_cv.fit(x_train, y_train)
            best_para = model_cv.best_params_
            cv_score = model_cv.best_score_
            best_model = model_cv.best_estimator_
            time_used = time()-t_0
            ml_result_list.append(pd.Series([model_name, best_para, cv_score, time_used, best_model],
                                  index=columns))
        df_result = pd.DataFrame(ml_result_list, 
                                 columns=columns)
        self.ml_result_df = df_result.sort_values('cv score', ascending=False)
        
        self.ml_result_best = df_result.iloc[:n_best,:]
        self.ml_models_best = df_result['model'][:n_best]
        self.ml_cv_score = df_result['cv score'][:n_best]
        with open(hyperparameter.file_model_ml,"wb") as f:
            pickle.dump(self.ml_models_best, f)
        if n_best == 1:
            # self.ml_result_best = df_result.iloc[0,:]
            # self.ml_models_best = [self.ml_result_best['model']]
            # self.ml_cv_score = self.ml_result_best['cv score']
            # with open(hyperparameter.file_model_ml,"wb") as f:
            #     pickle.dump(self.ml_models_best, f)
            pass
        else:
            ## Train 2nd layer NN model
            x_stack = self._get_multi_pred(x_train, self.ml_models_best)
            y_stack = torch.tensor(y_train.array)
            
            train_val_set = nnfunc.CustomDataset(x_stack, y_stack)
            trainset, valset = nnfunc.split_dataset(train_val_set, train_frac, seed=seed)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle = False)

            model = MLPCustom(input_dim=x_stack.shape[1], 
                              h_sizes_list=hyperparameter.h_sizes,
                              d_prob_list=hyperparameter.dprob, 
                              output_dim=hyperparameter.output_dim, 
                              non_linearity=hyperparameter.non_linearity, 
                              dropout=hyperparameter.dropout,
                              batch_norm=hyperparameter.batch_norm, 
                              momentum_batch=hyperparameter.momentum,  
                              vocab_size=None, 
                              embed_dim=None, 
                              mode='normal')
            model_load = copy.copy(model)
            model.to(hyperparameter.device)
            model.apply(nnfunc.init_weights)
            if classification:
                loss_function = nn.CrossEntropyLoss()
            else:
                loss_function = nn.MSELoss(reduction='mean')
            if optimizer is None:
                optimizer = torch.optim.SGD(model.parameters(), 
                                            lr = hyperparameter.learning_rate, 
                                            momentum = hyperparameter.momentum,
                                            weight_decay = hyperparameter.weight_decay)
            if scheduler is None:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                                max_lr=hyperparameter.max_lr, 
                                                                total_steps=(len(train_loader)*hyperparameter.epochs),  
                                                                three_phase=True)
            #batch_ct_train, batch_ct_valid = 0, 0
            train_loss_history, train_score_history, valid_loss_history, valid_score_history = nnfunc.train_loop(train_loader, 
                                                                                                             val_loader, 
                                                                                                             model, 
                                                                                                             optimizer,
                                                                                                             loss_function,
                                                                                                             hyperparameter.epochs,
                                                                                                             hyperparameter.device,
                                                                                                             hyperparameter.patience,
                                                                                                             hyperparameter.early_stopping,
                                                                                                             hyperparameter.file_model_nn,
                                                                                                             hyperparameter.grad_clipping,
                                                                                                             hyperparameter.max_norm,
                                                                                                             scheduler,
                                                                                                             scoring)
                                                                                                 
            model_load.load_state_dict(torch.load(hyperparameter.file_model_nn))
            self.model_nn = model_load                                                                                                
                                                                                                             

    def predict(self, x, proba=False):
        
        if self.ml_models_best is None:
            print('Fit the model first!')
        else:
            with open(hyperparameter.file_model_ml, 'rb') as f:
                ml_model_list_trained = pickle.load(f)
            if len(self.ml_models_best)==1:
                x_nn= self.ml_models_best[0].predict(x)
                return x_nn.cpu().numpy()
            else:
                x_nn= self._get_multi_pred(x, ml_model_list_trained, proba=True)
            y_nn = torch.empty(len(x_nn))
            ds = nnfunc.CustomDataset(x_nn, y_nn)
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle = False)
            model_nn = self.model_nn
            model_nn.to(hyperparameter.device)
            preds, _ = nnfunc.get_pred(data_loader = dl, 
                                       model = model_nn, 
                                       device = hyperparameter.device, 
                                       return_ytrue=False,
                                       return_proba=proba)
                         
            return preds.cpu().numpy()
    
    
    def score(self, x, y, scoring='r2'):
        if len(self.ml_models_best) == 0:
            print('Fit the model first!')
        else:
            y_pred = self.predict(x)
            if scoring == 'r2':
                return r2_score(y, y_pred)
            elif scoring == 'f1':
                return f1_score(y, y_pred)
            elif scoring == 'accuracy':
                return accuracy_score(y, y_pred)
            elif scoring == 'precision':
                return precision_score(y, y_pred)
            elif scoring == 'recall':
                return recall_score(y, y_pred)
            else:
                print('Only accept: r2, f1, accuracy, precision, or recall')
                return None


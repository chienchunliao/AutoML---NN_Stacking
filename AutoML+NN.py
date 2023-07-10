# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import torch
import nn_training_classification as nnfunc
import nn_models as nnm
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
import random

#%%

class autoML():
    def __init__(self):
        self.best_model = None
        self.cv_score = None
    def fit(self, models, x_train, y_train, scoring='f1', n_jobs=-1, n_best=1, train_frac=3/4, seed=42, batch_size=32):
        self.x_train = x_train
        self.y_train = y_train
        self.models = models
        self.result_list = []
        self.columns = ['model name', 'hyperparameter', 'cv score', 'time used', 'model']
        for model in self.models:
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
            self.result_list.append(pd.Series([model_name, best_para, cv_score, time_used, best_model],
                                          index=self.columns))
        df_result = pd.DataFrame(self.result_list, 
                                 columns=self.columns)
        self.result_df = df_result.sort_values('cv score', ascending=False)
        if n_best == 1:
            self.best_result = self.result_df.iloc[0,:]
            self.best_model = self.best_result['model']
            self.cv_score = self.best_result['cv score']
        else:
            self.best_result = self.result_df.iloc[:n_best,:]
            self.best_model = self.result_df['model'][:n_best]
            self.cv_socre = self.result_df['cv score'][:n_best]
            
            preds_init = []
            for model in self.best_model:
                preds_init.append(torch.tensor(model.predict_proba(x_train)[:,1]))
            x_stack = torch.stack(preds_init, dim=1)
            x_stack = x_stack.to(torch.float32)
            y_stack = torch.tensor(y_train.array)
            train_val_set = nnfunc.CustomDataset(x_stack, y_stack)
            trainset, valset = nnfunc.split_dataset(train_val_set, train_frac, seed=seed)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle = False)

            global hyperparameter
            model = nnm.MLPCustom(input_dim=x_stack.shape[1], 
                                  h_sizes_list=hyperparameter.h_sizes,
                                  d_prob_list=hyperparameter.dprob, 
                                  output_dim=2, 
                                  non_linearity=hyperparameter.non_linearity, 
                                  dropout=True,
                                  batch_norm=False, 
                                  momentum_batch=hyperparameter.momentum,  
                                  vocab_size=None, 
                                  embed_dim=None, 
                                  mode='normal')
            print(model.module_list)
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
                                                                                                             

    def predict(self, x):
        if len(self.best_model) == 0:
            print('Fit the model first!')
        else:
            y_pred = self.best_model.predict(x)
            return y_pred
    
    
    def score(self, x, y, scoring='f1'):
        if len(self.best_model) == 0:
            print('Fit the model first!')
        else:
            y_pred = self.predict(x)
            if scoring == 'f1':
                return f1_score(y, y_pred)
            elif scoring == 'accuracy':
                return accuracy_score(y, y_pred)
            elif scoring == 'precision':
                return precision_score(y, y_pred)
            elif scoring == 'recall':
                return recall_score(y, y_pred)
            else:
                print('Only accept: f1, accuracy, precision, recall')
                return None
    
class model():
    def __init__(self, model_name, model, parameters):
        self.name = model_name
        self.model = model
        self.parameters = parameters


base_folder = Path(r'D:\Python\autoML_test')
#data_folder = base_folder/'data'
model_folder = base_folder/'models'
#custom_functions = base_folder/'custom-functions'

df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].astype(int)
df = df.sample(n=10000)
y = df['HeartDiseaseorAttack']
x = df.drop(['HeartDiseaseorAttack'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model_knn = model('Knn', 
                  KNeighborsClassifier(n_jobs=-1), 
                  {'n_neighbors':[3,5,10,15]}
                  )
model_logi = model('Logistic Regression',
                   LogisticRegression(n_jobs=-1),
                   {'penalty':['l1', 'l2'],
                    'C': [0.1,1,10]}
                   )
model_svm = model('SVM',
                  SVC(),
                  {'C': [1, 10], 
                   'kernel': ['linear', 'rbf']}
                  )
model_rf = model('Random Froest', 
                 RandomForestClassifier(n_jobs=-1), 
                 {'n_estimators': [10,50,100,150], 
                  'max_depth': [10,30,50,100,150, None]}
                 )

hyperparameter = SimpleNamespace(
    epochs = 40,
    input_dim = 32*32*3,
    output_dim = 10,
    h_sizes = [100]*5, 
    dprob = [0]*5,
    non_linearity = nn.ReLU(),
    batch_norm = True,
    batch_size= 64,
    learning_rate=0.005,
    dataset="CIFAR10",
    architecture="MLP",
    log_interval = 1,
    log_batch = True,
    file_model = model_folder/'autoML_test.pt',
    grad_clipping = False, 
    early_stopping = True,
    max_norm = 1,
    momentum = 0,
    patience = 5,
    # scheduler_factor = 0.5,
    # scheduler_patience = 0,
    weight_decay = 0.00, 
    device = torch.device('cpu'),
    max_lr = 0.15)
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

models = [model_knn, model_logi, model_rf, model_svm]

model = autoML()

model.fit(models, x_train, y_train, n_best=3)
#y_pred = model.predict(x_test)
test_score = model.score(x_test, y_test)
best_model = model.best_model
cv_result = model.result_df
restul_best = model.best_result


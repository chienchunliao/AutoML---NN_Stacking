# -*- coding: utf-8 -*-
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, r2_score, ConfusionMatrixDisplay
import torch
import nn_training_classification as nnfunc
import nn_models as nnm
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
import pickle
import copy
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
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

            model = nnm.MLPCustom(input_dim=x_stack.shape[1], 
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
                         
            return preds
    
    
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
    
class model():
    def __init__(self, model_name, model, parameters):
        self.name = model_name
        self.model = model
        self.parameters = parameters


#%%
base_folder = Path(r'D:\Python\autoML_test')
#data_folder = base_folder/'data'
model_folder = base_folder/'models'
#custom_functions = base_folder/'custom-functions'

hyperparameter = SimpleNamespace(
    epochs = 50,
    output_dim = 2,
    h_sizes = [16,8,4], 
    dprob = [0.2]*3,
    non_linearity = nn.ELU(),
    dropout = True, 
    batch_norm = True,
    batch_size= 32,
    learning_rate=0.005,
    dataset="CIFAR10",
    architecture="MLP",
    log_interval = 1,
    log_batch = True,
    file_model_nn = model_folder/'model_pt_nn.md',
    file_model_ml = model_folder/'model_pt_multi.md',
    grad_clipping = False, 
    early_stopping = True,
    max_norm = 1,
    momentum = 0,
    patience = 5,
    # scheduler_factor = 0.5,
    # scheduler_patience = 0,
    weight_decay = 0.0000, 
    max_lr = 0.001,
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

#%%
df_train = pd.read_pickle('train_clean.bt')
df_test = pd.read_pickle('test_clean.bt')
df_train = df_train.sample(n=100000, random_state=42)
df_test = df_test.sample(n=60000, random_state=42)
y_train = df_train['is_fraud']
x_train = df_train.drop(['is_fraud'],axis=1)
y_test = df_test['is_fraud']
x_test = df_test.drop(['is_fraud'],axis=1)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

sm = RandomOverSampler(random_state=42)
x_res, y_res = sm.fit_resample(x_train, y_train)

model_knn = model('Knn', 
                  KNeighborsClassifier(n_jobs=-1), 
                  {'n_neighbors':[3,5,10,15]}
                  )
model_logi = model('Logistic Regression',
                   LogisticRegression(n_jobs=-1, max_iter=500),
                   {'penalty':['l1', 'l2'],
                    'C': [25,50,100]}
                   )
model_svm = model('SVM',
                  SVC(random_state=42),
                  {'C': [1, 10, 100, 1000], 
                   'kernel': ['linear', 'rbf']}
                  )
model_rf = model('Random Froest', 
                 RandomForestClassifier(n_jobs=-1, random_state=42), 
                 {'n_estimators': [25,30,35], 
                  'max_depth': [75,100,125]}
                 )
model_xb = model('XGboost',
                 XGBClassifier(n_jobs=-1, random_state=42, eval_metric=recall_score),
                 {'n_estimators': [10,15,20],
                  'max_depth':[25,50,75],
                  'max_leaves':[0,200,400]})



models = [model_knn, model_logi, model_rf, model_xb]#, model_svm]

model = nnm.autoML()

model.fit(models, x_res, y_res, n_best=2, scoring='recall')
best_model = model.ml_models_best
cv_result = model.ml_result_df
y_pred = model.predict(x_test)

train_score = recall_score(y_train, model.predict(x_train))
test_score = recall_score(y_test, y_pred)
test_score_single_list = []
for idx_row, row in cv_result.iterrows():
    model_name = row['model name']
    model_ml = row['model']
    y_pred_single = model_ml.predict(x_test)
    test_score_single_list.append(recall_score(y_test, y_pred_single))
cv_result['test_score'] = test_score_single_list

#restul_best = model.best_result
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


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

class autoML():
    def __init__(self):
        self.best_model = None
        self.cv_score = None
    def fit(self, models, x_train, y_train, scoring='f1', n_jobs=-1):
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
        self.result_df = df_result
        self.best_result = df_result.sort_values('cv score', ascending=False).iloc[0,:]
        self.best_model = self.best_result['model']
        self.cv_score = self.best_result['cv score']
    
    def predict(self, x):
        if self.best_model == None:
            print('Fit the model first!')
        else:
            y_pred = self.best_model.predict(x)
            return y_pred
    
    def score(self, x, y, scoring='f1'):
        if self.best_model == None:
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


df = pd.read_csv('winequality-red.csv')
df = df.iloc[:,1:]
y = df['quality']
x = df.drop(['quality'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model_knn = model('Knn', 
                  KNeighborsClassifier(), 
                  {'n_neighbors':[3,5,10,15]}
                  )
model_logi = model('Logistic Regression',
                   LogisticRegression(),
                   {'penalty':['l1', 'l2'],
                    'C': [0.1,1,10]}
                   )
model_svm = model('SVM',
                  SVC(),
                  {'C': [1, 10], 
                   'kernel': ['linear', 'rbf']}
                  )
model_rf = model('Random Froest', 
                 RandomForestClassifier(), 
                 {'n_estimators': [10,50,100,150], 
                  'max_depth': [10,30,50,100,150, None]}
                 )

models = [model_knn, model_logi, model_rf, model_svm]

model = autoML()

model.fit(models, x_train, y_train)
y_pred = model.predict(x_test)
test_score = model.score(x_test, y_test)
best_model = model.best_model
cv_result = model.result_df
restul_best = model.best_result


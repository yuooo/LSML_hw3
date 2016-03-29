# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:24:40 2016

"""

import numpy as np
import sklearn as sk
from sklearn.grid_search import GridSearchCV
import time
import pandas as pd
import pickle
import os
from xgboost import XGBClassifier

#os.chdir('/Users/jessicahoffmann/Desktop/Cours/UTAustin/S2/LargeScaleML/PS3')


#%% Load/prepare data
with open('train.csv') as data_train:
    dataset = np.loadtxt(data_train, delimiter=",", skiprows=1)
    
#print "Shape of the dataset:", dataset.shape

x = dataset[:,1:]
target = dataset[:,0]

with open('test.csv') as data_test:
    dataset_test = np.loadtxt(data_test, delimiter=",", skiprows=1)

print "Shape of the test dataset:", dataset_test.shape

x_test = dataset_test[:,1:]
ID = (dataset_test[:,0]).astype(int)
head = ['Id', 'Action']

#%% helper functions
def write_submission(filename, predicted_proba):
    submission = np.column_stack((ID, predicted_proba))
    df = pd.DataFrame(data=submission, columns = head)
    df['Id'] = df['Id'].astype('int')
    df.to_csv(filename, index=False , header=head)
    
def get_proba_one(model, X):
    predicted = model.predict_proba(X)
    return predicted[:, 1]

def print_time(model_name, start_time):
    t = time.time() - start_time
    print("--- %s %s hours %s minutes %s seconds ---" % (model_name, t//3600, (t%3600)//60, t%60))

def runModel(model, model_name, X, target, X_test):
    start_time = time.time()
    model.fit(X, target)
    print_time(model_name, start_time)
    
    predicted = get_proba_one(model, X_test)
    write_submission('predictions/%s.csv' % model_name, predicted)
    return model

#%% vanilla xgb
xgb_vanilla = runModel(XGBClassifier(), \
'xgb_vanilla', x, target, x_test)  

print "Test results for vanilla xgb:", 0.74789

##%% grid search xgb
#param = {}
#param['max_depth'] = [200,250,300,150]
#param['n_estimators'] = [100, 150, 50]
#param['colsample_bytree'] = [0.4]
##param['booster'] = [gbtree, gblinear]
##param['eval_metric'] = ['auc'] 
#
#print "Start grid search"
#start_time = time.time()
#gs_xgb = GridSearchCV(estimator=XGBClassifier(), \
#param_grid=param, scoring='roc_auc')
#gs_xgb.fit(x, target)
#print_time('GS XGB', start_time)
#
#predicted_gs_xgb = get_proba_one(gs_xgb, x_test)
#write_submission('predictions/gs_xgb.csv', predicted_gs_xgb)

#%% Best tuned XGB
xgb_tuned_q2 = runModel(XGBClassifier(max_depth=200, n_estimators=100, colsample_bytree = 0.4), \
'xgb_tuned_q2', x, target, x_test) 

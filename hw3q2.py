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
from hw3q3 import write_submission, print_time, get_proba_one

os.chdir('/Users/jessicahoffmann/Desktop/Cours/UTAustin/S2/LargeScaleML/PS3')


#%% Load/prepare data
with open('train.csv') as data_train:
    dataset = np.loadtxt(data_train, delimiter=",", skiprows=1)
    
#print "Shape of the dataset:", dataset.shape

X = dataset[:,1:]
target = dataset[:,0]

with open('test.csv') as data_test:
    dataset_test = np.loadtxt(data_test, delimiter=",", skiprows=1)

print "Shape of the test dataset:", dataset_test.shape

X_test = dataset_test[:,1:]
ID = (dataset_test[:,0]).astype(int)
head = ['Id', 'Action']

#%% vanilla xgb
start_time = time.time()
xgbm = XGBClassifier()
xgbm.fit(X, target)
t = time.time() - start_time
print("--- XGBM  %s hours %s minutes %s seconds ---" % (t//3600, (t%3600)//60, t%60))

predicted_xgbm = xgbm.predict(X_test)
write_submission('predictions/xgbm.csv', predicted_xgbm)

print "Test results for vanilla xgb:", 0.74789

#%% grid search xgb
param = {}
param['max_depth'] = [200]
param['learning_rate'] = [1]
param['n_estimators'] = [100]
param['objective'] = ['binary:logistic']
param['gamma'] = [0.5]
param['min_child_weight']=[1]
param['subsample']= [1]
param['scale_pos_weight']=[1]
param['reg_alpha']=[1]
param['reg_lambda']=[0.1]
#param['booster'] = [gbtree, gblinear]
#param['eval_metric'] = ['auc'] 

print "Start grid search"
start_time = time.time()
gs_xgb = GridSearchCV(estimator=XGBClassifier(), \
param_grid=param, scoring='roc_auc')
gs_xgb.fit(X, target)
print_time('GS XGB', start_time)

predicted_gs_xgb = get_proba_one(gs_xgb, X_test)
write_submission('predictions/gs_xgb.csv', predicted_gs_xgb)

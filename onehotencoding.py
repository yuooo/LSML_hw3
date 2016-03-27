# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 23:15:15 2016


"""

import numpy as np
import operator
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import time
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier

# os.chdir('/Users/jessicahoffmann/Desktop/Cours/UTAustin/S2/LargeScaleML/PS3')

#%% Load/prepare data
with open('train.csv') as data_train:
    dataset = np.loadtxt(data_train, delimiter=",", skiprows=1)
    
#print "Shape of the dataset:", dataset.shape

X = dataset[:,1:]
target = dataset[:,0]

# print X
# sorted_mappings = []
# (num_rows, num_cols) = X.shape

enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(X)

# print X
with open('test.csv') as data_test:
    dataset_test = np.loadtxt(data_test, delimiter=",", skiprows=1)

# print "Shape of the test dataset:", dataset_test.shape

X_test = dataset_test[:,1:]


# (num_rows, num_cols) = X_test.shape
X_test = enc.transform(X_test)

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

 # %% grid search rf
# print "Start grid search"
# start_time = time.time()
# param_grid = {'n_estimators': np.linspace(100,1000,5).astype(int), \
# 'min_samples_split' : [4,5,6,10]}
# gs = GridSearchCV(estimator=RandomForestClassifier(min_samples_leaf=1), \
# param_grid=param_grid, scoring='roc_auc')
# gs.fit(X, target)
# print("--- GS RF train %s seconds ---" % (time.time() - start_time))

# predicted_gs_rf = get_proba_one(gs, X_test)
# write_submission('predictions/gs_rf_ohe.csv', predicted_gs_rf)

#%% Best Tuned RF

start_time = time.time()
rf_tuned = RandomForestClassifier(min_samples_leaf=1, min_samples_split=5, \
n_estimators = 500)
rf_tuned.fit(X,target)
print_time('RF tuned', start_time)

predicted_rf_tuned = get_proba_one(rf_tuned, X_test)
write_submission('predictions/rf_tuned_ohe.csv', predicted_rf_tuned)

#%% GBC
start_time = time.time()
gbc = GradientBoostingClassifier(n_estimators=500, min_samples_split=5)
gbc.fit(X, target)
print_time('GBC', start_time)

predicted_gbc = get_proba_one(gbc, X_test)
write_submission('predictions/gbc_ohe.csv', predicted_gbc)

#%% LR
runModel(LogisticRegression(), 'lr', X.toarray(), target, X_test)

#%%
param_lr = {'C': [1.5, 2, 2.5, 3, 3.5, 5, 5.5],
                           'class_weight': ['auto']}


 #%% results
print "Vanilla Logistic Regression ROC:", 0.53115
print "Random Forest ROC:", 0.84669
print "Random Forest 100 trees ROC:", 0.84731
print "Random Forest GS 5,2,120 ROC:",0.86971

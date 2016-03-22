# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 23:15:15 2016


"""

import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import time
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import pandas as pd
import pickle
import os

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

#%% helper functions
def write_submission(filename, predicted_proba):
    submission = np.column_stack((ID, predicted_proba))
    df = pd.DataFrame(data=submission, columns = head)
    df['Id'] = df['Id'].astype('int')
    df.to_csv(filename, index=False , header=head)
    
def get_proba_one(model, X):
    predicted = model.predict_proba(X)
    return predicted[:, 1]


#%% grid search rf
print "Start grid search"
start_time = time.time()
param_grid = {'n_estimators': np.linspace(100,1000,5).astype(int), \
'min_samples_split' : [4,5,6,10]}
gs = GridSearchCV(estimator=RandomForestClassifier(min_samples_leaf=1), \
param_grid=param_grid)
gs.fit(X, target)
print("--- GS RF train %s seconds ---" % (time.time() - start_time))

predicted_gs_rf = get_proba_one(gs, X_test)
write_submission('predictions/gs_rf.csv', predicted_gs_rf)

#%% Best Tuned RF

start_time = time.time()
rf_tuned = RandomForestClassifier(min_samples_leaf=1, min_samples_split=5, \
n_estimators = 500)
rf_tuned.fit(X,target)
print("--- RF tuned train %s seconds ---" % (time.time() - start_time))

predicted_rf_tuned = get_proba_one(rf_tuned, X_test)
write_submission('predictions/rf_tuned.csv', predicted_rf_tuned)

#%% results
print "Vanilla Logistic Regression ROC:", 0.53115
print "Random Forest ROC:", 0.84669
print "Random Forest 100 trees ROC:", 0.84731
print "Random Forest GS 5,2,120 ROC:",0.86971








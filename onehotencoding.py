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
from sklearn.ensemble import VotingClassifier
import XGBClassifier

os.chdir('/Users/jessicahoffmann/Desktop/Cours/UTAustin/S2/LargeScaleML/PS3')

#%% Load/prepare data
with open('train.csv') as data_train:
    dataset = np.loadtxt(data_train, delimiter=",", skiprows=1)
    
#print "Shape of the dataset:", dataset.shape

x = dataset[:,1:]
target = dataset[:,0]

# print X
# sorted_mappings = []
# (num_rows, num_cols) = X.shape

enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(x)

# print X
with open('test.csv') as data_test:
    dataset_test = np.loadtxt(data_test, delimiter=",", skiprows=1)

# print "Shape of the test dataset:", dataset_test.shape

x_test = dataset_test[:,1:]


# (num_rows, num_cols) = X_test.shape
X_test = enc.transform(x_test)

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

#%% RF grid search
param_rf_ohe = {'n_estimators' : [400, 500, 600, 700], \
'min_samples_split' : [5,7,10,12]}
                           
gs_rf_ohe = runModel(GridSearchCV(estimator=RandomForestClassifier(min_samples_leaf=1), \
param_grid=param_rf_ohe, scoring='roc_auc'), \
'gs_rf_ohe', X, target, X_test)                           



#%% Best Tuned RF
rf_tuned_ohe = runModel(RandomForestClassifier(min_samples_leaf=1, min_samples_split=5, \
n_estimators = 500), \
'rf_tuned_ohe', X, target, X_test)

#%% bagged rf
param_rf = {'n_estimators' : [450, 475, 500, 525, 550]}
rf_bagged_ohe = runModel(GridSearchCV(RandomForestClassifier(min_samples_leaf=1, min_samples_split=5), \
param_grid=param_rf, scoring='roc_auc'), \
'gs_rf_ohe', X, target, X_test)


#%% GBC
param_gbc_ohe = {'max_features': [4, 5, 6, 7], 'learning_rate': [.05, .08, .1], \
'max_depth': [8, 10, 13]}

gs_gbc_ohe = runModel(GridSearchCV(estimator=GradientBoostingClassifier(), \
param_grid=param_gbc_ohe, scoring='roc_auc'), \
'gs_gbc_ohe', X.toarray(), target, X_test.toarray()) 

#%% Best gbc
gbc_tuned = runModel(GradientBoostingClassifier(max_features = 6, learning_rate=.1, max_depth=8), \
'gbc_tuned_ohe', X.toarray(), target, X_test.toarray())   

   
#%% GS_LR
param_lr_ohe = {'C': [1.5, 2, 2.5, 3, 3.5, 5, 5.5]}
                           
gs_lr_ohe = runModel(GridSearchCV(estimator=LogisticRegression(class_weight='balanced'), \
param_grid=param_lr_ohe, scoring='roc_auc'), \
'gs_lr_ohe', X, target, X_test)                           

#%% best lr
lr_tuned_ohe = runModel(LogisticRegression(C=1.5, class_weight='balanced'), \
'lr_ohe', X, target, X_test)  

#%% SVC
param_svm = {'C' : [0.09, 0.2, 0.3]}
svm_linear009 = runModel(SVC(kernel='linear', probability=True, class_weight='balanced', C=0.09), \
'gs_svm_linear', X, target, X_test)
         
#%% XBG
xgb_tuned = runModel(XGBClassifier(max_depth=200, n_estimators=100, colsample_bytree = 0.4), \
'xgb_tuned', x, target, x_test) 

#%% Voting classifier
vc_ohe = runModel(VotingClassifier(estimators=[('lr', lr_tuned_ohe), ('rf', rf_tuned_ohe), ('SVC', svm_linear009)], \
voting='soft'), 'vc_ohe', X, target, X_test)



#%% Stacking
rf_train = rf_tuned_ohe.predict_proba(X)
#svc_train = svm_linear009.predict_proba(X)
lr_train = lr_tuned_ohe.predict_proba(X)
xgb_train = xgb_tuned.predict_proba(x)
print "end train"

rf_test = rf_tuned_ohe.predict_proba(X_test)
#svc_test = svm_linear009.predict_proba(X_test)
lr_test = lr_tuned_ohe.predict_proba(X_test)
xgb_test = xgb_tuned.predict_proba(x_test)
print "end test"

stack = np.column_stack((rf_train, lr_train, xgb_train))
stack_test = np.column_stack((rf_test, lr_test, xgb_test))

print "end stacking"


#%% LR on stack
lr_stack = runModel(LogisticRegression(C=1.5, class_weight='balanced'), \
'lr_stack', stack, target, stack_test)  




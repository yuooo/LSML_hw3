# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:25:12 2016

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

#print "lapin"
#
#os.chdir('/Users/jessicahoffmann/Desktop/Cours/UTAustin/S2/LargeScaleML/PS3')

from hw3q3 import write_submission, get_proba_one
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

#%%
start_time = time.time()
rf_tuned = RandomForestClassifier(min_samples_leaf=1, min_samples_split=5, n_estimators = 500)
rf_tuned.fit(X,target)
print("--- RF tuned train %s seconds ---" % (time.time() - start_time))

predicted_rf_tuned = get_proba_one(rf_tuned, X_test)
write_submission('predictions/rf_tuned.csv', predicted_rf_tuned)

#%% SVM linear train - bagged
n_estimators = 10
svm_linear = BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), \
max_samples=1.0 / n_estimators, n_estimators=n_estimators)
start_time = time.time()
svm_linear.fit(X, target)
print("--- Linear SVM train %s seconds ---" % (time.time() - start_time))

##%% Pickle
#file_name = 'models/svm_linear_model'
#with open(file_name,'wb') as f:
#    pickle.dump(svm_linear,f) 
#print f
#f.close()
#
##%% Unpickle
#file_name = 'models/svm_linear_model'
#with open(file_name,'rb') as f:
#    svm_linear = pickle.load(f) 

#%% SVM Linear test
print "SVM Linear train accuracy:", svm_linear.score(X, target)

predicted_svm_linear = get_proba_one(svm_linear, X_test)
write_submission('predictions/svm_linear.csv', predicted_svm_linear)


##%% SVM poly
#start_time = time.time()
#svm_poly = SVC(kernel = 'poly', probability=True)
#svm_poly.fit(X, target)
#print("--- Poly SVM train %s seconds ---" % (time.time() - start_time))
##
###%% Pickle
##file_name = 'svm_poly_model'
##with open(file_name,'wb') as f:
##    pickle.dump(svm_poly,f) 
#
##%% SVM poly test
#print "SVM Poly train accuracy:", svm_poly.score(X, target)
#
#predicted_svm_poly = get_proba_one(svm_poly, X_test)
#write_submission('predictions/svm_poly.csv', predicted_svm_poly)
#

##%% SVM RBF
#start_time = time.time()
#svm_rbf = sk.svm.SVC(kernel = 'rbf')
#svm_rbf.fit(X, target)
#print("--- rbf SVM train %s seconds ---" % (time.time() - start_time))
#
##%% SVM rbf test
#print "SVM rbf train accuracy:", svm_rbf.score(X, target)
#predicted_svm_rbf = svm_rbf.predict(X_test)
#submission = np.column_stack((ID, predicted_svm_rbf))
#df = pd.DataFrame(data=submission, columns = ['Id', 'Action'])
#df['Id'] = df['Id'].astype('int')
#df.to_csv("predictions/svm_rbf.csv", index=False , header=head)
#


#%% Random forests
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=20)
rf.fit(X, target)
print("--- RF train %s seconds ---" % (time.time() - start_time))

print "Random Forest train accuracy:", rf.score(X,target)

##%% Pickle
#file_name = 'rf_model'
#with open(file_name,'wb') as f:
#    pickle.dump(rf,f) 

#%% RF test
predicted_rf = get_proba_one(rf, X_test)
write_submission('predictions/rf.csv', predicted_rf)


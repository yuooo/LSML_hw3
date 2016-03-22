# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 23:15:15 2016


"""

import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import time
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import pandas as pd
import pickle

#%% Load/prepare data
with open('train.csv') as data_train:
    dataset = np.loadtxt(data_train, delimiter=",", skiprows=1)
    
#print "Shape of the dataset:", dataset.shape

X = dataset[:,1:]
target = dataset[:,0]

with open('test.csv') as data_test:
    dataset_test = np.loadtxt(data_test, delimiter=",", skiprows=1)

#print "Shape of the test dataset:", dataset_test.shape

X_test = dataset_test[:,1:]
ID = (dataset_test[:,0]).astype(int)


head = ['Id', 'Action']

#%% SVM linear train
start_time = time.time()
svm_linear = SVC(kernel = 'linear', probability=True)
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
predicted_svm_linear = svm_linear.predict_proba(X_test)
predicted_svm_linear = predicted_svm_linear[:, 1]
submission = np.column_stack((ID, predicted_svm_linear))
df = pd.DataFrame(data=submission, columns = head)
df['Id'] = df['Id'].astype('int')
df.to_csv("predictions/svm_linear.csv", index=False , header=head)


#%% SVM poly
start_time = time.time()
svm_poly = SVC(kernel = 'poly', probability=True)
svm_poly.fit(X, target)
print("--- Poly SVM train %s seconds ---" % (time.time() - start_time))
#
##%% Pickle
#file_name = 'svm_poly_model'
#with open(file_name,'wb') as f:
#    pickle.dump(svm_poly,f) 

#%% SVM poly test
print "SVM Poly train accuracy:", svm_poly.score(X, target)
predicted_svm_poly = svm_poly.predict_proba(X_test)
predicted_svm_poly = predicted_svm_poly[:, 1]
submission = np.column_stack((ID, predicted_svm_poly))
df = pd.DataFrame(data=submission, columns = ['Id', 'Action'])
df['Id'] = df['Id'].astype('int')
df.to_csv("predictions/svm_poly.csv", index=False , header=head)


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
rf = RandomForestClassifier(min_samples_leaf=20)
rf.fit(X, target)
print("--- RF train %s seconds ---" % (time.time() - start_time))

print "Random Forest train accuracy:", rf.score(X,target)

##%% Pickle
#file_name = 'rf_model'
#with open(file_name,'wb') as f:
#    pickle.dump(rf,f) 

#%% RF test
predicted_rf = rf.predict_proba(X_test)
predicted_rf = predicted_rf[:,1]
submission = np.column_stack((ID, predicted_rf))
df = pd.DataFrame(data=submission, columns = head)
df['Id'] = df['Id'].astype('int')
df.to_csv("predictions/rf.csv", index=False , header=head)

#%% results
print "Vanilla Logistic Regression ROC:", 0.53115
print "Random Forest ROC:", 0.82773







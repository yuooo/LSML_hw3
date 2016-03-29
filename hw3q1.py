# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:44:03 2016
"""

import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import pandas as pd


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
    
#%% Run LR
lr_vanilla = runModel(LogisticRegression(), \
'lr_vanilla', x, target, x_test)  





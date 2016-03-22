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
    
print "Shape of the dataset:", dataset.shape

X = dataset[:,1:]
target = dataset[:,0]

#print "Shape of X:", X.shape
#print "Shape of target:", target.shape


#%% Vanilla logistic regression
model = LogisticRegression()
model.fit(X, target)

#print(model)

print "Train accuracy:", model.score(X,target)

#%% Test
with open('test.csv') as data_test:
    dataset_test = np.loadtxt(data_test, delimiter=",", skiprows=1)

print "Shape of the test dataset:", dataset_test.shape

X_test = dataset_test[:,1:]
ID = dataset_test[:,0]

predicted_vlr = model.predict_proba(X_test)
predicted_vlr = predicted_vlr[:,1]
submission = np.column_stack((ID, predicted_vlr))

#%% write
head = ['Id', 'Action']
df = pd.DataFrame(data=submission, columns = head)
df['Id'] = df['Id'].astype('int')
df.to_csv("predictions/vlr.csv", index=False , header=head)




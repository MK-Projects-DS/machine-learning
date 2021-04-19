# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:17:22 2020

@author: mpkan
"""

import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt 
import sys
import ctypes

data = np.loadtxt("timeseries_data.txt")

reg = linear_model.LinearRegression()
testlabels=data[:,-1]
testlabels=testlabels[:,np.newaxis]
window=4
n=len(data[:,0])
m=len(data[0,:])
predicted=np.empty(n)

for x in range(n):
    trainlabels=data[x,window:-1]
    trainlabels=trainlabels[:,np.newaxis]
    test=data[x,-window-1:-1]
    test=test[np.newaxis,:]
    train=[]

    for y in range(m-window-1):
        X_w = data[x,y:y+window]
        X_w=X_w[np.newaxis,:]
        if len(train)==0:
            train=X_w
        else:
            train=np.append(train,X_w,axis=0)
    reg.fit(train,trainlabels)
    predicted[x]=reg.predict(test)
print(predicted)
print("mean squared error",mean_squared_error(testlabels,predicted))

plt.plot(testlabels)
plt.plot(predicted)
plt.legend()
plt.show()









    
    
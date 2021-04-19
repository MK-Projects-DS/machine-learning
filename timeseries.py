# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:43:17 2020

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


#data = np.loadtxt("data1.txt")
data = np.loadtxt("timeseries_data.txt")
n=len(data[0,:])
m=len(data[:,0])
train=np.arange(0,n,1)
train=train[:-1][:,np.newaxis]
test=train[-1][:,np.newaxis]
testlabels=data[:,-1]
regmodel = linear_model.LinearRegression()
reg_x=np.empty(m)
start=20
predicted=np.empty(m)
for x in range(m):
    trainlabels=data[x,:-1]
    reg = regmodel.fit(train,trainlabels)
    predicted[x]=regmodel.predict(test)
print(len(predicted),m,predicted)
print("mean squared error",mean_squared_error(testlabels,predicted))


"""
pred = reg.predict(train)
print("\npredicted train",pred)
testpred = reg.predict(X_test)
print("\ntestlabels",y_test,"\npredicted test",testpred)
print("rsquare",r2_score(y_test,testpred))
print("median absolute error",median_absolute_error(y_test,testpred))



n=len(train[:,0])
nt=len(test[:,0])
t1=np.ones((n,1))
t2=np.ones((nt,1))

train = np.append(train,t1,axis=1)
test = np.append(test,t2,axis=1)

w = np.random.rand(len(train[0,:]))

h = train.dot(w)
diff = h-trainlabels
obj = np.sum(np.square(diff))/n
print("\ninitial obj",obj)
eta=0.001
prev=obj+10

while prev-obj>0.0001:
    grad = 2*train.dot(diff)/n
    w=w-eta*grad
    print("w",w,"\ngrad",grad)
    
    h = train.dot(w)
    diff = h-trainlabels
    prev=obj
    obj = np.sum(np.square(diff))/n
    print("diff","\nobj",obj)

print("\nlabels",trainlabels,"\npredicted",train.dot(w))

testpred = train.dot(w)
print("\ntestlabels",testlabels,"\nmtest prediction",testpred)

 """   

   


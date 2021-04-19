# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:31:19 2020

@author: mpkan
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


data = np.loadtxt("bc.train.0")
#data = np.loadtxt("data.txt")
trainlabels = data[:,0]
train = data[:,1:]
#norm = np.linalg.norm(train, axis = 0)
#train = train/norm
train = (train-np.amin(train,axis=0))/(np.amax(train,axis=0)-np.amin(train,axis=0))

test = np.loadtxt("bc.test.0")
#test = np.loadtxt("testdata.txt")
testlabels = test[:,0]
test = test[:,1:]
test = (test-np.amin(test,axis=0))/(np.amax(test,axis=0)-np.amin(test,axis=0))

n = len(train[:,0])
m = len(train[0,:])
w = np.random.rand(10,m)
b = np.random.rand(10)
C = 0.0000002

for i in range (10):
    
    print(w[i])
    
    h_loss = 1-(trainlabels*(train.dot(w[i])+b[i]))
    is_loss = np.greater(h_loss, 0)
    loss = np.sum(np.multiply(h_loss,is_loss))/n
   
    prev = loss + 10

    eta = 0.1
    print(prev,loss)
    
    while prev - loss > 0.0000001:
        grad = -trainlabels[:,np.newaxis] * train
        print(grad*(is_loss[:,np.newaxis]))
        grad = np.sum(grad * (is_loss[:,np.newaxis]), axis = 0)/n
    
        w[i] = w[i] - (C*np.linalg.norm(w[i])) - (eta*grad)
        b[i] = b[i] - (0.01*-np.sum(trainlabels*is_loss))/n
       
        h_loss = 1-(trainlabels*(train.dot(w[i])+b[i]))
        is_loss = np.greater(h_loss, 0)
        prev = loss 
        loss = np.sum(np.multiply(h_loss,is_loss))/n 
        print("prev-loss",prev-loss)
        
        eta = eta 
        y_pred = np.sign(train.dot(w[i])+b[i])
        print(accuracy_score(trainlabels,y_pred),np.sum(y_pred!=trainlabels))
            
        print(prev,loss)
print("w[i]",w,"b",b)
for i in range(10):
    y_pred = np.sign(test.dot(w[i])+b[i])
    print(accuracy_score(testlabels,y_pred),np.sum(y_pred!=testlabels))    
    
   
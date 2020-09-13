# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:09:55 2020

@author: mpkan
"""
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.spatial import distance
from numpy import array
import sys


#data = np.loadtxt("data1.txt")
data = np.loadtxt("bc.train.0")
trainlabels = data[:,0]
train = data[:,1:]

data = np.loadtxt("bc.test.0")
testlabels = data[:,0]
test = data[:,1:]

ones = np.ones((1,len(train[0,:])))
train_temp = np.append(train, ones, axis=0)
column_norm = np.linalg.norm(train_temp,axis=0)
train = train/column_norm
test = test/column_norm


m = len(train[0,:])

b = .01*np.random.rand(1)
w = .01*np.random.rand(m)
w_b = np.concatenate((w,b))

eta = 0.001
stop = 0.001

obj = np.sum(np.square((train.dot(w) + b) - trainlabels))
prev = obj + 10

while (prev - obj > stop):   

    prev = obj
    
    diff = (trainlabels - (train.dot(w) + b))
    
    grad = train * diff[:,np.newaxis]
    grad = np.sum(grad, axis=0)
    
    w = w + eta * grad
    b = b + eta * np.sum(diff)
    obj = np.sum(np.square((train.dot(w) + b) - trainlabels))

    

predictions = np.sign(train.dot(w) + b)
print("no of errors=", np.sum(predictions != trainlabels))

error = np.sum(np.abs(.5*(predictions - trainlabels)))
print("error=", error)
err_rate = error/train.shape[0]
print("error rate=",err_rate)

test_predictions = np.sign(test.dot(w) + b)
test_error = np.sum(test_predictions != testlabels)
print("no of test errors=", test_error)
test_err_rate = test_error/test.shape[0]
print("test error rate=", test_err_rate)


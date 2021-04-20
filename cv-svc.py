# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:35:09 2020

@author: mpkan
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sys

'''
data = np.loadtxt("ion.train.0")
#data = np.loadtxt("data.txt")
trainlabels = data[:,0]
train = data[:,1:]
#norm = np.linalg.norm(train, axis = 0)
#train = train/norm
#train = (train-np.amin(train,axis=0))/(np.amax(train,axis=0)-np.amin(train,axis=0))

       
test = np.loadtxt("ion.test.0")
#test = np.loadtxt("testdata.txt")
#test = np.loadtxt("data.txt")
testlabels = test[:,0]
test = test[:,1:]
#test = (test-np.amin(test,axis=0))/(np.amax(test,axis=0)-np.amin(test,axis=0))
'''

train = np.load("X_train.npy")
trainlabels = np.load("y_train.npy")
test = np.load("X_test.npy")
testlabels = np.load("y_test.npy")

######## train z ################

m = len(train[:,0])
n = len(train[0,:])
k = 10000

w = np.random.rand(k,n)
w = 2*w - 1 
w_x = train.dot(w.T)

z = w_x 
a= np.amin(z,axis=0)
b= np.amax(z,axis=0)

w0 = np.random.rand(k)
w0 = w0 * (b-a)
w0 = w0 + a
z = z+w0
z = np.sign(z)

######## test z ########################

m = len(test[:,0])
n = len(test[0,:])

w_x = test.dot(w.T)
z_ = w_x 
z_ = z_+ w0
z_ = np.sign(z_)

################# svm ##############

print("original")
print(train.shape)
clf = LinearSVC(max_iter=10000)
clf.fit(train,trainlabels)
pred = clf.predict(test)
#print(pred)
#print(testlabels)
error=np.sum(pred != testlabels)
print(error/len(testlabels))

print("new data")
print(z.shape)
clf.fit(z,trainlabels)
pred = clf.predict(z_)
print(pred)

print(testlabels)
error=np.sum(pred != testlabels)
print(error/len(testlabels))


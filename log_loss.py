# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:22:32 2020

@author: mpkan
"""
import numpy as np
import sys

train = np.loadtxt("bc.train.0")
#train = np.loadtxt("data.txt")

m = len(train[:,0])
trainlabels = train[:,0]
trainlabels[trainlabels == -1] = 0
#print("trainlabels", trainlabels)

train = train[:,1:]
train = train/np.linalg.norm(train,axis=0)
train = np.append(train, np.ones((m))[:,np.newaxis], axis = 1)

train_l = np.append(trainlabels[:,np.newaxis], train, axis = 1)

train_p = train[np.where(train_l[:,0]==1)]
train_0 = train[np.where(train_l[:,0]==0)]

#print("train_p", train_p)

trainlabels_p = trainlabels[np.where(trainlabels == 1)]
trainlabels_0 = trainlabels[np.where(trainlabels == 0)]
#print("trainlabels_p",trainlabels_p)
w = np.random.rand(len(train[0,:]))
w = 2*w - 1
w = w * 1
#w = np.array([1,0,-3])
print("m", m, "w",w)

#sys.exit()
#print("exp train dot w",1/np.exp(-train.dot(w)))
 
obj = 1/(1+np.exp(-train.dot(w)))
obj_p = 1/(1+np.exp(-train_p.dot(w)))
obj_0 = 1/(1+np.exp(-train_0.dot(w)))

#print(train.dot(w))
print("obj", obj, "obj_0", obj_0) 

#print("log obj_p", np.log(obj_p))   

log_loss = -1/m*(trainlabels_p.dot(np.log(obj_p)) + (1 - trainlabels_0).dot(np.log(1 - obj_0)))
prev_loss = log_loss + 10
eta = .1
print("log_loss",log_loss)

while prev_loss - log_loss > .00001:
    prev_loss = log_loss
    
    ot = obj - trainlabels
    tot = train * ot[:,np.newaxis]    
    d_dw = 1/m*np.sum(tot, axis=0)

#    print(d_dw)
    
    w = w - eta * d_dw
    
    obj = 1/(1+np.exp(-train.dot(w)))
    obj_p = 1/(1+np.exp(-train_p.dot(w)))
    obj_0 = 1/(1+np.exp(-train_0.dot(w)))
    log_loss = -1/m*(trainlabels_p.dot(np.log(obj_p)) + (1 - trainlabels_0).dot(np.log(1 - obj_0)))
    print("prev_loss",prev_loss, "log_loss",log_loss)
    
#    print("obj_p", obj_p, "obj_0", obj_0)
#    print("errors",np.sum(obj_p < 0.5) + np.sum(obj_0 > 0.5))
    
print(obj)  
print("errors",np.sum(obj_p < 0.5) + np.sum(obj_0 > 0.5))
print(w)



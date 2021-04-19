# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:54:00 2020

@author: mpkan
"""
import numpy as np



data = np.loadtxt("ion.train.0")
testdata = np.loadtxt("ion.test.0")
train = data[:,1:]
trainlabels = data[:,0]
test = testdata[:,1:]
testlabels = np.loadtxt("naive_bayes.txt") 
testlabels = testlabels[:,0]

n = len(train[:,0])
trainm1 = train[np.where(trainlabels==-1)]
train1 = train[np.where(trainlabels==1)]

m_j = np.ones((1,len(train[0,:])))
m_j = m_j*0.01
trainm1 = np.append(trainm1,m_j,axis=0)
train1 = np.append(train1,m_j,axis=0)
print("x",trainm1)
train_meanm1 = np.sum(trainm1,axis=0)/(len(trainm1[:,0])-1)
train_mean1 = np.sum(train1,axis=0)/(len(train1[:,0])-1)

trainm1 = trainm1[:-2,:]
train1 = train1[:-2,:]
sm1 = np.sqrt(np.sum(np.square(trainm1 - train_meanm1),axis=0)/(len(trainm1[:,0])-1))
s1 = np.sqrt(np.sum(np.square(train1 - train_mean1),axis=0)/(len(train1[:,0])-1))

print("trainmean",train_meanm1,train_mean1,"\n sm",sm1,s1)

class_xp = np.greater(np.sum(np.square((test-train_meanm1)/sm1),axis=1),np.sum(np.square((test-train_mean1)/s1),axis=1))

error = np.sum(class_xp != testlabels)    

print("class_xp",class_xp, error)
                                                     


 

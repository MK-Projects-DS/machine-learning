# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:46:10 2020

@author: mpkan
"""

from sklearn import svm
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


data = np.loadtxt("bc.train.0")
#data = np.loadtxt("ion.train.0")
#data = np.loadtxt("data.txt")

trainlabels = data[:,0]
train = data[:,1:]

#test = np.loadtxt("testdata.txt")
test = np.loadtxt("bc.test.0")
#test = np.loadtxt("ion.test.0")
testlabels = test[:,0]
test = test[:,1:]

svm_clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=0.5, loss="hinge"))])

svm_clf.fit(train,trainlabels)
result = svm_clf.predict(test)
print(result , "errors",np.sum(result!=testlabels))

train_mean = np.mean(train, axis = 0)
trainlabels_mean = np.mean(trainlabels)
yi_ym = trainlabels - trainlabels_mean
yi_ym_sq = np.square(yi_ym)

n = len(train[0,:])
p = np.ones((n))

for i in range(n): 
    xi_xm = train[:,i] - train_mean[i]    
    xi_xm_sq = np.square(xi_xm)
    print(np.sqrt(np.sum(xi_xm_sq)*np.sum(yi_ym_sq)))
        
    p[i] = np.sum(np.multiply(xi_xm,yi_ym))/np.sqrt(np.sum(xi_xm_sq)*np.sum(yi_ym_sq))
    
print("p",p)
train = train[:,p.argsort()]  #Sorted train
train = train[:,1:20]         # No of features selected

test = test[:,p.argsort()]    #Sorted test
test = test[:,1:20]           # No of features selected

svm_clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=0.5, loss="hinge"))])

svm_clf.fit(train,trainlabels)
result = svm_clf.predict(test)
print(result , "errors",np.sum(result!=testlabels))










# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:41:10 2020

@author: mpkan
"""

from statistics import mode
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier


#data = np.loadtxt("data.txt")
data = np.loadtxt("bc.train.0")

#test = np.loadtxt("testdata.txt")
test = np.loadtxt("bc.test.0")
testlabels = test[:,0]
test = test[:,1:]

a = len(data[:,0])

def bagging():
    p = np.random.randint(0,10,a)
    p = p/np.sum(p)
    
    index = np.arange(a)

    index = np.random.choice(index,a,replace=True,p=p)

    return data[index]
    
    
svm_clf = SVC()
rig_clf = RidgeClassifier()
dtr_clf = DecisionTreeClassifier()

pred = np.empty((len(test[:,0]),1),float)
#pred = pred[:,np.newaxis]
print("shape", np.shape(pred))
final_pred = np.array([])
for i in range(4):
    mybag = bagging()
    trainlabels = mybag[:,0]
    train = mybag[:,1:]
    
#    svm_clf.fit(train,trainlabels)
    svm_clf.fit(train,trainlabels)
#    y_pred = svm_clf.predict(tes    
    y_pred = svm_clf.predict(test)
    print("accuracy score", accuracy_score(testlabels,y_pred))
    pred = np.append(pred,y_pred[:,np.newaxis],axis=1)
    print(pred)
  

for i in range(len(test[:,0])):
    final_pred = np.append(final_pred,mode([pred[i,0],pred[i,1],pred[i,2],pred[i,3],pred[i,4]]))  
    print("final", final_pred)

print("accuracy score", accuracy_score(testlabels,final_pred))    
    
    


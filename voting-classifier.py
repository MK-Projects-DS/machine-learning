# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:25:59 2020

@author: mpkan
"""

import numpy as np
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#data = np.loadtxt("data.txt")
data = np.loadtxt("bc.train.0")

trainlabels = data[:,0]
train = data[:,1:]

#test = np.loadtxt("testdata.txt")
test = np.loadtxt("bc.test.0")
testlabels = test[:,0]
test = test[:,1:]

#lin_clf = LinearRegression()
#log_clf = LogisticRegression(max_iter=200)
dtr_clf = DecisionTreeClassifier()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators=[('lr',dtr_clf),('rf',rnd_clf),('svc',svm_clf)], voting = 'hard')
voting_clf.fit(train,trainlabels)

for clf in (dtr_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(train,trainlabels)
    y_pred = clf.predict(test)
    print(clf.__class__.__name__, accuracy_score(testlabels,y_pred))
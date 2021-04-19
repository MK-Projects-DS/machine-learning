# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:57:53 2020

@author: mpkan
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold

#data = np.loadtxt("bc.train.0")
#data = np.loadtxt("ion.train.0")
data = np.loadtxt("data.txt")

trainlabels = data[:,0]
train = data[:,1:]

test = np.loadtxt("testdata.txt")
#test = np.loadtxt("bc.test.0")
#test = np.loadtxt("ion.test.0")
testlabels = test[:,0]
test = test[:,1:]

selector = VarianceThreshold(threshold=3)
train = selector.fit_transform(train)

print("new train", train)
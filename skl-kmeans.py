# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:16:38 2020

@author: mpkan
"""

from sklearn.cluster import KMeans
import numpy as np

train = np.loadtxt("data.txt")

kmeans = KMeans(n_clusters=4,random_state=0).fit(train)
print("labels",kmeans.labels_)
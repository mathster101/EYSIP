# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:12:47 2020

@author: Mathew
"""
import warnings
from sklearn import cluster
from sklearn import metrics
#warnings.filterwarnings("ignore")
NUM_CLUSTERS = 10
from gensim.models import Word2Vec
sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
            ['this', 'is',  'another', 'book'],
            ['one', 'more', 'book'],
            ['this', 'is', 'the', 'new', 'post'],
                        ['this', 'is', 'about', 'machine', 'learning', 'post'],  
            ['and', 'this', 'is', 'the', 'last', 'post']]

model = Word2Vec(sentences,size = 100, min_count=1)

X = model[model.wv.vocab]

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
 
print ("Cluster id labels for inputted data")
print (labels)

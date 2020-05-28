# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:28:11 2020

@author: Mathew
"""

from sklearn import cluster
from sklearn import metrics
from imports import *
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 200)
candidate_sentences = pd.read_csv("D:\Downloads\sen.csv")
entity_pairs = []
k =candidate_sentences["sentence"]


for i in tqdm(k):
  entity_pairs.append(get_entities(i))
relations = [get_relation(i) for i in tqdm(k)]
source = [i[0] for i in entity_pairs]
target = [i[1] for i in entity_pairs]
triplets = []
for i in range(len(relations)):
   temp = [source[i],relations[i],target[i]]
   triplets.append(temp)
del source, target, relations, temp


#%%
NUM_CLUSTERS = 10

model = Word2Vec([[row[2]] for row in triplets],size = 100, min_count=1)

X = model[model.wv.vocab]

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
y_kmeans = kmeans.predict(X)
#scatter(X,centroids,y_kmeans)
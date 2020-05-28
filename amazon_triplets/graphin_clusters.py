# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:39:16 2020

@author: Mathew
"""
from sklearn import cluster
from sklearn import metrics
import json
import numpy as np
import spacy
import neuralcoref
from gensim.models import Word2Vec
from imports import *
from matplotlib import pyplot as plt
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

PATH = '../../Datasets/lstm2/'
topics = ['electronics', 'home', 'pets', 'clothing', 'toys']
VAL_PER_TOPIC= 20
NUM_CLUSTERS = 5

def fetch_data(PATH,topics):
   #get data from json files
   raw =[]
   for val in topics:
      count = 1
      for line in open(PATH + val + '.json','r'):
         raw.append([json.loads(line),val])
         count+=1
         if count>VAL_PER_TOPIC:
            break
   return raw

def resolve(raw):
   # perform coreference resolution
   text_ref = []
   for point in tqdm(raw):
      review = point[0]['reviewText']
      doc = nlp(review)
      text_ref.append(doc._.coref_resolved)
   return text_ref

def splitter(text):
   #split paragraphs into individual sentences
   sentences = []
   for para in text:
      lines = para.split('.')
      for l in lines:
         if len(l)>0:
            sentences.append(l.strip()+ '.')
   return sentences

def trip_gen(k):
   #generate triplets
   entity_pairs = []
   for i in tqdm(k):
     entity_pairs.append(get_entities(i))
   relations = [get_relation(i) for i in tqdm(k)]
   source = [i[0] for i in entity_pairs]
   target = [i[1] for i in entity_pairs]
   triplets = []
   for i in range(len(relations)):
      temp = [source[i],relations[i],target[i]]
      triplets.append(temp)
   return triplets

def get2d(triplets,dims):
   #perform w2v embedding
   x = []
   y = []
   model = Word2Vec(triplets,size =dims, min_count=1)
   X = model[model.wv.vocab]
   for i in X:
      x.append(i[0])
      y.append(i[1])
   return x,y,X
   
raw = fetch_data(PATH,topics)
text = resolve(raw)
sentences = splitter(text)
del text, raw, topics
triplets = trip_gen(sentences)
#%%
x,y,X = get2d(triplets,2)

plt.scatter(x,y)
plt.show()
#%%
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
y_kmeans = kmeans.predict(X)
scatter(X,centroids,y_kmeans)







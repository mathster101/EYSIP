# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:21:29 2020

@author: Mathew
"""
#'electronics','home','pets','clothing','toys'
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import random

PATH = '../Datasets/lstm2/'
f = open(PATH + 'data.csv')
m = f.readlines()
print('file read')
random.shuffle(m)
print("dataset shuffled")
documents = m[:3000]
documents = [(x.split(',')[0] +x.split(',')[1])  for x in documents]
f.close()



vect = TfidfVectorizer(stop_words='english')
X = vect.fit_transform(documents)
df = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
clusters = 5
model = KMeans(n_clusters=clusters)
model.fit(X)

print("Top terms per cluster:\n________________________")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(clusters):
    print("\nCluster %d:" % i)
    for ind in order_centroids[i,:4]:
        print(terms[ind])


test1 = ['home']
test2 = ['my pet is very hyperactive and this toy works great for my puppy']
print("\n__________________\nPrediction")

Y = vect.transform(test1)
prediction = model.predict(Y)
print(prediction,test1)

Y = vect.transform(test2)
prediction = model.predict(Y)
print(prediction,test2)
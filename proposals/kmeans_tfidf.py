# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:06:30 2020

@author: Mathew
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:21:29 2020

@author: Mathew
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import random

PATH = 'D:\Documents\Datasets\proposals\proposals.csv'
PATH = 'resolved.txt'
f = open(PATH,'r',encoding="utf-8")
m = f.readlines()
print('file read')
random.shuffle(m)
print("dataset shuffled")
f.close()
documents = [i.lower() for i in m]
vect = TfidfVectorizer(stop_words='english')
X = vect.fit_transform(documents)
df = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
clusters = 4
model = KMeans(n_clusters=clusters)
model.fit(X)

print("Top terms per cluster:\n________________________")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(clusters):
    print("\nCluster %d:" % i)
    for ind in order_centroids[i,:6]:
        print(terms[ind])


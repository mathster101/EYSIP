# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:21:29 2020

@author: Mathew
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

documents = ["a car will often tend to have four wheels",
             "my laptop has an hd screen",
             "the most popular car brand in japan in suzuki",
             "very thin laptop is called an ultrabook",
             "one has to get a driving license to legally drive a car",
             "most IT professionals carry a laptop while travelling",
             "my first car didn't have working brakes",
             "the most important feature of a laptop is probably battery life"
             ]

vect = TfidfVectorizer(stop_words='english')
X = vect.fit_transform(documents)
df = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
clusters = 2
model = KMeans(n_clusters=clusters)
model.fit(X)

print("Top terms per cluster:\n________________________")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(clusters):
    print("\nCluster %d:" % i)
    for ind in order_centroids[i,:4]:
        print(terms[ind])



print("\n__________________\nPrediction")

Y = vect.transform(["used laptop for sale"])
prediction = model.predict(Y)
print(prediction,"<-- used laptop for sale")

Y = vect.transform(["green cars are the fastest "])
prediction = model.predict(Y)
print(prediction,"<-- green cars are the fastest ")
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

f = open('D:\Documents\Datasets\lstm1\data.csv','r')
data = f.readlines()
neigh = int(pow(len(data),0.5))
knn = KNeighborsClassifier(n_neighbors=50)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', knn),
])

random.shuffle(data)
j = data[-300:]
l = data[:-30000]

train_data = []
train_target = []
test_data = []
test_target = []
for i in l:
    t = i.split(',')
    train_data.append(t[0])
    train_target.append(int(t[1][:1]))

for i in j:
    t = i.split(',')
    test_data.append(t[0])
    test_target.append(int(t[1][:1]))

print('starting to fit data')    
text_clf.fit(train_data, train_target)
print('starting to predict')
predicted = text_clf.predict(test_data)
print('We got an accuracy of',np.mean(predicted == test_target)*100, '% over the test data.')
t = confusion_matrix(test_target,predicted)
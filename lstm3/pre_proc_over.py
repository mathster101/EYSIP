# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:58:54 2020

@author: Mathew
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:38:49 2020

@author: Mathew
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:43:55 2020

@author: Mathew
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import string 
import json
import re
from nltk.corpus import stopwords 
import numpy as np
import time
import io
import pickle
from config import *


COUNT = 1200000
class_count = {0: 0, 1: 0}

topics = ['electronics']
SAVE_CSV = 1
SAVE_NPY = 1

def clean_text (s, stp_wrd):
   s = s.lower()
   output = s.translate(str.maketrans('', '', string.punctuation))#remove punctuation
   output = re.sub(r'\d+', '',output)#remove digits
   temp = output.split(' ')
   output = ''
   for i in temp:#remove stopwords
      if i not in stp_wrd:
         output += i + ' '
   return output

def fetch_data(PATH,topics):
   raw =[]
   for val in topics:
      count = 1
      for line in open(PATH + val + '.json','r'):
         raw.append(json.loads(line))
         count+=1
         if count>VAL_PER_TOPIC:
            break
   return raw

raw_data = fetch_data(PATH,topics)
d = set(stopwords.words('english'))#set of stopwords
strlens = []
revs = []
labels = []

revs_t = []
labels_t = []

for point in raw_data:
   review = point['reviewText'] + ' ' + point['summary']
   review = clean_text(review,d)
   label = int(point['overall'])
   if label > 2:
      label = 1
   else:
      label = 0
   strlens.append(int(len(review.split(' '))))
   revs_t.append(review)
   labels_t.append(label)
flag = 0

while flag == 0:
   for i in range(len(revs_t)):
      if class_count[labels_t[i]] < COUNT:
         revs.append(revs_t[i])
         labels.append(labels_t[i])
         class_count[labels_t[i]] += 1
         
   if class_count[0] == class_count[1]:
      flag = 1
   else:
      print('going back for a new pass')
      flag = 0
   
      



   
if SAVE_CSV:
   f = open(PATH + 'data.csv', 'w')
   for i in range(len(revs)):
      f.write(revs[i]+','+str(labels[i])+'\n')
   f.close()
   print('csv generated')


with open(PATH + 'tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
token = tokenizer.texts_to_sequences(revs)
token = pad_sequences(token, maxlen=70)
X_train, X_test, Y_train, Y_test = train_test_split(token, labels, test_size = 0.20, random_state = 42)

if SAVE_NPY:
   np.save(PATH+'y_train_over',Y_train)
   np.save(PATH+'x_train_over',X_train)
   np.save(PATH+'y_test_over',Y_test)
   np.save(PATH+'x_test_over',X_test)
   np.save
   
   print('numpy files generated')



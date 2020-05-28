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

import pickle
from config import *
import tqdm


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
for point in tqdm.tqdm(raw_data):
   review = point['reviewText']
   review = clean_text(review,d)
   label = int(point['overall'])
   if label > 2:
      label = 1
   else:
      label = 0
   strlens.append(int(len(review.split(' '))))
   revs.append(review)
   labels.append(label)
   
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
token = pad_sequences(token, maxlen=50)
X_train, X_test, Y_train, Y_test = train_test_split(token, labels, test_size = 0.20, random_state = 42)

if SAVE_NPY:
   np.save(PATH+'y_train',Y_train)
   np.save(PATH+'x_train',X_train)
   np.save(PATH+'y_test',Y_test)
   np.save(PATH+'x_test',X_test)
   
   print('numpy files generated')



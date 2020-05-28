# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:58:33 2020

@author: Mathew
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import string 
import json
import re
from nltk.corpus import stopwords 
import numpy as np
import time
import pickle
from config import *
import random

PATH = '../../Datasets/lstm1/'
NUM_PER_SCORE = 50000
sc_count = {'1':0,'2':0,'3':0,'4':0,'5':0} #bias towards 1 star

f = open(PATH + 'data.csv','r')
lines = f.readlines()
random.shuffle(lines)
data = [x.split(',') for x in lines]

revs = []
y = []

for entry in data:
   score = entry[1][:1]
   if sc_count[score] < NUM_PER_SCORE:
      revs.append(entry[0])
      y.append(int(score))
      sc_count[score] += 1


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
with open(PATH + 'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
word_index = tokenizer.word_index
token = tokenizer.texts_to_sequences(revs)
token = pad_sequences(token, maxlen=120)

np.save(PATH + 'x_balance',token)
np.save(PATH + 'y_balance',np.asarray(y))
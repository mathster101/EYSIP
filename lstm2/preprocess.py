# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:00:05 2020

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

init_time  = time.time()
PATH = '../../Datasets/lstm2/'
topics = ['electronics', 'home', 'pets', 'clothing', 'toys']
#%%
VAL_PER_TOPIC = 10
SAVE_CSV = 1 
SAVE_NPY = 0


#function to clean text
#inputs: string, set of stop words
#return: cleaned string
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

#function to fetch datapoints
#inputs: file path, list of topics
#return: list of text,topic
def fetch_data(PATH,topics):
   raw =[]
   for val in topics:
      count = 1
      for line in open(PATH + val + '.json','r'):
         raw.append([json.loads(line),val])
         count+=1
         if count>VAL_PER_TOPIC:
            break
   return raw


raw_data = fetch_data(PATH,topics)

revs = []
labels = []                                                                

d = set(stopwords.words('english'))#set of stopwords

for point in raw_data:
   review = point[0]['reviewText']
   #review = clean_text(review,d)
   label = point[1]
   revs.append(review)
   labels.append(label)
#%%
if SAVE_CSV:
   f = open('../to_give/data.csv', 'w')
   for i in range(len(revs)):
      f.write(revs[i]+','+labels[i]+'\n')
   f.close()
   print('csv generated')
#%%
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(revs)
word_index = tokenizer.word_index
token = tokenizer.texts_to_sequences(revs)
token = pad_sequences(token, maxlen=250)

X_train, X_test, Y_train, Y_test = train_test_split(token, labels, test_size = 0.20, random_state = 42)
if SAVE_NPY:
   np.save('y_train',Y_train)
   np.save('x_train',X_train)
   np.save('y_test',Y_test)
   np.save('x_test',X_test)
   print('numpy files generated')

print('elapsed =',time.time() - init_time)
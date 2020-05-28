# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:00:11 2020

@author: Mathew
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from config import *

def build_model():# builds the model and returns it
   model = Sequential() #defining the model
   model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=250))
   model.add(SpatialDropout1D(0.2))
   model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
   model.add(Dense(5, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   print(model.summary())
   return model

def one_hot(s):
   if s == 'electronics':
      ret = [1,0,0,0,0]
   if s == 'home':
      ret = [0,1,0,0,0]
   if s == 'pets':
      ret = [0,0,1,0,0]
   if s == 'clothing':
      ret = [0,0,0,1,0]
   if s == 'toys':
      ret = [0,0,0,0,1]
   return ret

def load_data(PATH):
   a = np.load(PATH + 'x_train.npy') #load train Xs
   b = np.load(PATH + 'y_train.npy') #load train Ys
   c = np.load(PATH + 'x_test.npy') #load test Xs
   d = np.load(PATH + 'y_test.npy') #load test Ys
   return a,b,c,d

X_train, Y_tr, X_test, Y_te = load_data(PATH)

Y_train = []
Y_test = []

for s in Y_tr: #one hot encoding labels
   Y_train.append(one_hot(s))
for s in Y_te: #one hot encoding labels
   Y_test.append(one_hot(s))
   
Y_train = np.asarray(Y_train) #make it numpy arr
Y_test = np.asarray(Y_test)   #make it numpy arr


checkpoint = ModelCheckpoint(PATH + WEIGHT_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #checkpointing it
callbacks_list = [checkpoint,EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]

current_model = build_model() #retrieve model

history = current_model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=VALID_SPLIT,callbacks=callbacks_list)  #train time

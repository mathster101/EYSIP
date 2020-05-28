# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:56:34 2020

@author: Mathew
"""

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from config import *

def build_model():
   model = Sequential()
   model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=70))
   model.add(Dropout(0.2))
   model.add(Conv1D(250 ,kernel_size,padding='valid',activation='relu',strides=1))
   model.add(Dropout(0.2))
   model.add(Conv1D(500,kernel_size,padding='valid',activation='relu',strides=1))
   model.add(GlobalMaxPooling1D())
   model.add(Dense(50))
   model.add(Dropout(0.2))
   model.add(Activation('relu'))
   model.add(Dense(2, activation='softmax'))
   try:
      model.load_weights(PATH + WEIGHT_PATH)
   except:
      print('pre trained weights not found!')
   model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
   return model



def load_data(PATH):#load testing datasets
   a = np.load(PATH + 'x_test.npy')
   b = np.load(PATH + 'y_test.npy')
   return a,b

def test(PATH):#test model
   X_test, Y_te = load_data(PATH)#load test
   Y_test = []
   Y_test = np.asarray([np.identity(2)[x] for x in Y_te])
   Y_test = np.asarray(Y_test)
   
   current_model = build_model() #retrieve model
   scores = current_model.evaluate(X_test, Y_test, verbose=1)
   print("%s: %.2f%%" % (current_model.metrics_names[1], scores[1]*100))


test(PATH)
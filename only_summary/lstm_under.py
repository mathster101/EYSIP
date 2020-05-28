# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:53:37 2020

@author: Mathew
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:48:43 2020

@author: Mathew
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from config import *

def build_model():# builds the model and returns it
   model = Sequential() #defining the model
   model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=70))#embed layer
   model.add(SpatialDropout1D(0.2))#dropout
   model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))#bidir lstm
   model.add(Dense(2, activation='softmax'))#feed forward
   try:
      model.load_weights(PATH + "weights_bidir_undersample.best.hdf5")#check if pre trained exists
   except:
      print('pre trained weights not found!')
      
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   #print(model.summary())
   return model

def load_data(PATH):
   a = np.load(PATH + 'x_train.npy') #load train Xs
   b = np.load(PATH + 'y_train.npy') #load train Ys
   c = np.load(PATH + 'x_test.npy') #load test Xs
   d = np.load(PATH + 'y_test.npy') #load test Ys
   return a,b,c,d



def train_it(PATH,WEIGHT_PATH): #train model 
   X_train, Y_tr, X_test, Y_te = load_data(PATH) #load data
   
   Y_train = np.asarray([np.identity(2)[x] for x in Y_tr])#one-hot-ify
   
   checkpoint = ModelCheckpoint(PATH + "weights_bidir_undersample.best.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #checkpointing it
   callbacks_list = [checkpoint,EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
   
   current_model = build_model() #build model
   class_weight = {0: 0.55, 1: 0.45}
   history = current_model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=VALID_SPLIT,callbacks=callbacks_list,class_weight =class_weight )  #train time



train_it(PATH,WEIGHT_PATH)
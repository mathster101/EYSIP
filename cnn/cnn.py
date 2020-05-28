# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:09:25 2020

@author: Mathew
"""

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from config import *
import numpy as np

kernel_size = 2

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
   print(model.summary())
   return model

def load_data(PATH):
   a = np.load(PATH + 'x_train_over.npy') #load train Xs
   b = np.load(PATH + 'y_train_over.npy') #load train Ys
   c = 'na' #load test Xs
   d = 'a'
   return a,b,c,d

def train_it(PATH,WEIGHT_PATH): #train model 
   X_train, Y_tr, X_test, Y_te = load_data(PATH) #load data
   
   Y_train = np.asarray([np.identity(2)[x] for x in Y_tr])#one-hot-ify
   
   checkpoint = ModelCheckpoint(PATH + WEIGHT_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #checkpointing it
   callbacks_list = [checkpoint,EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
   
   current_model = build_model() #build model
   class_weight = {0: 1, 1: 1}
   history = current_model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=VALID_SPLIT,callbacks=callbacks_list,class_weight = class_weight )  #train time

train_it(PATH,WEIGHT_PATH)
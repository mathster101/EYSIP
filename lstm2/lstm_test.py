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
   model.load_weights(PATH + "weights.best.hdf5")
   print('weights loaded')
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

def load_data(PATH):#load testing datasets
   a = np.load(PATH + 'x_test.npy')
   b = np.load(PATH + 'y_test.npy')
   return a,b


X_test, Y_te = load_data(PATH)#load test



Y_test = []
for s in Y_te: #one hot encoding labels
   Y_test.append(one_hot(s))   
Y_test = np.asarray(Y_test)   #make it numpy arr

current_model = build_model() #retrieve model
scores = current_model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (current_model.metrics_names[1], scores[1]*100))

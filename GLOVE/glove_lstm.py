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
embedding_matrix = np.load(PATH + 'glove_mat.npy')
def build_model(embedding_matrix):# builds the model and returns it
   model = Sequential() #defining the model
   model.add(Embedding(len(embedding_matrix),EMBEDDING_DIM,weights=[embedding_matrix],input_length=50,trainable=False))#embed layer
   model.add(SpatialDropout1D(0.2))#dropout
   model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))#bidir lstm
   model.add(Dense(2, activation='softmax'))#feed forward
   try:
      model.load_weights(PATH + WEIGHT_PATH)#check if pre trained exists
   except:
      print('pre trained weights not found!')
      
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   #print(model.summary())
   return model

def load_data(PATH):
   a = np.load(PATH + 'x_train_over.npy') #load train Xs
   b = np.load(PATH + 'y_train_over.npy') #load train Ys
   return a,b



def train_it(PATH,WEIGHT_PATH): #train model 
   X_train, Y_tr = load_data(PATH) #load data
   
   Y_train = np.asarray([np.identity(2)[x] for x in Y_tr])#one-hot-ify
   
   checkpoint = ModelCheckpoint(PATH + WEIGHT_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #checkpointing it
   callbacks_list = [checkpoint,EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
   
   current_model = build_model(embedding_matrix) #build model
   history = current_model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=VALID_SPLIT,callbacks=callbacks_list)  #train time



train_it(PATH,WEIGHT_PATH)
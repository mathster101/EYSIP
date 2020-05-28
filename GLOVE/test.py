import numpy as np
from config import *
import pickle
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding

print('loading numpy files')
x = np.load(PATH + 'x_train.npy')[:10]
y = np.load(PATH + 'y_train.npy')[:10]
print('loading tokenizer')
with open(PATH + 'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

embeddings_index = {}
print('loading GloVe vectors')
f = open(GPATH + 'glove.6B.100d.txt',encoding="utf8")

for line in f:
   values = line.split()
   word = values[0]
   coefs = np.asarray(values[1:], dtype='float32')
   embeddings_index[word] = coefs   
f.close()
del x
del y
del values
del word
word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=70,trainable=False)
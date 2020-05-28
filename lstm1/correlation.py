import numpy as np
import tqdm
from config import *

def load_data(PATH):
   a = np.load(PATH + 'x_train.npy') #load train Xs
   b = np.load(PATH + 'y_train.npy') #load train Ys
   return a,b

def get_tok_sums(X):
   x_tokenized = []
   for i in tqdm.tqdm(X):
      x_tokenized.append(sum(i))
   return np.asarray(x_tokenized)

X,Y = load_data(PATH)

x = get_tok_sums(X)
r = np.corrcoef(x, Y)

print(r)

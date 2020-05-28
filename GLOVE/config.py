# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:02:34 2020

@author: Mathew
"""

MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
GPATH = '../../Datasets/gloves/'
#PATH = '../../Datasets/lstm3/'
PATH = '../../Datasets/only_summary/'
VALID_SPLIT = 0.1
EPOCHS = 10
BATCH_SIZE = 1024
WEIGHT_PATH="weights_bidir_glove.best.hdf5"

VAL_PER_TOPIC = 1500000 # a million



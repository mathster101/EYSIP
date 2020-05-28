# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:38:49 2020

@author: Mathew
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:43:55 2020

@author: Mathew
"""

import numpy as np
from config import *
from random import shuffle

COUNT = 1000000
class_count = {1: 0,0: 0}

x = np.load(PATH + 'x_train.npy')
y = np.load(PATH + 'y_train.npy')
xfin = []
yfin = []


flag = 0
while flag == 0:
   for i in range(len(x)):
      if class_count[y[i]] < COUNT:
         xfin.append(x[i])
         yfin.append(y[i])
         class_count[y[i]] += 1
   if class_count[1] == class_count[0]:
      flag = 1
   else:
      print('going back for more')
c = list(zip(xfin,yfin))
shuffle(c)
xfin,yfin = zip(*c)
xfin = np.asarray(xfin)
yfin = np.asarray(yfin)


np.save(PATH+'y_train_over',yfin)
np.save(PATH+'x_train_over',xfin)
print('files written')


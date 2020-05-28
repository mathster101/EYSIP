# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:33:34 2020

@author: Mathew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris = sns.load_dataset('iris')
#sns.pairplot(data=iris, hue='species', palette='Set2')

x=iris.iloc[:,:4]
y=iris.iloc[:,4]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
model=SVC()
model.fit(x_train,y_train)
pred = model.predict(x_test)
confusion = confusion_matrix(y_test,pred)

# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:31:57 2020

@author: Mathew
"""
#Microcontrollers/Embedded_Systems,Web Design,IOT,Arduino/ESPseries,Image Processing
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from random import randint

def get_data(PATH=""):
   f = open(PATH+'resolved.txt','r',encoding="utf-8")
   props = f.readlines()
   f.close()
   f = open('skill_dict.txt')
   skills = f.readlines()
   f.close()
   skills = [x[:-1] for x in skills]
   return props,skills


def skill_search(texts,skills):
   skillset = []
   for p in texts:#pick up a proposal at a time
      skill_small =[]
      prop = p.lower()
      prop = prop.translate(str.maketrans('', '', string.punctuation))#remove punctuation
      for skill in skills:
         flag = 0 
         key = skill.split(':')[0] # extract true skill name
         vals = skill.split(':')[1].split(',') # extract rules
         for v in vals:
            flag += prop.split(' ').count(v)
#            if v in prop.split(' '):
#               flag+=1
         if flag>=1:
            skill_small.append(key)
         
      skillset.append(skill_small)
   return skillset  


def make_out_vecs(skill_for_prop, chosen_topics):
   y = []
   for i in skill_for_prop:
      vect = []
      for topic in chosen_topics:
         if topic in i:
            vect.append(1)
         else:
            vect.append(0)
      y.append(vect)
   y = np.asarray(y)
   return y

chosen_topics = ['Microcontrollers/Embedded_Systems','Image Processing',
                 'Web Design','IOT','Robotics','Arduino/ESPseries']
props,skill_dict = get_data()
props 
skill_for_prop = skill_search(props,skill_dict)


Y = make_out_vecs(skill_for_prop, chosen_topics)
vect = TfidfVectorizer(stop_words='english')
X = vect.fit_transform(props)
Xtr,Xte,Ytr,Yte =train_test_split(X,Y,random_state=42, test_size=0.1 ,shuffle=True)

classifier = BinaryRelevance(GaussianNB())
#classifier = ClassifierChain(LogisticRegression(solver='lbfgs',multi_class='auto'))
#classifier = LabelPowerset(LogisticRegression(solver='lbfgs',multi_class='auto'))


classifier.fit(Xtr, Ytr)
predictions = np.asarray(classifier.predict(Xte).toarray())
print("Accuracy = ",accuracy_score(Yte,predictions))
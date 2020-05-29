# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:34:50 2020

@author: Mathew
"""
PATH_MENTOR_SKILLS = 'D:\Documents\Datasets\proposals\mentors.csv'

def read_skills(PATH):
   f = open(PATH)
   l = f.readlines()
   l = [x.rstrip('\n') for x in l]
   arranged = []
   temp =[]
   for i in l:
      if len(i)>50:
         s = i.split(',')[1:]
         
         s = s[:23] + s[-1:]
         temp.append(s)
   k = temp[0]
   keys = []
   for i in k:
      if len(i.split("["))>1:
         keys.append(i.split("[")[1][:-1])
      else:
         keys.append(i)
   data = temp[1:]
   t = []
   for d in data:
      td ={}
      for i in range(len(d)):
         td[keys[i]] = d[i] 
      t.append(td)
   return t

def eliminate_skills(skills):
   keys = list(skills[0].keys())
   for i in range(len(skills)):
      for j in range(len(keys)):
         if skills[i][keys[j]] == 'novice':
            skills[i].pop(keys[j])
   return skills
   

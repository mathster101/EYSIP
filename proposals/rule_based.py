# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:00:56 2020

@author: Mathew
"""
from gensim.summarization import keywords
import string
import re
import tqdm

PATH = 'D:\Documents\Datasets\proposals\proposals.csv'
PATH = 'resolved.txt'

def get_data(PATH):
   f = open(PATH,'r',encoding="utf-8")
   props = f.readlines()
   f.close()
   f = open('skill_dict.txt')
   skills = f.readlines()
   f.close()
   skills = [x[:-1] for x in skills]
   
   return props,skills
     
def skill_search(proposals,skills):
   skillset = []
   for p in tqdm.tqdm(proposals):#pick up a proposal at a time
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
            skill_small.append([key,flag])
         
      skillset.append(skill_small)
   return skillset            
   

proposals,skills = get_data(PATH)
s = skill_search(proposals,skills)

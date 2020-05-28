# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:03:48 2020

@author: Mathew
"""
import spacy
import neuralcoref
import tqdm
PATH = 'D:\Documents\Datasets\proposals\proposals.csv'
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp, conv_dict={'UAV': ['drone', 'robot']})

def get_data(PATH):
   f = open(PATH,'r',encoding="utf-8")
   props = f.readlines()
   f.close()
   f = open('skills.txt')
   skills = f.readlines()
   f.close()
   skills = [x[:-1] for x in skills]
   return props,skills

props,skills = get_data(PATH)

props_res = []

for i in tqdm.tqdm(props):
   doc = nlp(i)
   props_res.append(doc._.coref_resolved)
   
#%%
f = open('resolved.txt','w',encoding="utf-8")
for i in props_res:
   f.write(i)
f.close()
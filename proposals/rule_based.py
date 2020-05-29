# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:00:56 2020

@author: Mathew
"""
from gensim.summarization import keywords
import string
import re
import tqdm
from mentorskills import *
PATH = 'D:\Documents\Datasets\proposals\proposals.csv'
#PATH = 'resolved.txt'



def get_data(PATH):
   f = open(PATH,'r',encoding="utf-8")
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

def convert2std(mentor_data_cleaned,skills):
   temp = []
   for i in range(len(mentor_data_cleaned)):
      current = mentor_data_cleaned[i]
      name = current['Full Name']
      skillset = list(current.keys())[2:]
      skill_text = ""
      for s in skillset:
         skill_text += s+' '
#      print("___________________________________")
#      print(skillset)
#      print(skill_search([skill_text.lower()],skills))   
      temp.append([name,skill_search([skill_text.lower()],skills)[0]])
   return temp   
def match_proposals(converted,proposal_skills,matches):
   return_list = []
   for i in converted:#pick a mentor first
      n = i[0]
      #print(n)
      s = set(i[1]) # mentor's skills
      matched_val = []
      for j in range(len(proposal_skills)):
         ps = proposal_skills[j] #skills of a selected proposal
         ps = set(ps)
         common = list(s & ps)
         if len(common)>=0:
            matched_val.append(len(common))#no. of similarities
            #print(j,common,len(common))
      res = sorted(range(len(matched_val)), key = lambda sub: matched_val[sub])[-matches:]
      return_list.append([n,res])
   return return_list
         
def wrapper(PATH,PATH_MENTOR_SKILLS,results):       
   proposals,skills = get_data(PATH)
   proposal_skills = skill_search(proposals,skills)
   mentor_data = read_skills(PATH_MENTOR_SKILLS)
   mentor_data_cleaned = eliminate_skills(mentor_data)
   temp = convert2std(mentor_data_cleaned,skills)
   final = match_proposals(temp,proposal_skills,results)
   return final,proposals

assigned,proposals= wrapper(PATH,PATH_MENTOR_SKILLS,3)
   


#proposals,skills = get_data(PATH)
#proposal_skills = skill_search(proposals,skills)
#mentor_data = read_skills(PATH_MENTOR_SKILLS)
#mentor_data_cleaned = eliminate_skills(mentor_data)
#temp = convert2std(mentor_data_cleaned,skills)
#final = match_proposals(temp,proposal_skills,3)
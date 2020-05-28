# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:26:15 2020

@author: Mathew
"""
fname =''
data = {}
f = open('fname','r')
lines = f.readlines()
f.close()


for line in lines:
   cs = line.split(',')
   data[cs[0]] = cs[1:]
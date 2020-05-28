# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:07:05 2020

@author: Mathew
"""

from imports import *
pd.set_option('display.max_colwidth', 200)

candidate_sentences = pd.read_csv("D:\Downloads\sen.csv")
get_entities("the film had 200 patents")
f = open('D:\Documents\Datasets\lstm2\data.csv','r')
docs = f.readlines()

entity_pairs = []
k =candidate_sentences["sentence"][:100]
for i in tqdm(k):
  entity_pairs.append(get_entities(i))

relations = [get_relation(i) for i in tqdm(k)]
#%%
# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]
triplets = []
for i in range(len(relations)):
   temp = [source[i],relations[i],target[i]]
   triplets.append(temp)


#kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
##%%
## create a directed-graph from a dataframe
#G=nx.from_pandas_edgelist(kg_df, "source", "target", 
#                          edge_attr=True, create_using=nx.MultiDiGraph())
#
#plt.figure(figsize=(12,12))
#pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
#nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
#plt.show()
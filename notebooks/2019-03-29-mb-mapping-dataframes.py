#!/usr/bin/env python
# coding: utf-8

# ## 2019-03-28 Making Dataframes
# Today has been spent turning the large .sd file into a
# series of more manageable dataframes. 
# The script /src/data/make_panda_dataframes.py
# will do so and take about 20 minutes on my laptop.

# In[2]:


import numpy as np
import sklearn
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, IFrame
import gzip
import os
import pickle
import pandas as pd


# In[3]:


with open("../data/processed/curated_set_with_publication_year.pd.pkl", "rb") as infile:
    df = pickle.load(infile)


# In[4]:


for index, item in enumerate(df.dtypes):
    print(index, item)


# Extract the numerical columns from the table, and put them into a numpy array so we can try to map them.
# 

# In[5]:


numerical_cols = [df.columns[pos] for pos, item in enumerate(df.dtypes) if item in [np.float64, np.int64]]
print(numerical_cols)
data = df[numerical_cols].values
print(data.shape)


# In[ ]:





# In[11]:


import kmapper as km
from kmapper import jupyter
from sklearn import cluster
mapper = km.KeplerMapper(verbose=1)
projected_data = mapper.project(data, projection="sum")
graph = mapper.map(projected_data,
                   data,
                   cover=km.Cover(n_cubes=5, perc_overlap=0.75),
                   clusterer=cluster.AgglomerativeClustering(n_clusters=100,
                                                             affinity="cosine"))
# Visualize it
mapper.visualize(graph, path_html="map-dataframe-test.html",
                 title="Map Dataframe Test")
IFrame("map-dataframe-test.html", 800, 600)


# In[ ]:





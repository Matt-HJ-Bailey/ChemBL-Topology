#!/usr/bin/env python
# coding: utf-8

# ## 2019-03-28 Making Dataframes
# Today has been spent turning the large .sd file into a
# series of more manageable dataframes. 
# The script /src/data/make_panda_dataframes.py
# will do so and take about 20 minutes on my laptop.

# In[3]:


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


# In[4]:


def draw_molecule(molec, molsize):
    rdDepictor.Compute2DCoords(molec)
    drawer = rdMolDraw2D.MolDraw2DSVG(molsize[0], molsize[1])
    drawer.DrawMolecule(molec)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace("svg:", "")))


# The data frames are now stored in /data/processed/*.pd.pkl as pickled
# pandas datasets. Here we load one in to see what it looks like.

# In[17]:


with open("../data/processed/curated_set_with_publication_year.pd.pkl", "rb") as infile:
    df = pickle.load(infile)
pd.options.display.max_columns = None
display(df)


# In[18]:


df.dtypes


# Now how do we get useful information out of this? We have to select the numerical rows from the table.
# 

# In[ ]:


numerical_cols = ["BIOACT_PCHEMBL_VALUE", "CMP_ACD_LOGD", "CMP_ACD_LOGP", "CMP_ALOGP",
                  "CMP_AROMATIC_RINGS", "CMP_FULL_MWT", "CMP_HBA", "CMP_HBD", "CMP_LOGP",
                  "CMP_MOLECULAR_SPECIES_ACID", "CMP_MOLECULAR_SPECIES_BASE", "CMP_MOLECULAR_SPECIES_BASE",
                  "CMP_MOLECULAR_SPECIES_NEUTRAL", "CMP_MOLECULAR_SPECIES_ZWITTERION", "CMP_PSA"]


# In[23]:


df["CMP_STRUCTURE_TYPE"] == "MOL"


# In[ ]:





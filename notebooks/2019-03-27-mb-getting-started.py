#!/usr/bin/env python
# coding: utf-8

# Here is the usual list of tedious imports

# In[1]:


import numpy as np
import sklearn
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, IFrame
import gzip
import os
import pickle


# In[2]:


print(os.listdir())


# In[3]:


curated_set_scaffolds = []
for file in os.listdir("../data/processed/"):
    if file.startswith("curated_set_scaffolds"):
        with open(os.path.join("../data/processed/", file), "rb") as infile:
            curated_set_scaffolds.extend(pickle.load(infile))
            print(len(curated_set_scaffolds))
        
print(len(curated_set_scaffolds))


# In[4]:


def draw_molecule(molec, molsize):
    rdDepictor.Compute2DCoords(molec)
    drawer = rdMolDraw2D.MolDraw2DSVG(molsize[0], molsize[1])
    drawer.DrawMolecule(molec)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace("svg:", "")))


# In[5]:


draw_molecule(curated_set_scaffolds[10000], (450, 100))


# In[6]:


import kmapper as km
from kmapper import jupyter
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)


# In[7]:


mapper = km.KeplerMapper(verbose=1)
projected_data = mapper.fit_transform(data, projection=[0,1]) # X-Y axis
graph = mapper.map(projected_data, data, cover=km.Cover(n_cubes=[15,3], perc_overlap=[0.3, 0.3]))
# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
IFrame("make_circles_keplermapper_output.html", 800, 600)


# In[5]:


curated_set_scaffolds[10000].GetPropsAsDict()


# In[ ]:





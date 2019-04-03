#!/usr/bin/env python
# coding: utf-8

# # Today's Aim
# In the Mapper visualisation, make small images of the molecules show up instead of their index in the dataset.
# This allows everything to look much neater and practical chemists can see what's going on intuitively.
# 
# I also intend to try out some different clustering algorithms.
# ## Other Aims:
# 1. ~~Use PCA on the fingerprints to identify a few important sections, and use that as the lens.~~
#     1. We could use nonlinear Multidimensional Scaling on the Tanimoto similarity, which avoids the pitfalls of PCA.
# 2. Use a different colouring function (presence of functional group, max(activity), min(activity), stdev(activity))
# 3. For each drug, generate a bitvector of "which target has this been tested on". Then cluster in "drug-target" space using that bitvectorand see what we spot.
# 4. For a specific and well-tested target, generate a classifier (e.g. random forest) to predict how effective a drug is against it. Then, use the Fibres of Failure method (see [L. Carlsson, G. Carlsson and M. Vejdemo-Johansson, Pre-print.](https://arxiv.org/abs/1803.00384)) to predict when it goes wrong.
# 
# ## Dead Ends:
# 1. Highlighting molecules by what they share with the links.
# 2. Using PCA on the fingerprints - this over-weights the absence of features (it is $ \propto \text{XNOR}(A, B) $), such that small molecules show up very similar (see [E. Martin and E. Cao, J. Comput. Aided. Mol. Des., 2015, 29, 387–395.](http://doi.org/10.1007/s10822-014-9819-y))
# 
# ## Pitfalls:
# 1. Watch out for the lens just discretising the dataset. This will often show up as a ladder in 1D, but make sure to plot it in 2D
# 2. Compute2DCoords will often mangle the data when outputting. Workaround is to just plot them from the .sd file and load them in as you see fit.
# 

# In[17]:


import numpy as np
import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import rdkit.Chem.Fingerprints.ClusterMols
from IPython.display import SVG, IFrame
import gzip
import os
import pickle
import pandas as pd
import kmapper as km
from kmapper import jupyter
from sklearn import cluster
import scipy
import hdbscan # Provides a better clustering algorithm.


# In[9]:


with open("../data/processed/curated_set_with_publication_year.pd.pkl", "rb") as infile:
    df = pickle.load(infile)


# In[10]:


from collections import Counter
possible_targets = Counter([item for item in df["TGT_CHEMBL_ID"]])
print(len(possible_targets))
print(len(df))
print(possible_targets)
first_target = df["TGT_CHEMBL_ID"] == "CHEMBL240"
sub_df = df[first_target]


# In[11]:


fingerprint_data = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),3) for smiles in sub_df["SMILES"]]
try:
    sub_df.insert(0, "FINGERPRINT",fingerprint_data)
except ValueError:
    # If we re-run this cell, we can't reinsert the data (so instead we just replace it)
    sub_df.loc["FINGERPRINT"] = fingerprint_data


# In[23]:


sub_df


# In[12]:


fingerprint_data = []
for index, series in sub_df.iterrows():
    fingerprint_data.append(series["FINGERPRINT"])


# In[19]:


sq_distance_matrix = np.zeros([len(fingerprint_data), len(fingerprint_data)])
for row in range(len(fingerprint_data)):
    fingerprint = fingerprint_data[row]
    for col in range(row):
        other = fingerprint_data[col]
        dissimiliarity = 1.0 - rdkit.DataStructs.TanimotoSimilarity(fingerprint, other)
        sq_distance_matrix[row, col] = dissimiliarity
        sq_distance_matrix[col, row] = dissimiliarity
# Bafflingly. doing it the SciPy way is about twice as slow (57s vs 28s)
# scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(fingerprint_data, metric="jaccard"))


# In[27]:


numerical_cols = [sub_df.columns[pos] for pos, item in enumerate(sub_df.dtypes) if item in [np.float64, np.int64]]
new_data = sub_df[numerical_cols].to_numpy()
dimensional_data = np.array([row[0] for row in new_data])
print(dimensional_data)
mapper = km.KeplerMapper(verbose=1)
graph = mapper.map(dimensional_data, X=sq_distance_matrix, precomputed=True, cover=km.Cover(n_cubes=15, perc_overlap=0.20), clusterer=hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5, min_samples=2))


# In[55]:


for index, series in sub_df.iterrows():
    molec = Chem.MolFromSmiles(series["SMILES"])
    chembl_id = series["CMP_CHEMBL_ID"]
    rdDepictor.Compute2DCoords(molec)
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
    drawer.DrawMolecule(molec)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open(f"./Figures/{chembl_id}.svg", "w") as svgfile:
        svgfile.write(svg)


# In[25]:


# Visualize it
custom_tooltips=np.array([f"<img src='./Figures/{chembl_id}.svg'>" for chembl_id in sub_df["CMP_CHEMBL_ID"]])
print(custom_tooltips)
mapper.visualize(graph, path_html="2019-04-03-mb-improved-visuals-map.html",
                 title="Map Dataframe Test", color_function=dimensional_data, custom_tooltips=custom_tooltips)
IFrame("2019-04-03-mb-improved-visuals-map.html", 800, 600)


# How do we actually extract meaningful data from this list? Time to visualise it!

# In[27]:


def draw_molecule(molec, molsize, highlight_atoms=None):
    rdDepictor.Compute2DCoords(molec)
    drawer = rdMolDraw2D.MolDraw2DSVG(molsize[0], molsize[1], highlight_atoms=highlight_atoms)
    drawer.DrawMolecule(molec)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace("svg:", "")))


# ## Different lenses
# Here I am going to try to do a multidimensional scaling analysis of the distance data to utal the most important dimesions. This requires a bit of thinking on what the MDS actually outputs, and how it relates to the physical features of chemical space.
# 
# We must use a nonlinear MDS because $ 1 - T_C $ is not necessarily positive semidefinite.

# In[22]:


from sklearn.manifold import MDS
print(sq_distance_matrix.shape)
transformed_data = MDS(n_components=2, dissimilarity="precomputed", metric=False).fit_transform(sq_distance_matrix)
transformed_data


# In[25]:


plt.scatter(transformed_data[:, 0], transformed_data[:, 1])


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.imshow(sq_distance_matrix, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()


# In[38]:


mapper = km.KeplerMapper(verbose=1)
custom_tooltips=np.array([f"<img src='./Figures/{chembl_id}.svg'>" for chembl_id in sub_df["CMP_CHEMBL_ID"]])
graph = mapper.map(transformed_data,
                   X=sq_distance_matrix,
                   precomputed=True,
                   cover=km.Cover(n_cubes=10, perc_overlap=0.25),
                   clusterer=hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=10, min_samples=3))
mapper.visualize(graph, path_html="2019-04-03-mb-improved-visuals-map-mds.html",
                 title="Lensed by MDS", color_function=dimensional_data, custom_tooltips=custom_tooltips)
IFrame("2019-04-03-mb-improved-visuals-map-mds.html", 800, 600)


# In[ ]:





# In[ ]:





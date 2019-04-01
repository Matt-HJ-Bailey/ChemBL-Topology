#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, IFrame
import gzip
import os
import pickle
import pandas as pd
import kmapper as km
from kmapper import jupyter
from sklearn import cluster


# In[60]:


with open("../data/processed/curated_set_with_publication_year.pd.pkl", "rb") as infile:
    df = pickle.load(infile)


# In[61]:


first_target = df["TGT_CHEMBL_ID"] == "CHEMBL209"
sub_df = df[first_target]


# In[62]:


fingerprint_data = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2) for smiles in sub_df["SMILES"]]
try:
    sub_df.insert(0, "FINGERPRINT",fingerprint_data)
except ValueError:
    sub_df.loc["FINGERPRINT"] = fingerprint_data


# In[63]:


sub_df


# In[76]:


fingerprint_data = []
for index, series in sub_df.iterrows():
    fingerprint_data.append((series["CMP_CHEMBL_ID"], series["FINGERPRINT"]))
len(fingerprint_data)


# In[70]:


def GetDistanceMatrix(data,metric,isSimilarity=1):
    """
    Adapted from rdkit, because their implementation has a bug
    in it (it relies on Python 2 doing integer division by default).
    It is also poorly documented. Metric is a function
    that returns the 'distance' between points 1 and 2.
    
    Data should be a list of tuples with fingerprints in position 1
    (the rest of the elements of the tuple are not important)

    Returns the symmetric distance matrix.
    (see ML.Cluster.Resemblance for layout documentation)
    """
    nPts = len(data)
    num_pairs = int(nPts*(nPts-1)/2)
    res = np.zeros(num_pairs ,np.float)
    print(res)
    nSoFar=0
    for col in range(1,nPts):
        for row in range(col):
            fp1 = data[col][1]
            fp2 = data[row][1]
            if fp1.GetNumBits()>fp2.GetNumBits():
                fp1 = DataStructs.FoldFingerprint(fp1,fp1.GetNumBits()/fp2.GetNumBits())
            elif fp2.GetNumBits()>fp1.GetNumBits():
                fp2 = DataStructs.FoldFingerprint(fp2,fp2.GetNumBits()/fp1.GetNumBits())
            sim = metric(fp1,fp2)
            if isSimilarity:
                sim = 1.-sim
            res[nSoFar] = sim
            nSoFar += 1
    return res   


# In[89]:


import rdkit.Chem.Fingerprints.ClusterMols
distance_matrix = GetDistanceMatrix(fingerprint_data, metric=rdkit.DataStructs.DiceSimilarity)
distance_matrix


# In[94]:


# Now we need to mangle this flat distance matrix into a sane square one.
# The indices of (row, col) are at (col*(col-1))/2 + row
# in the flat matrix.

sq_distance_matrix = np.empty([len(fingerprint_data), len(fingerprint_data)])
for row in range(len(fingerprint_data)):
    for col in range(row + 1):
        index = int((col * (col - 1)) / 2) + row
        if row == col:
            sq_distance_matrix[row, col] = 0.0
        else:
            sq_distance_matrix[row, col] = distance_matrix[index]
            sq_distance_matrix[col, row] = distance_matrix[index]
pd.DataFrame(sq_distance_matrix)


# In[125]:


numerical_cols = [sub_df.columns[pos] for pos, item in enumerate(sub_df.dtypes) if item in [np.float64, np.int64]]
new_data = sub_df[numerical_cols].to_numpy()
data_shape = [80]
data_shape
dimensional_data = np.empty(data_shape)
for i, row in enumerate(new_data):
    dimensional_data[i] = row[0]
print(dimensional_data)
mapper = km.KeplerMapper(verbose=1)
graph = mapper.map(dimensional_data, X=sq_distance_matrix, precomputed=True, nr_cubes=4, clusterer=sklearn.cluster.DBSCAN(algorithm='auto', eps=0.5, leaf_size=30, metric='precomputed', metric_params=None, min_samples=3, n_jobs=None, p=None))


# In[126]:


# Visualize it
mapper.visualize(graph, path_html="map-dataframe-test.html",
                 title="Map Dataframe Test")
IFrame("map-dataframe-test.html", 800, 600)


# In[ ]:





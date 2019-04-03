#!/usr/bin/env python
# coding: utf-8

# # Today's Aim
# In the Mapper visualisation, make small images of the molecules show up instead of their index in the dataset.
# This allows everything to look much neater and practical chemists can see what's going on intuitively.
# 
# ## Other Aims:
# 1. Use PCA on the fingerprints to identify a few important sections, and use that as the lens.
# 2. Use a different colouring function (presence of functional group, max(activity), min(activity), stdev(activity))
# 3. For each drug, generate a bitvector of "which target has this been tested on". Then cluster in "drug-target" space using that bitvectorand see what we spot.
# 4. For a specific and well-tested target, generate a classifier (e.g. random forest) to predict how effective a drug is against it. Then, use the Fibres of Failure method (Carlsson, L., Carlsson G., Vejdemo-Johansson M., https://arxiv.org/abs/1803.00384 ) to predict when it goes wrong.
# 
# ## Dead Ends:
# 1. Highlighting molecules by what they share with the links.
# 
# ## Pitfalls:
# 1. Watch out for the lens just discretising the dataset. This will often show up as a ladder in 1D, but make sure to plot it in 2D
# 

# In[9]:


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


# In[10]:


with open("../data/processed/curated_set_with_publication_year.pd.pkl", "rb") as infile:
    df = pickle.load(infile)


# In[11]:


from collections import Counter
possible_targets = Counter([item for item in df["TGT_CHEMBL_ID"]])
print(len(possible_targets))
print(len(df))
print(possible_targets)
first_target = df["TGT_CHEMBL_ID"] == "CHEMBL240"
sub_df = df[first_target]


# In[12]:


fingerprint_data = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),3) for smiles in sub_df["SMILES"]]
try:
    sub_df.insert(0, "FINGERPRINT",fingerprint_data)
except ValueError:
    # If we re-run this cell, we can't reinsert the data (so instead we just replace it)
    sub_df.loc["FINGERPRINT"] = fingerprint_data


# In[13]:


sub_df


# In[14]:


fingerprint_data = []
for index, series in sub_df.iterrows():
    fingerprint_data.append((series["CMP_CHEMBL_ID"], series["FINGERPRINT"]))
len(fingerprint_data)


# In[23]:


def GetDistanceMatrix(data,metric,isSimilarity=1):
    """
    Adapted from rdkit, because their implementation has a bug
    in it (it relies on Python 2 doing integer division by default).
    It is also poorly documented. Metric is a function
    that returns the 'distance' between points 1 and 2.
    
    This is fixed in RDKit 2019.03.01
    Data should be a list of tuples with fingerprints in position 1
    (the rest of the elements of the tuple are not important)

    Returns the symmetric distance matrix.
    (see ML.Cluster.Resemblance for layout documentation)
    """
    nPts = len(data)
    num_pairs = int(nPts*(nPts-1)/2)
    res = np.zeros(num_pairs ,np.float)
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


# In[16]:


distance_matrix = GetDistanceMatrix(fingerprint_data, metric=rdkit.DataStructs.TanimotoSimilarity)
distance_matrix


# Now we need to mangle this flat distance matrix into a sane square one.
# The indices of $(\text{row}, \text{col})$ are at $\frac{(\text{col}\times(\text{col}-1))}{2} + \text{row} $
# in the flat matrix.

# In[17]:


sq_distance_matrix = np.empty([len(fingerprint_data), len(fingerprint_data)])
for row in range(len(fingerprint_data)):
    for col in range(row + 1):
        index = int((col * (col - 1)) / 2) + row
        if row == col:
            sq_distance_matrix[row, col] = 0.0
        else:
            sq_distance_matrix[row, col] = distance_matrix[index]
            sq_distance_matrix[col, row] = distance_matrix[index]


# In[18]:


numerical_cols = [sub_df.columns[pos] for pos, item in enumerate(sub_df.dtypes) if item in [np.float64, np.int64]]
new_data = sub_df[numerical_cols].to_numpy()
dimensional_data = np.array([row[0] for row in new_data])
print(dimensional_data)
mapper = km.KeplerMapper(verbose=1)
graph = mapper.map(dimensional_data, X=sq_distance_matrix, precomputed=True, cover=km.Cover(n_cubes=35, perc_overlap=0.2), clusterer=sklearn.cluster.DBSCAN(algorithm='auto', eps=0.40, leaf_size=30, metric='precomputed', min_samples=3, n_jobs=4))


# In[19]:


# Visualize it
mapper.visualize(graph, path_html="map-dataframe-test.html",
                 title="Map Dataframe Test", color_function=dimensional_data)
IFrame("map-dataframe-test.html", 800, 600)


# How do we actually extract meaningful data from this list? Time to visualise it!

# In[20]:


mols = [Chem.MolFromSmiles(sub_df.iloc[i]["SMILES"]) for i in graph["nodes"]["cube2_cluster0"]]
from rdkit.Chem import rdFMCS
res =rdFMCS.FindMCS(mols)
newmol = Chem.MolFromSmarts(res.smartsString)


# In[21]:


def draw_molecule(molec, molsize, highlight_atoms=None):
    rdDepictor.Compute2DCoords(molec)
    drawer = rdMolDraw2D.MolDraw2DSVG(molsize[0], molsize[1], highlight_atoms=highlight_atoms)
    drawer.DrawMolecule(molec)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace("svg:", "")))


# In[22]:


for index, node in enumerate(graph["nodes"]):
    mols = [Chem.MolFromSmiles(sub_df.iloc[i]["SMILES"]) for i in graph["nodes"][node]]
    mean_bioactivity = np.mean([sub_df.iloc[i]["BIOACT_PCHEMBL_VALUE"] for i in graph["nodes"][node]])
    if len(mols) > 1:
        max_substructure = rdFMCS.FindMCS(mols, ringMatchesRingOnly=True).smartsString
        mol_smarts = Chem.MolFromSmarts(max_substructure)
        highlight_list = [mol.GetSubstructMatches(mol_smarts)[0] for mol in mols]
        print(node, mean_bioactivity)
        display(SVG(Chem.Draw._MolsToGridSVG(mols, highlightAtomLists=highlight_list)))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Today I have been looking at some basic clustering of molecules.
# 
# ## Todo:
# 1. See if we can cluster by things other than fingerprints
# 2. Make the pandas data handling a bit nicer
# 3. Consider what the topology is actually meaning here
# 
# 

# In[3]:


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


# In[4]:


with open("../data/processed/curated_set_with_publication_year.pd.pkl", "rb") as infile:
    df = pickle.load(infile)


# In[8]:


from collections import Counter
possible_targets = Counter([item for item in df["TGT_CHEMBL_ID"]])
print(len(possible_targets))
print(len(df))
print(possible_targets)
first_target = df["TGT_CHEMBL_ID"] == "CHEMBL4336"
sub_df = df[first_target]


# In[48]:


fingerprint_data = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),3) for smiles in sub_df["SMILES"]]
try:
    sub_df.insert(0, "FINGERPRINT",fingerprint_data)
except ValueError:
    sub_df.loc["FINGERPRINT"] = fingerprint_data


# In[49]:


sub_df


# In[50]:


fingerprint_data = []
for index, series in sub_df.iterrows():
    fingerprint_data.append((series["CMP_CHEMBL_ID"], series["FINGERPRINT"]))
len(fingerprint_data)


# In[51]:


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


# In[52]:


distance_matrix = GetDistanceMatrix(fingerprint_data, metric=rdkit.DataStructs.TanimotoSimilarity)
distance_matrix


# Now we need to mangle this flat distance matrix into a sane square one.
# The indices of $(\text{row}, \text{col})$ are at $\frac{(\text{col}\times(\text{col}-1))}{2} + \text{row} $
# in the flat matrix.

# In[53]:


sq_distance_matrix = np.empty([len(fingerprint_data), len(fingerprint_data)])
for row in range(len(fingerprint_data)):
    for col in range(row + 1):
        index = int((col * (col - 1)) / 2) + row
        if row == col:
            sq_distance_matrix[row, col] = 0.0
        else:
            sq_distance_matrix[row, col] = distance_matrix[index]
            sq_distance_matrix[col, row] = distance_matrix[index]


# In[73]:


numerical_cols = [sub_df.columns[pos] for pos, item in enumerate(sub_df.dtypes) if item in [np.float64, np.int64]]
new_data = sub_df[numerical_cols].to_numpy()
dimensional_data = np.array([row[0] for row in new_data])
print(dimensional_data)
mapper = km.KeplerMapper(verbose=1)
graph = mapper.map(dimensional_data, X=sq_distance_matrix, precomputed=True, cover=km.Cover(n_cubes=35, perc_overlap=0.2), clusterer=sklearn.cluster.DBSCAN(algorithm='auto', eps=0.40, leaf_size=30, metric='precomputed', min_samples=3, n_jobs=4))


# In[74]:


# Visualize it
mapper.visualize(graph, path_html="map-dataframe-test.html",
                 title="Map Dataframe Test", color_function=dimensional_data)
IFrame("map-dataframe-test.html", 800, 600)


# How do we actually extract meaningful data from this list? Time to visualise it!

# In[56]:


mols = [Chem.MolFromSmiles(sub_df.iloc[i]["SMILES"]) for i in graph["nodes"]["cube2_cluster0"]]
from rdkit.Chem import rdFMCS
res =rdFMCS.FindMCS(mols)
newmol = Chem.MolFromSmarts(res.smartsString)


# In[57]:


def draw_molecule(molec, molsize, highlight_atoms=None):
    rdDepictor.Compute2DCoords(molec)
    drawer = rdMolDraw2D.MolDraw2DSVG(molsize[0], molsize[1], highlight_atoms=highlight_atoms)
    drawer.DrawMolecule(molec)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace("svg:", "")))


# In[ ]:


for index, node in enumerate(graph["nodes"]):
    mols = [Chem.MolFromSmiles(sub_df.iloc[i]["SMILES"]) for i in graph["nodes"][node]]
    mean_bioactivity = np.mean([sub_df.iloc[i]["BIOACT_PCHEMBL_VALUE"] for i in graph["nodes"][node]])
    if len(mols) > 1:
        max_substructure = rdFMCS.FindMCS(mols, ringMatchesRingOnly=True).smartsString
        mol_smarts = Chem.MolFromSmarts(max_substructure)
        highlight_list = [mol.GetSubstructMatches(mol_smarts)[0] for mol in mols]
        print(node, mean_bioactivity)
        display(SVG(Chem.Draw._MolsToGridSVG(mols, highlightAtomLists=highlight_list)))


# In[59]:


print(graph)


# The theory goes that these clusters are linked by specific molecules. If they are linked, perhaps we should
# look at the maximum common substructure of the linking atom within each cluster.

# In[ ]:


cm = [(1,0,0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
for node in graph["nodes"]:
    for othernode in graph["nodes"]:
        if node == othernode:
            break
        
        intersection = set(graph["nodes"][node]).intersection(graph["nodes"][node])
        print(node, "and", othernode," are linked by", intersection)
        for shared_mol in intersection:
            cmap_index = 0
            shared_mol_smiles = Chem.MolFromSmiles(sub_df.iloc[shared_mol]["SMILES"])
            mols = [Chem.MolFromSmiles(sub_df.iloc[i]["SMILES"]) for i in graph["nodes"][node]]
            highlight_list = []
            total_shared = set()
            for mol in graph["nodes"][node]:
                mol_smiles = Chem.MolFromSmiles(sub_df.iloc[mol]["SMILES"])
                max_substructure = rdFMCS.FindMCS([mol_smiles, shared_mol_smiles], ringMatchesRingOnly=True).smartsString
                mol_smarts = Chem.MolFromSmarts(max_substructure)
                matching_atoms = mol_smiles.GetSubstructMatches(mol_smarts)[0]
                highlight_list.append(matching_atoms)
                if not total_shared:
                    total_shared = set(matching_atoms)
                else:
                    total_shared = total_shared.intersection(set(matching_atoms))
                cmap_index += 1
            print(total_shared)
            print(len(highlight_list), len(mols))
            display(SVG(Chem.Draw._MolsToGridSVG(mols, highlightAtomLists=highlight_list)))
            break


# In[ ]:





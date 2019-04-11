import rdkit
from rdkit import Chem
from rdkit.Chem.Fingerprints import ClusterMols
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

mols = [Chem.MolFromSmiles("C1=CC=CC=C1"), Chem.MolFromSmiles("CC1=CC=CC=C1")]
fingerprints = [GetMorganFingerprintAsBitVect(mol,3) for mol in mols]
distance_matrix = ClusterMols.GetDistanceMatrix(fingerprints, metric=rdkit.DataStructs.DiceSimilarity)

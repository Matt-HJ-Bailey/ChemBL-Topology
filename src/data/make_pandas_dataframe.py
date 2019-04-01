# -*- coding: utf-8 -*-
import click
import logging
import pickle
import gzip
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

FILES_TO_LOAD = ["curated_set_with_publication_year", "curated_set_scaffolds"]

def make_pandas_dataframe(input_filepath, input_filename):
    """
    Takes in an input file in the form of a pickled
    list of RDChem Mol objects, and parses it into
    a pandas dataframe, which it then returns.

    Also generates a SMILES string so we can
    reconstruct the molecule later without having
    to do lookups of the ChEMBL ID.
    """
    total_df = None
    logger = logging.getLogger(__name__)
    for filechunk in sorted(os.listdir(input_filepath)):
        if filechunk.startswith(input_filename) and filechunk.endswith(".sd.pkl"):
            logger.info(f"Reading from {filechunk}")
            with open(os.path.join(input_filepath, filechunk), "rb") as infile:
                data_dict = {molobj.GetProp("TC_key"): {**molobj.GetPropsAsDict(),**{"SMILES":Chem.MolToSmiles(molobj)}} for molobj in pickle.load(infile)}
                if total_df is None:
                    total_df = pd.DataFrame(data_dict)
                else:
                    df = pd.DataFrame(data_dict)
                    total_df = pd.concat([total_df, df], sort=False, axis="columns", copy=False)
        
    return total_df.T

def clean_pandas_dataframe(df):
    """
    Takes in a pandas dataframe, and cleans up
    some of the 1s and 0s (or Ys and Ns) into
    Python Trues and Falses.
    """
    df["CMP_INORGANIC_FLAG"] = df["CMP_INORGANIC_FLAG"].astype(bool)
    df["CMP_MOLECULAR_SPECIES_BASE"] = df["CMP_MOLECULAR_SPECIES_BASE"].astype(bool)
    df["CMP_MOLECULAR_SPECIES_ACID"] = df["CMP_MOLECULAR_SPECIES_ACID"].astype(bool)
    df["CMP_MOLECULAR_SPECIES_NEUTRAL"] = df["CMP_MOLECULAR_SPECIES_NEUTRAL"].astype(bool)
    df["CMP_MOLECULAR_SPECIES_ZWITTERION"] = df["CMP_MOLECULAR_SPECIES_ZWITTERION"].astype(bool)
    df["CMP_TYPE_PROTEIN"] = df["CMP_TYPE_PROTEIN"].astype(bool)
    df["CMP_TYPE_SMALL_MOLECULE"] = df["CMP_TYPE_SMALL_MOLECULE"].astype(bool)

    df["BIOACT_PCHEMBL_VALUE"] = df["BIOACT_PCHEMBL_VALUE"].astype(float)
    # This one is a special case, because a blank value
    # has snuck in somehow.
    df["CMP_ACD_LOGD"] = df["CMP_ACD_LOGD"].apply(lambda x: float(x) if x else np.nan)
    df["CMP_ACD_LOGP"] = df["CMP_ACD_LOGP"].astype(float)
    df["CMP_ALOGP"] = df["CMP_ALOGP"].astype(float)
    df["CMP_AROMATIC_RINGS"] = df["CMP_AROMATIC_RINGS"].astype(int)
    df["CMP_FULL_MWT"] = df["CMP_FULL_MWT"].astype(float)
    df["CMP_HBA"] = df["CMP_HBA"].astype(int)
    df["CMP_HBD"] = df["CMP_HBD"].astype(int)
    df["CMP_HEAVY_ATOMS"] = df["CMP_HEAVY_ATOMS"].astype(int)
    df["CMP_LOGP"] = df["CMP_LOGP"].astype(float)
    df["CMP_NUM_ALERTS"] = df["CMP_NUM_ALERTS"].astype(int)
    df["CMP_NUM_RO5_VIOLATIONS"] = df["CMP_NUM_RO5_VIOLATIONS"].astype(int)
    df["CMP_PSA"] = df["CMP_PSA"].astype(float)
    df["CMP_RTB"] = df["CMP_RTB"].astype(int)
    # Publication year is not a feature of all
    # datasets.
    try:
        df["DOC_YEAR"] = df["DOC_YEAR"].astype(int)
    except KeyError:
        pass
    # Some columns are bafflingly given as Y and N
    mask = df == "Y"
    df = df.where(~mask, other=True)
    mask = df == "N"
    df = df.where(~mask, other=False)
    return df
     

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn chunked data from (../interim/) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making pandas dataframe from split data. Warning: This will eat 3GB of RAM.')

    for file_to_load in FILES_TO_LOAD:
        df = make_pandas_dataframe(input_filepath, file_to_load)
        df = clean_pandas_dataframe(df)
        output_pickle = os.path.join(output_filepath, file_to_load + ".pd.pkl")
        with open(output_pickle, "wb") as outfile:
            pickle.dump(df, outfile, protocol=4)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

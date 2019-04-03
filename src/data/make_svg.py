# -*- coding: utf-8 -*-
import click
import logging
import pickle
import gzip
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

ENTRIES_PER_CHUNK = 75000
FILESIZE_CUTOFF = 40e6
FILES_TO_LOAD = ["curated_set_with_publication_year", "curated_set_scaffolds"]

def draw_molecule(molec, molsize, filename, highlight_atoms=None):
    # rdDepictor.Compute2DCoords(molec)
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 250)
    drawer.DrawMolecule(molec)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open(filename, "w") as svgfile:
        svgfile.write(svg)

def load_from_gzip(input_filepath, filename, target):
    """
    Loads a gzipped .sd file, and returns it
    as a not-None python list for later
    pickling.
    """
    with gzip.open(os.path.join(input_filepath, filename)) as gzinfile:
        infile = Chem.ForwardSDMolSupplier(gzinfile)
        return [x for x in infile if x.GetProp("TGT_CHEMBL_ID") == target]     

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('target', type=str)
def main(input_filepath, output_filepath):
    """
    Draws all of the compounds that interact with target
    and plots them in outputpath/Figures/.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Plotting all the compounds that interact with {target}')
    
    molecules = load_from_gzip(input_filepath, "curated_set_with_publication_year.sd.gz", target)
    for mol in molecules:
        if mol.GetProp("TGT_CHEMBL_ID") == target:
            draw_molecule(mol, (500, 500), f"{output_filepath}/Figures/{mol.GetProp('CMP_CHEMBL_ID')}.svg")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

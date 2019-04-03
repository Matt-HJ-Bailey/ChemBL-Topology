# -*- coding: utf-8 -*-
import click
import logging
import pickle
import gzip
import os
from rdkit import Chem
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

ENTRIES_PER_CHUNK = 75000
FILESIZE_CUTOFF = 40e6
FILES_TO_LOAD = ["curated_set_with_publication_year", "curated_set_scaffolds"]

def split_sd_file(input_filepath, filename, num_in_file):
    """
    Splits an sd file into sets of fixed size, specified by
    the parameter num_in_file.
    Puts them into gzipped format containing a subset
    of the values. Warning: gzipping takes some time.

    Returns the number of fragments.
    """
    logger = logging.getLogger(__name__)
    number = 0
    file_count = 0
    prefix, ext = os.path.splitext(filename)
    prefix, ext2 = os.path.splitext(prefix)
    with gzip.open(os.path.join(input_filepath, filename)) as gzinfile:
        lines = []
        for line in gzinfile:
            lines.append(line)
            if line == b'$$$$\r\n':
                number += 1
            if number == num_in_file:
                with gzip.open(os.path.join(input_filepath, prefix + "." + str(file_count) + ext2 + ext), "wb") as fi:
                    logger.info(f"Gzipping fragment {file_count}")
                    fi.writelines(lines)
                lines = []
                number = 0
                file_count += 1
    # Now write out the final leftovers
    with gzip.open(os.path.join(input_filepath, prefix + "." + str(file_count) + ext2 + ext), "wb") as fi:
        logger.info(f"Gzipping fragment {file_count}")
        fi.writelines(lines)

    return file_count

def load_from_gzip(input_filepath, filename):
    """
    Loads a gzipped .sd file, and returns it
    as a not-None python list for later
    pickling.
    """
    with gzip.open(os.path.join(input_filepath, filename)) as gzinfile:
        infile = Chem.ForwardSDMolSupplier(gzinfile)
        return [x for x in infile if x is not None]

def pickle_one_file(input_filepath, output_filepath, filename):
    """
    Takes in a two filepaths (in and out) and the filename
    of a .sd file. Proceeds by loading it into a 
    Python list using load_from_gzip, and then
    pickles it into the output_filepath folder.
    """
    logger = logging.getLogger(__name__)
    prefix, ext = os.path.splitext(filename)
    molecules = load_from_gzip(input_filepath, filename)
    logger.info(f"Loaded in {len(molecules)} from file.")
    outfilepath = os.path.join(output_filepath, prefix + ".pkl")
    logger.info(f"Pickling {filename} to {outfilepath}")
    # RDKit doesn't preserve properties when pickling,
    # and this is not well documented.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)  
    with open(outfilepath, "wb") as outfile:
        pickle.dump(molecules, outfile, protocol=4)
     

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Splits the data set into more managable pickled chunks,
    and stores in the output_filepath.
    The pickles are python lists of RDKit mol files.
    They in theory preserve everything from the original 
    .sd file, and expand to being much bigger than we started.
    (2.3GB?)

    Adjust the parameter ENTRIES_PER_CHUNK in the top of the file if your machine
    has more than 8GB of RAM. For 8GB of RAM, a value of 75,000 is sane.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data. Warning: This will eat 7GB of RAM.')
    # We need to check if we can actually read these files without
    # catching fire and falling over.
    for file_to_load in FILES_TO_LOAD:
        if os.path.getsize(os.path.join(input_filepath, file_to_load + ".sd.gz")) > FILESIZE_CUTOFF:
            logger.info(f"Uh oh - file too big: {file_to_load}. Will split it into smaller segments.")
            fragments = split_sd_file(input_filepath, file_to_load + ".sd.gz", ENTRIES_PER_CHUNK)
            for i in range(fragments + 1):
                pickle_one_file(input_filepath, output_filepath, file_to_load + "." + str(i) + ".sd.gz")
        else:
            pickle_one_file(input_filepath, output_filepath, file_to_load + ".sd.gz")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

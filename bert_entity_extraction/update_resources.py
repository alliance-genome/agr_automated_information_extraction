"""
Before we run the bert model we need to get certain data from various places first.
"""
import configparser
import csv
import os
import logging
import argparse
import pickle
import re

from tqdm import tqdm


logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read("config/config.ini")


def get_pmid_to_pmcid(path: str):
    """Creates a pmid to pmcid dictionary

    Parameters:
        path, str
            The path to the file containing the pmid to pmcid mapping.
            Typically, this file is called "PMC-ids.csv"
    """
    pmid_to_pmcid = {}
    with open(path, "r") as f:
        length = len(f.readlines())
    with open(path, "r") as f:
        rd = csv.DictReader(f)
        for row in tqdm(rd, desc=f"Reading {path}", unit=" lines", total=length):
            pmcid = row["PMCID"]
            pmid = row["PMID"]
            if pmid and pmcid:
                pmid_to_pmcid[pmid] = pmcid
    # pickle the dictionary
    # get the directory path from the config parameter
    dir_path = os.path.dirname(config.get('PICKLES', 'PMC_ids_dict'))

    # create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # open the file for writing
    with open(config.get('PICKLES', 'PMC_ids_dict'), "wb") as out:
        pickle.dump(pmid_to_pmcid, out)


def get_genes_dict(gene_synonyms_path: str, current_genes_path: str):
    """Creates a dictionary of gene symbols to their flybase id

    Parameters:
        gene_synonyms_path, str
            The path to the file containing the gene synonyms.
            Typically, this file is called "fb_synonym_fb_[DATE].tsv"
        current_genes_path, str
            The path to the file containing the current genes.
            Typically, this file is called "currentDmelHsap.txt"
    """
    # tsv column indices
    PRIMARY_FBID = 0
    CURRENT_SYMBOL = 2
    CURRENT_FULLNAME = 3
    FULLNAME_SYNONYMS = 4
    SYMBOL_SYNONYM = 5
    relevant_genes = set()
    with open(current_genes_path, "r") as current_genes_file:
        for line in current_genes_file:
            gene = line.rstrip()
            relevant_genes.add(gene)
    gene_dict = {}
    fbid_to_symbol = {}

    with open(gene_synonyms_path, "r") as gene_file:
        length = len(gene_file.readlines())
    with open(gene_synonyms_path, "r") as gene_file:
        rd = csv.reader(filter(lambda row: row[0] != '#', gene_file), delimiter="\t", quotechar='"')
        for row in tqdm(rd, desc="Making genes dictionary", total=length, unit=" lines"):
            if (len(row) == 6 and row[PRIMARY_FBID].startswith("FBgn")):
                fbid = row[PRIMARY_FBID]
                if fbid in relevant_genes:
                    fullname_synonyms = row[FULLNAME_SYNONYMS]
                    if len(fullname_synonyms) > 0:
                        for syn in re.split('|', fullname_synonyms):
                            gene_dict[syn] = fbid

                    fullname = row[CURRENT_FULLNAME]
                    if len(fullname) > 0:
                        gene_dict[fullname] = fbid  # fullname takes precedence over symbol when ambiguous.
                        # Last ambiguous fullname wins.

                    symbol_synonyms = row[SYMBOL_SYNONYM]
                    if len(symbol_synonyms) > 0:
                        for syn in symbol_synonyms.split("|"):  # here we ignore commas, which is not the same as
                            # before, just because it seems like this one is less comma separated, who knows
                            gene_dict[syn] = fbid

                    symbol = row[CURRENT_SYMBOL]
                    gene_dict[symbol] = fbid  # this should be last to have absolute precedence
                    fbid_to_symbol[fbid] = symbol
        # pickle the dictionaries
        # get the directory paths from the config parameter
        dir_path = os.path.dirname(config.get('PICKLES', 'gene_dict'))
        fbid_to_symbol_dir_path = os.path.dirname(config.get('PICKLES', 'fbid_to_symbol_dict'))

        # create the directories if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(fbid_to_symbol_dir_path):
            os.makedirs(fbid_to_symbol_dir_path)

        # output the dictionaries
        with open(config.get('PICKLES', 'gene_dict'), "wb") as out:
            pickle.dump(gene_dict, out)
        with open(config.get('PICKLES', 'fbid_to_symbol_dict'), "wb") as out:
            pickle.dump(fbid_to_symbol, out)


def parse_arguments():
    """ Get the arguments from the command line """
    helptext = """ Something """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=helptext)
    parser.add_argument("-m", "--mod_abbreviation", type=str, default="FB",
                        help="Mod abbreviation: i.e. 'FB'")

    return parser.parse_args()


def main():
    """
    Update the resources needed for the BERT model
    """
    args = parse_arguments()

    pmcid = config.get('PUBMED', 'PMC_ids')
    gene_syns = config.get(args.mod_abbreviation, 'gene_synonyms')
    current_genes = config.get('PUBMED', 'current_genes')

    if os.path.exists(pmcid) and os.path.exists(gene_syns) and os.path.exists(current_genes):
        get_pmid_to_pmcid(config.get('PUBMED', 'PMC_ids'))
        get_genes_dict(gene_syns, current_genes)
    else:
        if not os.path.exists(pmcid):
            logger.error(f"{pmcid} not found. Please add proper path in config.ini")
        if not os.path.exists(gene_syns):
            logger.error(f"{gene_syns} not found. Please add proper path in config.ini")
        if not os.path.exists(current_genes):
            logger.error(f"{current_genes} not found. Please add proper path in config.ini")


if __name__ == '__main__':
    main()

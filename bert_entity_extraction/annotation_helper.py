# Copyright 2023 Charlie Grivaz
#
# Modified: Ian Longden to get data directly form Alliance databases.
#
# This file was part of Fly Base Annotation Helper
#
# Fly Base Annotation Helper is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fly Base Annotation Helper is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Fly Base Annotation Helper. If not, see <http://www.gnu.org/licenses/>.
import time
import os
import argparse
import configparser
import sys
import csv
import logging
import subprocess
# import tqdm
import requests
import xmltodict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from retry import retry
# from gene_finding import (get_genes)
from gene_finding import deep_learning
from utils.abc_utils import load_all_jobs, get_tet_source_id, \
    set_job_started, set_job_success, send_entity_tag_to_abc, \
    set_job_failure, set_blue_api_base_url
# get_cached_mod_abbreviation_from_id, get_tet_source_id, set_job_started, set_job_success)
# from agr_literature_service.lit_processing.utils.sqlalchemy_utils import create_postgres_session

parser = argparse.ArgumentParser(description='Extract biological entities from documents using Bert model')
parser.add_argument("-l", "--log_level", type=str,
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default='INFO', help="Set the logging level")
parser.add_argument("-s", "--stage", action="store_true",
                    help="Only run for on stage.", required=False)
parser.add_argument("-f", "--reference_curie", type=str,
                    help="Only run for this reference.", required=False)
parser.add_argument("-t", "--topic", type=str,
                    help="Only run for this topic.", required=False)
parser.add_argument("-m", "--mod_abbreviation", type=str,
                    default="FB", help="Only run for FB.")
parser.add_argument("-c", "---config_file", type=str,
                    default="/usr/src/app/bert_entity_extraction/config.ini",
                    help="Config file for FlyBert")

args = parser.parse_args()
print(args)
config_parser = configparser.ConfigParser()
config_parser.read(args.config_file)

logger = logging.getLogger(__name__)

# get pmid to pmcid pickle dictionary
# with open(config_parser.get('PICKLES','PMC_ids_dict'), "rb") as f:
#    pmid_to_pmcid_dict = pickle.load(f)

EXCEPTIONS_PATH = config_parser.get('PATHS', 'exceptions')


def create_postgres_engine(db):

    """Connect to database."""
    if args.stage:
        server = os.environ.get(f'{db}_HOST', 'literature-dev.cmnnhlso7wdi.us-east-1.rds.amazonaws.com')
    else:
        server = os.environ.get(f'{db}_HOST', 'literature-prod.cmnnhlso7wdi.us-east-1.rds.amazonaws.com')

    user = os.environ.get(f'{db}_USERNAME', 'postgres')
    password = os.environ.get(f'{db}_PASSWORD')
    if not password:
        print(f"No password for env {db}_PASSWORD")
    port = os.environ.get(f'{db}_PORT', '5432')
    db = os.environ.get(f'{db}_DATABASE', 'literature')

    # Create our SQL Alchemy engine from our environmental variables.
    engine_var = 'postgresql://' + user + ":" + password + '@' + server + ':' + port + '/' + db
    # future=True is recommended for 2.0-style behavior.
    # But referencefile unit test would fail, so removed it.
    engine = create_engine(engine_var)

    return engine


def create_postgres_session(db):

    engine = create_postgres_engine(db)

    # SQLAlchemy 2.0 recommends using 'autocommit=False' explicitly in sessionmaker
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()

    return session


@retry(requests.exceptions.RequestException, delay=30, backoff=2, tries=10)
def getFtpPath(pmcid: str):
    """Returns the ftp path to a paper given its pmcid

    Parameters:
        pmcid, str
            The pmcid of the paper
    """
    pmc_api_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=" + pmcid
    try:
        response = requests.get(pmc_api_url)
        response.raise_for_status()  # Check for any request errors
        dict_data = xmltodict.parse(response.content)
        ftplink = ""
        if "error" in dict_data['OA']:
            error_code = dict_data['OA']['error']['@code']
            logging.warning(f"ERROR: {error_code} {pmcid}")
        else:
            link = dict_data['OA']['records']['record']['link']
            if isinstance(link, list):
                ftplink = link[0]['@href']
            else:
                ftplink = link['@href']
            assert (".tar.gz" in ftplink)
        time.sleep(config_parser.getint('PARAMETERS', 'sleep_time_between_requests'))
        print(f"Link is '{ftplink}'")
        return ftplink
    except (requests.exceptions.RequestException, KeyError, AssertionError) as e:
        logging.warning(f"Failed to get FTP path for {pmcid}: {str(e)}")
    return None


@retry(subprocess.CalledProcessError, delay=30, backoff=2, tries=10)
def download(ftp: str):
    """Downloads a paper given its ftp path

    Parameters:
        ftp, str
            The ftp path to the paper
    """
    wget = f"wget -nc --timeout=10 -P {config_parser.get('PATHS', 'corpus')} {ftp}"
    try:
        subprocess.run(wget, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to download {ftp}: {str(e)}")


def getXmlFromTar(pmcid: str):
    """Extracts the xml file from the tar.gz file

    Parameters:
        pmcid, str
            The pmcid of the paper
    """
    f = f"{config_parser.get('PATHS', 'corpus')}/{pmcid}.tar.gz"
    untar = f"tar -xf {f} -C {config_parser.get('PATHS', 'corpus')}"
    try:
        subprocess.run(untar, shell=True, check=True)
        # make xml directory if it doesn't exist
        if not os.path.exists(config_parser.get('PATHS', 'xml')):
            os.makedirs(config_parser.get('PATHS', 'xml'))
        copy_xml = f"cp {config_parser.get('PATHS', 'corpus')}/{pmcid}/*.nxml {config_parser.get('PATHS', 'xml')}/{pmcid}.nxml"
        subprocess.run(copy_xml, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to extract XML from tar for {pmcid}: {str(e)}")


def get_ateam_dicts() -> (dict[str, str], dict[str, str]):
    db = create_postgres_session('PERSISTENT_STORE_DB')
    sql = """SELECT DISTINCT be.primaryexternalid, sa.displaytext
        FROM biologicalentity be
        JOIN slotannotation sa ON be.id = sa.singlegene_id
        JOIN ontologyterm ot ON be.taxon_id = ot.id
        WHERE sa.slotannotationtype  in (
            'GeneSymbolSlotAnnotation',
            'GeneSystematicNameSlotAnnotation',
            'GeneFullNameSlotAnnotation',
            'GeneSynonymSlotAnnotation'
        )
        and ot.curie = 'NCBITaxon:7227'"""
    rows = db.execute(text(sql)).all()
    gene_dict = {}
    fbid_to_symbol = {}
    for row in rows:
        gene_dict[row[1]] = row[0]
        fbid_to_symbol[row[0]] = row[1]
    return gene_dict, fbid_to_symbol


def get_pmcids_for_references(jobs):
    """Look up pmc_ids for the reference in the jobs."""
    refs = [str(j['reference_id']) for j in jobs]
    ref_to_pmc = {}
    if refs:
        db_session = create_postgres_session('PSQL')
        print(f"refs is {refs[:5]}")
        # NOTE: remove [:2] from sql after testing
        sql = f"""SELECT reference_id, curie
                   FROM cross_reference
                    WHERE curie_prefix = 'PMCID'
                         AND reference_id in ({','.join(refs)})"""
        print(sql)
        rs = db_session.execute(text(sql))
        rows = rs.fetchall()
        for row in rows:
            # PMCID:PMC11238292
            print(f"sql res: {row}")
            pmcid = row[1][6:]
            ref_to_pmc[row[0]] = pmcid
    else:
        print("No jobs to process")
    return ref_to_pmc


def get_data_from_alliance_db():
    """
    Get jobs to run.
    Get pmc to reference_id for those jobs
    """
    jobs = {}
    print(f"ARGS: {args}")
    mod_topic_jobs = load_all_jobs("gene_extraction_job", args=args)
    ref_to_pmc = {}
    for (mod_id, topic), jobs in mod_topic_jobs.items():
        print(f"mod_id = {mod_id}, topic = {topic}, first job {jobs[0]}")
        ref_to_pmc = get_pmcids_for_references(jobs)
        print(f"ref_to_pmc = {ref_to_pmc}")
    return jobs, ref_to_pmc


def removeFiles(pmcid: str):
    """Removes paper files that were downloaded

    Parameters:
        pmcid, str
            The pmcid of the paper
    """
    f = f"{config_parser.get('PATHS', 'corpus')}/{pmcid}.tar.gz"
    try:
        os.remove(f)
    except FileNotFoundError:
        pass
    f = f"{config_parser.get('PATHS', 'corpus')}/{pmcid}"
    # remove extracted directory
    subprocess.run(f"rm -r {f}", shell=True, check=False)
    f = f"{config_parser.get('PATHS', 'xml')}/{pmcid}.nxml"
    try:
        os.remove(f)
    except FileNotFoundError:
        pass


def main():  # noqa C901
    if args.stage:
        set_blue_api_base_url("https://stage-literature-rest.alliancegenome.org")
        os.environ['ABC_API_SERVER'] = "https://stage-literature-rest.alliancegenome.org"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    jobs, ref_to_pmc = get_data_from_alliance_db()
    # need ref_id to job_id
    ref_to_job = {item['reference_id']: item['reference_workflow_tag_id'] for item in jobs}
    print(ref_to_job)
    print(f"Number of jobs: {len(jobs)}")

    # unpickle gene dictionary
    # gene_dict = get_gene_dict()
    gene_dict, fbid_to_symbol = get_ateam_dicts()
    print(f"gene_dict -> {len(gene_dict)}")
    key = list(gene_dict.keys())[0]
    print(f"gene_dict Example: key -> {key} {gene_dict[key]}")
    key = list(fbid_to_symbol.keys())[0]
    print(f"fbid Exmaple: key -> {key} {fbid_to_symbol[key]}")
    print(f"fbid_to_symbol -> {len(fbid_to_symbol)}")
    # with open(config_parser.get('PICKLES', 'gene_dict'), "rb") as f:
    #     gene_dict = pickle.load(f)

    # unpickle fbid to symbol dictionary
    # with open(config_parser.get('PICKLES', 'fbid_to_symbol_dict'), "rb") as f:
    #     fbid_to_symbol = pickle.load(f)

    if config_parser.getboolean('PARAMETERS', 'use_deep_learning'):
        deep_learning.initialize(config_parser.get('PATHS', 'deep_learning_model'))

    results = {}
    # FB mod_topic_jobs = load_all_jobs("_extraction_job", args=args)
    # with open(args.input.name, "r") as f:
    #    input_list = f.readlines()
    # input list gained from search of db now
    tet_source_id = get_tet_source_id(
        mod_abbreviation=args.mod_abbreviation,
        source_method="abc_entity_extractor",
        source_description="Alliance entity extraction pipeline using machine learning "
                           "to identify papers of interest for curation data types")
    species = 'NCBITaxon:7227'
    # for pmcid in pmc_to_ref.keys():
    for job in jobs:
        ref_id = job['reference_id']
        job_id = job['reference_workflow_tag_id']
        if ref_id not in ref_to_pmc:
            # set_job_failure(job_id)
            print(f"job failed NO PMCID for {ref_id} and job {job_id}")
            continue
        pmcid = ref_to_pmc[ref_id]
        print(f"pmcid -> {pmcid} job_id-> {job_id} ref_id -> {ref_id}")
        ftp = getFtpPath(pmcid)
        if ftp is not None:
            print("Start Job")
            # set_job_started(job_id)
            download(ftp)
            getXmlFromTar(pmcid)
            try:
                results, status = deep_learning.get_genes_with_dl(
                    os.path.join(config_parser.get('PATHS', 'xml'), pmcid + ".nxml"),
                    gene_dict, fbid_to_symbol, EXCEPTIONS_PATH)
                if results:
                    print(f"results {results}")
                    for fbgn in results:
                        print(f"fbgn {fbgn}")
                        # send_entity_tag_to_abc(reference_curie=pmid,
                        #                       species=species,
                        #                       topic=args.topic,
                        #                       entity_type=args.topic,
                        #                       entity=fbgn,
                        #                       confidence_score=round(results[pmid][fbgn], 2),
                        #                       tet_source_id=tet_source_id,
                        #                       novel_data=False)
                        print(f"MATCH: reference_curie={ref_id}, entity={fbgn}, confidence_score={round(results[fbgn], 2)}")
                        # results[ref_id] = result
                    print("Finished successfully but with results :-)")
                    # set_job_success(job_id)
                else:
                    if status == 0:
                        print("Finished successfully but no results")
                        #send_entity_tag_to_abc(
                        #    reference_curie=ref_id,
                        #    species=species,
                        #    topic=args.topic,
                        #    negated=True,
                        #    tet_source_id=tet_source_id,
                        #    novel_data=False
                        #)
                        #set_job_success(job_id)
                        print(f"Job finished BUT No data. reference_curie={ref_id}")
                    else:
                        # set_job_failure(job_id)
                        print("job failed NO nxml")
                        # results[ref_id] = {'No_nxml': 0.000000000000000}
                if config_parser.getboolean('PARAMETERS', 'remove_files'):
                    removeFiles(pmcid)
            except Exception as e:
                set_job_failure(job_id)
                print("job failed somat went pear shaped")
                logging.warning(f"Error processing {pmcid}: {str(e)}")
        else:
            logging.warning(f"Error processing {pmcid}: No ftp file available")
            print("Start Failed No ftp file available")
            set_job_failure(job_id)
    with open(config_parser.get('PATHS', 'output'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        # write data to file
        # if we're using deep learning
        if config_parser.getboolean('PARAMETERS', 'use_deep_learning'):
            for pmid in results:
                for fbgn in results[pmid]:
                    # when using deep learning, we only output the pmid, fbgn, and confidence
                    writer.writerow([pmid, fbgn, results[pmid][fbgn]])
                    #send_entity_tag_to_abc(reference_curie=pmid,
                    #                       species=species,
                    #                       topic=args.topic,
                    #                       entity_type=args.topic,
                    #                       entity=fbgn,
                    #                       confidence_score=round(results[pmid][fbgn], 2),
                    #                       tet_source_id=tet_source_id,
                    #                       novel_data=False)
                    print(f"MATCH: reference_curie={pmid}, entity={fbgn}, confidence_score={round(results[pmid][fbgn], 2)}")


if __name__ == '__main__':
    main()

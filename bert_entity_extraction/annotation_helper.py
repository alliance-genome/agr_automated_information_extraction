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
import re
import argparse
import configparser
import sys
import logging
import subprocess
import requests
import xmltodict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from retry import retry
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
parser.add_argument("-t", "--topic", type=str, default='ATP:0000005',
                    help="Only run for this topic.", required=False)
parser.add_argument("-m", "--mod_abbreviation", type=str,
                    default="FB", help="Only run for FB.")
parser.add_argument("-c", "---config_file", type=str,
                    default="/usr/src/app/bert_entity_extraction/config.ini",
                    help="Config file for FlyBert")

args = parser.parse_args()
if args.stage:
    os.environ["ON_PRODUCTION"] = "no"
else:
    os.environ["ON_PRODUCTION"] = "yes"

config_parser = configparser.ConfigParser()
config_parser.read(args.config_file)

logger = logging.getLogger(__name__)
logger.debug(args)

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
        logger.debug(f"No password for env {db}_PASSWORD")
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
            logging.warning(f"ERROR: Could not find FTP path for {pmcid}: {error_code}")
        else:
            link = dict_data['OA']['records']['record']['link']
            if isinstance(link, list):
                ftplink = link[0]['@href']
            else:
                ftplink = link['@href']
            assert (".tar.gz" in ftplink)
        time.sleep(config_parser.getint('PARAMETERS', 'sleep_time_between_requests'))
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
    wget = f"wget -nc --timeout=10 --no-verbose -P {config_parser.get('PATHS', 'corpus')} {ftp}"
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
        # if just a number then ignore
        if re.search(r"^\d+$", row[1]):
            continue
        gene_dict[row[1]] = row[0]
        fbid_to_symbol[row[0]] = row[1]
    return gene_dict, fbid_to_symbol


def get_pmcids_for_references(jobs):
    """Look up pmc_ids for the reference in the jobs."""
    refs = [str(j['reference_id']) for j in jobs]
    ref_to_pmc = {}
    if refs:
        db_session = create_postgres_session('PSQL')
        # NOTE: remove [:2] from sql after testing
        sql = f"""SELECT reference_id, curie
                   FROM cross_reference
                    WHERE curie_prefix = 'PMCID'
                         AND reference_id in ({','.join(refs)})"""
        rs = db_session.execute(text(sql))
        rows = rs.fetchall()
        for row in rows:
            # PMCID:PMC11238292
            pmcid = row[1][6:]
            ref_to_pmc[row[0]] = pmcid
    else:
        logger.debug("No jobs to process")
    return ref_to_pmc


def get_data_from_alliance_db():
    """
    Get jobs to run.
    Get pmc to reference_id for those jobs
    """
    jobs = {}
    mod_topic_jobs = load_all_jobs("gene_extraction_job", args=args)
    ref_to_pmc = {}
    for (_, _), jobs in mod_topic_jobs.items():
        ref_to_pmc = get_pmcids_for_references(jobs)
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
    logger.debug(f"Number of jobs: {len(jobs)}")

    gene_dict, fbid_to_symbol = get_ateam_dicts()
    logger.debug(f"gene_dict has {len(gene_dict)} keys")
    key = list(gene_dict.keys())[0]
    logger.debug(f"gene_dict Example: key -> {key} {gene_dict[key]}")
    key = list(fbid_to_symbol.keys())[0]
    logger.debug(f"fbid_to_symbol has {len(fbid_to_symbol)} keys")
    logger.debug(f"fbid Exmaple: key -> {key} {fbid_to_symbol[key]}")

    if config_parser.getboolean('PARAMETERS', 'use_deep_learning'):
        deep_learning.initialize(config_parser.get('PATHS', 'deep_learning_model'))

    tet_source_id = get_tet_source_id(
        mod_abbreviation=args.mod_abbreviation,
        source_method="abc_entity_extractor",
        source_description="Alliance entity extraction pipeline using machine learning "
                           "to identify papers of interest for curation data types")
    # If we do other mods this will need changing to a look up
    species = 'NCBITaxon:7227'
    for job in jobs:
        logger.debug(f"Start Job {job}")
        if not set_job_started(job):
            logger.error(f"Problem setting to job started {job}!!!")
        ref_id = job['reference_id']
        if ref_id not in ref_to_pmc:
            logger.debug(f"job failed NO PMCID for reference: {ref_id} and job: {job['reference_workflow_tag_id']}")
            if not set_job_failure(job):
                logger.error(f"Problem setting to job failed {job}!!!")
            continue
        pmcid = ref_to_pmc[ref_id]
        logger.debug(f"pmcid -> {pmcid} job_id-> {job['reference_workflow_tag_id']} ref_id -> {ref_id}")
        ftp = getFtpPath(pmcid)
        if ftp is not None:
            download(ftp)
            getXmlFromTar(pmcid)
            try:
                results, status = deep_learning.get_genes_with_dl(
                    os.path.join(config_parser.get('PATHS', 'xml'), pmcid + ".nxml"),
                    gene_dict, fbid_to_symbol, EXCEPTIONS_PATH)
                if results:
                    okay = True
                    for fbgn in results:
                        confidence_level = 'Low'
                        if results[fbgn] > 0.7:
                            confidence_level = 'High'
                        elif results[fbgn] > 0.5:
                            confidence_level = 'Med'
                        logger.debug(f"MATCH: reference_curie={ref_id}, entity={fbgn}, confidence_score={round(results[fbgn], 2)}")
                        try:
                            stat = send_entity_tag_to_abc(
                                reference_curie=str(ref_id),
                                species=species,
                                topic=job['topic_id'],
                                entity_type=job['topic_id'],
                                entity=fbgn,
                                confidence_score=round(results[fbgn], 2),
                                confidence_level=confidence_level,
                                tet_source_id=tet_source_id,
                                novel_topic_qualifier='ATP:0000334')
                            if not stat:
                                logger.debug(f"""reference_curie={str(ref_id)},
                                species={species},
                                topic={job['topic_id']},
                                entity_type={job['topic_id']},
                                entity={fbgn},
                                confidence_score={round(results[fbgn], 2)},
                                confidence_level={confidence_level},
                                tet_source_id={tet_source_id},
                                novel_data=False""")
                                okay = False
                        except Exception as e:
                            okay = False
                            logger.error(f"Problem sending entity tag to abc: {e}")
                    if okay:
                        logger.debug("Finished successfully but with results :-)")
                        if not set_job_success(job):
                            logger.error(f"Problem setting to job success {job}!!!")
                else:
                    if status == 0:
                        stat = send_entity_tag_to_abc(
                            reference_curie=str(ref_id),
                            species=species,
                            topic=job['topic_id'],
                            negated=True,
                            tet_source_id=tet_source_id,
                            novel_data=False,
                            novel_topic_qualifier='ATP:0000335'
                        )
                        if not stat:
                            logger.error(f"PROBLEM sending negated job data {job}!!!")
                        logger.debug(f"Job finished BUT No data. reference_curie={ref_id}")
                        if not set_job_success(job):
                            logger.error(f"Problem setting to job success {job}!!!")
                    else:
                        logger.debug("job failed NO nxml")
                        if not set_job_failure(job):
                            logger.error(f"Problem setting to job failure {job}!!!")
                if config_parser.getboolean('PARAMETERS', 'remove_files'):
                    removeFiles(pmcid)
            except Exception as e:
                logger.error(f"job failed some thing went pear shaped. {e}")
                if not set_job_failure(job):
                    logger.debug(f"Problem setting to job started {job}!!!")
                logging.error(f"Error processing {pmcid}: {str(e)}")
        else:
            logging.warning(f"Error processing {pmcid}: No ftp file available")
            set_job_failure(job)


if __name__ == '__main__':
    main()

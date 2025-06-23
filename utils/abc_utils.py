import io
import json
import logging
import os
import html
import time
import urllib.request
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional
from urllib.error import HTTPError
from argparse import Namespace
import psycopg2
import requests
from fastapi_okta.okta_utils import get_authentication_token, generate_headers


blue_api_base_url = os.environ.get('ABC_API_SERVER', "https://literature-rest.alliancegenome.org")
if blue_api_base_url.startswith('literature'):
    blue_api_base_url = f"https://{blue_api_base_url}"
curation_api_base_url = os.environ.get('CURATION_API_SERVER', "https://curation.alliancegenome.org/api/")

logger = logging.getLogger(__name__)

cache = {}


def set_blue_api_base_url(value):
    global blue_api_base_url
    blue_api_base_url = value


def get_mod_species_map():
    url = f'{blue_api_base_url}/mod/taxons/default'
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return {mod["mod_abbreviation"]: mod["taxon_id"] for mod in resp_obj}
    except HTTPError as e:
        logger.error(e)


def get_mod_id_from_abbreviation(mod_abbreviation):
    url = f'{blue_api_base_url}/mod/{mod_abbreviation}'
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return resp_obj["mod_id"]
    except HTTPError as e:
        logger.error(e)


def get_cached_mod_species_map():
    if 'mod_species_map' not in cache:
        cache['mod_species_map'] = get_mod_species_map()
    return cache['mod_species_map']


def get_cached_mod_id_from_abbreviation(mod_abbreviation):
    if 'mod_abbreviation_id' not in cache:
        cache['mod_abbreviation_id'] = {}
    if mod_abbreviation not in cache['mod_abbreviation_id']:
        cache['mod_abbreviation_id'][mod_abbreviation] = get_mod_id_from_abbreviation(mod_abbreviation)
    return cache['mod_abbreviation_id'][mod_abbreviation]


def get_cached_mod_abbreviation_from_id(mod_id):
    if 'mod_id_abbreviation' not in cache:
        cache['mod_id_abbreviation'] = {}
        for mod_abbreviation in get_cached_mod_species_map().keys():
            cache['mod_id_abbreviation'][get_cached_mod_id_from_abbreviation(mod_abbreviation)] = mod_abbreviation
    return cache['mod_id_abbreviation'][mod_id]


def get_curie_from_reference_id(reference_id):
    url = f'{blue_api_base_url}/reference/{reference_id}'
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return resp_obj["curie"]
    except HTTPError as e:
        logger.error(e)


def get_tet_source_id(mod_abbreviation: str, source_method: str, source_description: str):
    url = (f'{blue_api_base_url}/topic_entity_tag/source/ECO:0008004/{source_method}/{mod_abbreviation}'
           f'/{mod_abbreviation}')
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return int(resp_obj["topic_entity_tag_source_id"])
    except HTTPError as e:
        if e.code == 404:
            # Create a new source if not exists
            create_url = f'{blue_api_base_url}/topic_entity_tag/source'
            token = get_authentication_token()
            headers = generate_headers(token)
            create_data = json.dumps({
                "source_evidence_assertion": "ECO:0008004",
                "source_method": source_method,
                "validation_type": None,
                "description": source_description,
                "data_provider": mod_abbreviation,
                "secondary_data_provider_abbreviation": mod_abbreviation
            }).encode('utf-8')
            create_request = urllib.request.Request(url=create_url, data=create_data, method='POST', headers=headers)
            create_request.add_header("Content-type", "application/json")
            create_request.add_header("Accept", "application/json")
            try:
                with urllib.request.urlopen(create_request) as create_response:
                    create_resp = create_response.read().decode("utf8")
                    return int(create_resp)
            except HTTPError as create_e:
                logger.error(f"Failed to create source: {create_e}")
        else:
            logger.error(e)
            raise


def send_classification_tag_to_abc(reference_curie: str, species: str, topic: str, negated: bool,
                                   novel_flag: bool, confidence_level: str, tet_source_id):
    url = f'{blue_api_base_url}/topic_entity_tag/'
    token = get_authentication_token()
    tet_data = json.dumps({
        "created_by": "default_user",
        "updated_by": "default_user",
        "topic": topic,
        # "species": get_cached_mod_species_map()[mod_abbreviation],
        "species": species,
        "topic_entity_tag_source_id": tet_source_id,
        "negated": negated,
        "novel_topic_data": novel_flag,
        "confidence_level": confidence_level,
        "reference_curie": reference_curie,
        "force_insertion": True
    }).encode('utf-8')
    headers = generate_headers(token)
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            create_request = urllib.request.Request(url=url, data=tet_data, method='POST', headers=headers)
            create_request.add_header("Content-type", "application/json")
            create_request.add_header("Accept", "application/json")
            with urllib.request.urlopen(create_request) as create_response:
                if create_response.getcode() == 201:
                    logger.debug("TET created")
                else:
                    logger.error(f"Failed to create TET (attempt {attempts}): {str(tet_data)}")
            return True
        except requests.exceptions.RequestException as exc:
            if attempts >= 3:
                logger.error(f"Error trying to send classification tag to ABC {attempts} times.")
                logger.error(f"curie: {reference_curie}, species: {species}, topic: {topic}")
                logger.error(f"novel_flag: {novel_flag}, negated: {negated}, confidence: {confidence_level}, tet_source_id: {tet_source_id}")
                raise RuntimeError("Error Sending classification tag to abc FAILED") from exc
            time.sleep(attempts)
    return False


def send_entity_tag_to_abc(reference_curie: str, species: str, novel_data: bool, topic: str, tet_source_id: int, entity: Optional[str] = None, entity_type: Optional[str] = None, negated: bool = False):
    url = f'{blue_api_base_url}/topic_entity_tag/'
    token = get_authentication_token()
    tet_data = json.dumps({
        "created_by": "default_user",
        "updated_by": "default_user",
        "topic": topic,
        "entity_type": entity_type,
        "entity_id_validation": "alliance" if entity else None,
        "entity": entity,
        "species": species,
        "topic_entity_tag_source_id": tet_source_id,
        "negated": negated,
        "novel_topic_data": novel_data,
        "confidence_level": None,
        "reference_curie": reference_curie,
        "force_insertion": True
    }).encode('utf-8')
    headers = generate_headers(token)
    try:
        create_request = urllib.request.Request(url=url, data=tet_data, method='POST', headers=headers)
        create_request.add_header("Content-type", "application/json")
        create_request.add_header("Accept", "application/json")
        with urllib.request.urlopen(create_request) as create_response:
            if create_response.getcode() == 201:
                logger.debug("TET created")
                return True
            else:
                logger.error(f"Failed to create TET: {str(tet_data)}")
                return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred during TET upload: {e}")
        return False


def get_jobs_batch(job_label: str = "classification_job", limit: int = 1000, offset: int = 0, args: Namespace = None):
    jobs_url = f'{blue_api_base_url}/workflow_tag/jobs/{job_label}?limit={limit}&offset={offset}'
    if args and args.mod_abbreviation:
        jobs_url += f'&mod={args.mod_abbreviation}'
    if args and args.reference_curie:
        jobs_url += f'&reference={args.reference_curie}'
    if args and args.topic:
        jobs_url += f'&topic={args.topic}'
    request = urllib.request.Request(url=jobs_url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return resp_obj
    except HTTPError as e:
        logger.error(e)


def set_job_started(job):
    url = f'{blue_api_base_url}/workflow_tag/job/started/{job["reference_workflow_tag_id"]}'
    token = get_authentication_token()
    headers = generate_headers(token)
    request = urllib.request.Request(url=url, method='POST', headers=headers)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            urllib.request.urlopen(request)
            return True
        except HTTPError:
            time.sleep(attempts)
    logger.error(f"Error setting job started after 3 attempts: {str(job)}")
    return False


def set_job_success(job):
    url = f'{blue_api_base_url}/workflow_tag/job/success/{job["reference_workflow_tag_id"]}'
    token = get_authentication_token()
    headers = generate_headers(token)
    request = urllib.request.Request(url=url, method='POST', headers=headers)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            urllib.request.urlopen(request)
            logger.debug("Successfully set job started")
            return True
        except HTTPError as e:
            logger.warning(f"Error setting to success for : {str(job)}: {e}")
            time.sleep(attempts)

    logger.error(f"Error setting job success after 3 attempts: {str(job)}")
    return False


def set_job_failure(job):
    url = f'{blue_api_base_url}/workflow_tag/job/failed/{job["reference_workflow_tag_id"]}'
    token = get_authentication_token()
    headers = generate_headers(token)
    request = urllib.request.Request(url=url, method='POST', headers=headers)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            urllib.request.urlopen(request)
            return True
        except HTTPError:
            time.sleep(attempts)
    logger.error(f"Error setting job failed: {str(job)}")
    return False


def get_file_from_abc_reffile_obj(referencefile_json_obj):
    file_download_api = (f"{blue_api_base_url}/reference/referencefile/download_file/"
                         f"{referencefile_json_obj['referencefile_id']}")
    token = get_authentication_token()
    headers = generate_headers(token)
    try:
        response = requests.request("GET", file_download_api, headers=headers)
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred for accessing/retrieving data from {file_download_api}: error={e}")
        return None


def get_curie_from_xref(xref):
    ref_by_xref_api = f'{blue_api_base_url}/reference/by_cross_reference/{xref}'
    request = urllib.request.Request(url=ref_by_xref_api)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return resp_obj["curie"]
    except HTTPError as e:
        logger.error(e)


def get_pmids_from_reference_curies(curies: List[str]):
    curie_pmid = {}
    for curie in curies:
        ref_data_api = f'{blue_api_base_url}/reference/{curie}'
        request = urllib.request.Request(url=ref_data_api)
        request.add_header("Content-type", "application/json")
        request.add_header("Accept", "application/json")
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            try:
                curie_pmid[curie] = [xref["curie"] for xref in resp_obj["cross_references"]
                                     if xref["curie"].startswith("PMID")][0]
            except IndexError:
                curie_pmid[curie] = None
    return curie_pmid


def get_link_title_abstract_and_tpc(curie):
    ref_data_api = f'{blue_api_base_url}/reference/{curie}'
    request = urllib.request.Request(url=ref_data_api)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return (f'https://literature.alliancegenome.org/Biblio?action=display&referenceCurie={curie}',
                    resp_obj["title"], resp_obj["abstract"])
    except HTTPError as e:
        logger.error(e)


def download_main_pdf(agr_curie, mod_abbreviation, file_name, output_dir):
    all_reffiles_for_pap_api = f'{blue_api_base_url}/reference/referencefile/show_all/{agr_curie}'
    request = urllib.request.Request(url=all_reffiles_for_pap_api)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            main_pdf_ref_file = None
            main_pdf_referencefiles = [ref_file for ref_file in resp_obj if
                                       ref_file["file_class"] == "main"
                                       and ref_file["file_publication_status"] == "final"
                                       and ref_file["file_extension"] == "pdf"]
            for ref_file in main_pdf_referencefiles:
                if any(ref_file_mod["mod_abbreviation"] == mod_abbreviation for ref_file_mod in
                       ref_file["referencefile_mods"]):
                    main_pdf_ref_file = ref_file
                    break
            else:
                for ref_file in main_pdf_referencefiles:
                    if any(ref_file_mod["mod_abbreviation"] is None for ref_file_mod in
                       ref_file["referencefile_mods"]):
                        main_pdf_ref_file = ref_file
                        break
            if main_pdf_ref_file:
                file_content = get_file_from_abc_reffile_obj(main_pdf_ref_file)
                with open(os.path.join(output_dir, file_name + ".pdf"), "wb") as out_file:
                    out_file.write(file_content)
    except HTTPError as e:
        logger.error(e)


def download_bib_data_for_need_prioritization_references(output_dir: str, mod_abbreviation):
    logger.info("Started retrieving bib data")
    os.makedirs(output_dir, exist_ok=True)
    url = f"{blue_api_base_url}/sort/need_prioritization?mod_abbreviation={mod_abbreviation}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            for x in response.json():
                reference_curie = x['curie']
                title = html.unescape(x['title'] or "")
                abstract = html.unescape(x['abstract'] or "")
                with open(os.path.join(output_dir, reference_curie + ".txt"), "w") as out_file:
                    out_file.write(f"title|{title}\nabstract|{abstract}\n")
                logger.info(f"{reference_curie}: the txt file is generated.")
        else:
            logger.info(f"Bib data not found for {url}: status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.info(f"Error occurred for accessing/retrieving bib data from {url}: error={e}")


def download_bib_data_for_references(reference_curies: List[str], output_dir: str, mod_abbreviation):
    logger.info("Started retrieving bib data")
    os.makedirs(output_dir, exist_ok=True)
    for _, reference_curie in enumerate(reference_curies, start=1):
        bib_url = f"{blue_api_base_url}/reference/get_bib_info/{reference_curie}?mod_abbreviation={mod_abbreviation}&return_format=txt"
        token = get_authentication_token()
        headers = generate_headers(token)
        try:
            response = requests.request("GET", bib_url, headers=headers)
            if response.status_code == 200:
                content = response.text.strip('"').replace('\\n', '\n')
                with open(os.path.join(output_dir, reference_curie + ".txt"), "w") as out_file:
                    out_file.write(content)
            else:
                logger.info(f"Bib data not found for {bib_url}: status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.info(f"Error occurred for accessing/retrieving bib data from {bib_url}: error={e}")


def download_tei_files_for_references(reference_curies: List[str], output_dir: str, mod_abbreviation):
    logger.info("Started downloading TEI files")
    for reference_curie in reference_curies:
        all_reffiles_for_pap_api = f'{blue_api_base_url}/reference/referencefile/show_all/{reference_curie}'
        request = urllib.request.Request(url=all_reffiles_for_pap_api)
        request.add_header("Content-type", "application/json")
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request) as response:
                resp = response.read().decode("utf8")
                resp_obj = json.loads(resp)
                for ref_file in resp_obj:
                    if ref_file["file_extension"] == "tei" and ref_file["file_class"] == "tei" and any(
                            ref_file_mod["mod_abbreviation"] == mod_abbreviation for ref_file_mod in
                            ref_file["referencefile_mods"]):
                        file_content = get_file_from_abc_reffile_obj(ref_file)
                        if file_content:
                            filename = os.path.join(output_dir, reference_curie.replace(":", "_") + ".tei")
                            with open(filename, "wb") as out_file:
                                out_file.write(file_content)
                        else:
                            logger.error(f"No TEI file found for {reference_curie}")
        except HTTPError as e:
            logger.error(e)
    logger.info("Finished downloading TEI files")


def convert_pdf_with_grobid(file_content):
    grobid_api_url = os.environ.get("GROBID_API_URL",
                                    "https://grobid.alliancegenome.org/api/processFulltextDocument")
    # Send the file content to the GROBID API
    response = requests.post(grobid_api_url, files={'input': ("file", file_content)})
    return response


def get_model_data(mod_abbreviation: str, task_type: str, topic: str):
    model_data = None
    get_model_url = f"{blue_api_base_url}/ml_model/metadata/{task_type}/{mod_abbreviation}/{topic}"
    token = get_authentication_token()
    headers = generate_headers(token)

    # Make the request to download the model
    response = requests.get(get_model_url, headers=headers)
    if response.status_code == 200:
        model_data = response.json()
        logger.info("Model meta data downloaded successfully.")
    else:
        logger.error(f"Failed to download model meta data: {response.text}")
        response.raise_for_status()
    return model_data


def download_abc_model(mod_abbreviation: str, task_type: str, output_path: str, topic: str = None):
    # TODO: Question, How can there be NO topic?
    download_url = f"{blue_api_base_url}/ml_model/download/{task_type}/{mod_abbreviation}/{topic}" if (
        topic is not None) else f"{blue_api_base_url}/ml_model/download/{task_type}/{mod_abbreviation}"
    token = get_authentication_token()
    headers = generate_headers(token)

    # Make the request to download the model
    response = requests.get(download_url, headers=headers)

    if response.status_code == 200:
        with open(output_path, "wb") as model_file:
            model_file.write(response.content)
        logger.info("Model downloaded successfully.")
    else:
        logger.error(f"Failed to download model: {response.text}")
        response.raise_for_status()


def upload_ml_model(task_type: str, mod_abbreviation: str, model_path, stats: dict, dataset_id: int = None,
                    topic: str = None, file_extension: str = "", production: Union[bool, None] = False,
                    no_data: Union[bool, None] = True, species: Union[str, None] = None, novel_data: Union[bool, None] = False):
    upload_url = f"{blue_api_base_url}/ml_model/upload"
    token = get_authentication_token()
    headers = generate_headers(token)

    # Prepare the metadata payload
    metadata = {
        "task_type": task_type,
        "mod_abbreviation": mod_abbreviation,
        "topic": topic,
        "version_num": None,
        "file_extension": file_extension,
        "model_type": stats["model_name"],
        "precision": stats["average_precision"],
        "recall": stats["average_recall"],
        "f1_score": stats["average_f1"],
        "parameters": str(stats["best_params"]) if stats["best_params"] is not None else None,
        "dataset_id": dataset_id,
        "production": production,
        "negated": no_data,
        "novel_topic_data": novel_data,
        "species": species
    }

    model_dir = os.path.dirname(model_path)
    if topic is None:
        topic = "notopic"
    metadata_filename = f"{task_type}_{mod_abbreviation}_{topic.replace(':', '_')}_metadata.json"
    metadata_path = os.path.join(model_dir, metadata_filename)
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    model_data_json = json.dumps(metadata)

    # Prepare the files payload
    files = {
        "file": (f"{mod_abbreviation}_{topic.replace(':', '_')}.{file_extension}", open(model_path, "rb"),
                 "application/octet-stream"),
        "model_data_file": ("model_data.txt", io.BytesIO(model_data_json.encode('utf-8')), "text/plain")
    }

    # Make the request to upload the model
    mod_headers = headers.copy()
    del mod_headers["Content-Type"]
    response = requests.post(upload_url, headers=mod_headers, files=files, data=metadata)

    if response.status_code == 201:
        logger.info("Model uploaded successfully.")
        logger.info(f"A copy of the model has been saved to: {model_path}.")
        logger.info(f"The following metadata was uploaded: {metadata}")
        logger.info(f"Metadata file saved to: {metadata_path}")
    else:
        logger.error(f"Failed to upload model: {response.text}")
        response.raise_for_status()


# Function to create a dataset
def create_dataset(title: str, description: str, mod_abbreviation: str, topic: str, dataset_type: str) -> (int, int):
    create_dataset_url = f"{blue_api_base_url}/datasets/"
    token = get_authentication_token()
    headers = generate_headers(token)
    payload = {
        "title": title,
        "description": description,
        "mod_abbreviation": mod_abbreviation,
        "data_type": topic,
        "dataset_type": dataset_type
    }
    response = requests.post(create_dataset_url, json=payload, headers=headers)
    if response.status_code == 201:
        dataset_id = response.json()["dataset_id"]
        version = response.json()["version"]
        logger.info(f"Dataset created with ID: {dataset_id}")
        return dataset_id, version
    else:
        logger.error(f"Failed to create dataset: {response.text}")
        response.raise_for_status()


# Function to add an entry to the dataset
def add_entry_to_dataset(mod_abbreviation: str, topic: str, dataset_type: str, version: int, reference_curie: str,
                         classification_value: str):
    add_entry_url = f"{blue_api_base_url}/datasets/data_entry/"
    token = get_authentication_token()
    headers = generate_headers(token)
    payload = {
        "mod_abbreviation": mod_abbreviation,
        "data_type": topic,
        "dataset_type": dataset_type,
        "version": version,
        "reference_curie": reference_curie,
        "classification_value": classification_value
    }
    response = requests.post(add_entry_url, json=payload, headers=headers)
    if response.status_code == 201:
        logger.info("Entry added to dataset")
    else:
        logger.error(f"Failed to add entry to dataset: {response.text}")
        response.raise_for_status()


def set_indexing_priority(ref_curie, mod_abbr, priority_name):
    indexing_url = f"{blue_api_base_url}/workflow_tag/set_priority/{ref_curie}/{mod_abbr}/{priority_name}"
    token = get_authentication_token()
    headers = generate_headers(token)
    response = requests.post(indexing_url, headers=headers)
    if response.status_code == 200:
        logger.info(f"{ref_curie} is successfully set to {priority_name}")
    else:
        logger.error(f"Failed to set the indexing priority: {response.text}")


def get_training_set_from_abc(mod_abbreviation: str, topic: str, metadata_only: bool = False):
    endpoint = "metadata" if metadata_only else "download"
    response = requests.get(f"{blue_api_base_url}/datasets/{endpoint}/{mod_abbreviation}/{topic}/document/")
    if response.status_code == 200:
        dataset = response.json()
        logger.info(f"Dataset {endpoint} downloaded successfully.")
        return dataset
    else:
        logger.error(f"Failed to download dataset {response.text}")
        response.raise_for_status()


def get_entity_name(entity_type, a_team_api_search_result_obj, mod_abbreviation: str = None):
    if entity_type == 'gene':
        gene_name = a_team_api_search_result_obj['geneSymbol']['formatText']
        if mod_abbreviation and mod_abbreviation == 'WB':
            return gene_name.lower()
        return gene_name
    elif entity_type == 'protein':
        # currently just for WB
        gene_name = a_team_api_search_result_obj['geneSymbol']['formatText']
        return gene_name.upper()
    elif entity_type == 'allele':
        return a_team_api_search_result_obj['alleleSymbol']['formatText']
    elif entity_type == 'fish':
        if a_team_api_search_result_obj['subtype']['name'] != 'fish':
            return None
        return a_team_api_search_result_obj['name']


def get_all_curated_entities(mod_abbreviation: str, entity_type_str):
    all_curated_entity_names = []
    entity_name_curie_mappings = {}
    params = {
        "searchFilters": {
            "dataProviderFilter": {
                "dataProvider.abbreviation": {
                    "queryString": mod_abbreviation,
                    "tokenOperator": "OR"
                }
            }
        },
        "sortOrders": [],
        "aggregations": [],
        "nonNullFieldsTable": []
    }
    if entity_type_str in ["strain", "genotype", "fish"]:
        params["searchFilters"]["subtypeFilter"] = {
            "subtype.name": {
                "queryString": entity_type_str,
                "tokenOperator": "OR"
            }
        }
        entity_type_str = "agm"

    current_page = 0
    while True:
        logger.info(f"Fetching page {current_page} of entities from A-team API")
        url = f"{curation_api_base_url}{entity_type_str}/search?limit=2000&page={current_page}"
        request_data_encoded = json.dumps(params).encode('utf-8')
        request = urllib.request.Request(url, data=request_data_encoded)
        request.add_header("Authorization", f"Bearer {get_authentication_token()}")
        request.add_header("Content-type", "application/json")
        request.add_header("Accept", "application/json")

        with urllib.request.urlopen(request) as response:
            resp_obj = json.loads(response.read().decode("utf8"))

        if resp_obj['returnedRecords'] < 1:
            break

        for result in resp_obj['results']:
            if result['obsolete'] or result['internal']:
                continue
            entity_name = None
            if entity_type_str == "agm":
                if 'agmFullName' in result and 'displayText' in result['agmFullName']:
                    entity_name = result['agmFullName']['displayText']
                else:
                    entity_name = get_entity_name(entity_type_str, result, mod_abbreviation)
                if entity_name:
                    all_curated_entity_names.append(entity_name)
                    entity_name_curie_mappings[entity_name] = result['primaryExternalId']
        current_page += 1
    return all_curated_entity_names, entity_name_curie_mappings


def get_all_ref_curies(mod_abbreviation: str):
    db_params = {
        "dbname": os.getenv("DB_NAME", "default_dbname"),
        "user": os.getenv("DB_USER", "default_user"),
        "password": os.getenv("DB_PASSWORD", "default_password"),
        "host": os.getenv("DB_HOST", "default_host"),
        "port": os.getenv("DB_PORT", "default_port")
    }

    curies = []
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()

        mod_query = f"SELECT mod_id FROM mod WHERE abbreviation = '{mod_abbreviation}'"
        cursor.execute(mod_query)
        mod_id = cursor.fetchone()[0]
        # Query to fetch CURIEs
        query = f"""
            SELECT curie
                FROM reference
                WHERE reference_id IN (
                    SELECT reference_id
                    FROM mod_corpus_association
                    WHERE mod_id = {mod_id} AND corpus is true)"""
        cursor.execute(query)

        # Fetch the result
        curies = [row[0] for row in cursor.fetchall()]

        # Close cursor and connection
        cursor.close()
        connection.close()
    except Exception as e:
        logger.error(f"Error while fetching CURIEs from database: {e}")

    return curies


def load_all_jobs(job_label: str, args: Namespace) -> Dict[Tuple[str, str], List[dict]]:
    """
    Loads and processes all jobs with a specified label from an external source, organizing
    them by module ID and topic.

    This function retrieves jobs in batches from an external source, identified by a
    specific `job_label`. It processes each job, ensuring duplicates (based on their
    module ID, topic, and reference ID) are not added multiple times. The jobs are
    grouped by `(module ID, topic)` and returned in a dictionary for further usage.

    :param job_label: The label used to filter and load jobs from the external database.
    :type job_label: str
    :param args: The arguments passed to the external source.
    :return: A dictionary where keys are tuples of `(module ID, topic)` and the values
        are lists of job dictionaries filtered and grouped accordingly.
    :rtype: Dict[Tuple[str, str], List[dict]]
    """
    mod_datatype_jobs = defaultdict(list)
    limit = 1000
    offset = 0
    jobs_already_added = set()
    logger.info("Loading jobs from ABC ...")

    while all_jobs := get_jobs_batch(job_label=job_label, limit=limit, offset=offset, args=args):
        for job in all_jobs:
            reference_id = job["reference_id"]
            topic = job["topic_id"]
            mod_id = job["mod_id"]
            if (mod_id, topic, reference_id) not in jobs_already_added:
                mod_datatype_jobs[(mod_id, topic)].append(job)
                jobs_already_added.add((mod_id, topic, reference_id))
        offset += limit

    logger.info("Finished loading jobs to classify from ABC ...")
    return mod_datatype_jobs

import logging

from agr_curation_api import AGRCurationAPIClient, APIConfig

PAGE_LIMIT = 1000

# will add XB one later
MOD_TAXON_MAPPING = {
    'RGD': 'NCBITaxon:10116',
    'MGI': 'NCBITaxon:10090',
    'ZFIN': 'NCBITaxon:7955',
    'WB': 'NCBITaxon:6239',
    'SGD': 'NCBITaxon:559292',
    'FB': 'NCBITaxon:7227'
}

_CURATED_ENTITY_CACHE: dict[tuple[str, str], tuple[list[str], dict[str, str]]] = {}

logger = logging.Logger(__name__)


def species_to_exclude():
    return {
        "NCBITaxon:4853",
        "NCBITaxon:30023",
        "NCBITaxon:8805",
        "NCBITaxon:216498",
        "NCBITaxon:1420681",
        "NCBITaxon:10231",
        "NCBITaxon:156766",
        "NCBITaxon:80388",
        "NCBITaxon:101142",
        "NCBITaxon:31138",
        "NCBITaxon:88086",
        "NCBITaxon:34245",
        "NCBITaxon:5482",
        "NCBITaxon:1",
        "NCBITaxon:2"
    }


def fetch_entities_page(api_client: AGRCurationAPIClient, mod: str, entity_type: str, page: int):
    """Fetch a single page of entities from the API."""
    if entity_type == 'species':
        return api_client.get_species(limit=PAGE_LIMIT, page=page)

    taxon = MOD_TAXON_MAPPING.get(mod)
    if not taxon:
        logger.info(f"'{mod}' is not in the MOD_TAXON_MAPPING")
        return []

    if entity_type == 'gene':
        # return api_client.get_genes(data_provider=mod, limit=PAGE_LIMIT, page=page)
        return api_client.get_genes(
            data_provider=mod,
            taxon=taxon,
            limit=PAGE_LIMIT,
            offset=page * PAGE_LIMIT,
            data_source='db'
        )

    elif entity_type == 'transgene':
        return api_client.get_alleles(
            data_provider=mod,
            limit=PAGE_LIMIT,
            page=page,
            transgenes_only=True
        )
    elif entity_type == 'allele':
        # WB extraction subset: force DB + correct params
        if mod == 'WB':
            return api_client.get_alleles(
                taxon=taxon,
                limit=PAGE_LIMIT,
                offset=page * PAGE_LIMIT,
                wb_extraction_subset=True,
                data_source='db'
            )
        # other MODs: use normal API/GraphQL path
        return api_client.get_alleles(
            data_provider=mod,
            limit=PAGE_LIMIT,
            page=page
        )
    elif entity_type in ['strain', 'genotype', 'fish']:
        return api_client.get_agms(
            data_provider=mod,
            subtype=entity_type,
            limit=PAGE_LIMIT,
            page=page
        )
    else:
        logger.info(f"Unknown entity_type '{entity_type}' requested; returning empty list.")
        return []


def get_name_from_entity(entity_symbol):
    if entity_symbol is None:
        return None
    if getattr(entity_symbol, "obsolete", False) or getattr(entity_symbol, "internal", False):
        return None
    if hasattr(entity_symbol, 'formatText'):
        return entity_symbol.formatText
    elif hasattr(entity_symbol, 'displayText'):
        return entity_symbol.displayText
    return None


def get_all_curated_entities(mod_abbreviation: str, entity_type_str: str, *, force_refresh: bool = False):  # noqa: C901
    """
    Return (all_curated_entity_names, entity_name_curie_mappings)
    Results are cached per (mod_abbreviation, original_entity_type_str)
    """
    cache_key = (mod_abbreviation, entity_type_str)

    if not force_refresh and cache_key in _CURATED_ENTITY_CACHE:
        names, mapping = _CURATED_ENTITY_CACHE[cache_key]
        return names.copy(), mapping.copy()

    all_curated_entity_names: list[str] = []
    entity_name_curie_mappings: dict[str, str] = {}

    api_config = APIConfig()  # type: ignore
    api_client = AGRCurationAPIClient(api_config)

    if entity_type_str == 'transgenic_allele':
        entity_type_str = 'transgene'

    species_to_exclude_set = species_to_exclude()

    current_page = 0
    while True:
        entities = fetch_entities_page(api_client, mod_abbreviation, entity_type_str, current_page)
        if not entities:
            logger.info(f"No entities returned for {entity_type_str} page {current_page}")
            break

        for entity in entities:
            entity_name = None
            curie = None
            if entity_type_str == 'species':
                if hasattr(entity, 'curie'):
                    if entity.obsolete or entity.internal:
                        continue
                    curie = entity.curie
                    entity_name = entity.name
                    if curie in species_to_exclude_set:
                        continue
            else:
                if hasattr(entity, 'primaryExternalId'):
                    curie = entity.primaryExternalId
                if not curie:
                    continue
                if entity_type_str == 'gene':
                    if hasattr(entity, 'geneSymbol'):
                        entity_symbol = entity.geneSymbol
                elif entity_type_str in ['transgene', 'allele']:
                    if hasattr(entity, 'alleleSymbol'):
                        entity_symbol = entity.alleleSymbol
                elif entity_type_str in ['fish', 'genotype', 'strain']:
                    entity_symbol = entity.agmFullName
                entity_name = get_name_from_entity(entity_symbol)
            if not entity_name or not curie:
                continue
            if entity_name not in entity_name_curie_mappings:
                all_curated_entity_names.append(entity_name)
            entity_name_curie_mappings[entity_name] = curie
        current_page += 1

    # Deduplicate + stable sort output names
    all_curated_entity_names = sorted(set(all_curated_entity_names), key=str.lower)

    _CURATED_ENTITY_CACHE[cache_key] = (
        all_curated_entity_names.copy(),
        entity_name_curie_mappings.copy()
    )
    if mod_abbreviation == 'WB' and entity_type_str == 'strain':
        wbid = entity_name_curie_mappings.get("HT115(DE3)")
        if wbid and entity_name_curie_mappings.get("HT115"):
            entity_name_curie_mappings["HT115"] = wbid
    return all_curated_entity_names, entity_name_curie_mappings

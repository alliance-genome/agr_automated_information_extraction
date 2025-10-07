# this script is updated by AI
"""
species_taxon.py

Parent/child species relationships for the species extraction pipeline.

What this returns
-----------------
Dict[str, Set[str]] mapping parent CURIE -> set of child CURIEs

Key behavior
------------
- Fetches lineage for curated taxa using resilient fallbacks (NCBI v2 → v2alpha → Ensembl)
  and constructs edges via "nearest curated ancestor → child".
- **Merges** the online map with the baked DEFAULT map (unique parent list, union of children),
  then filters to your curated set (so you'll keep species-level parents from the default and
  non-species anchors like 'NCBITaxon:2' from the online run).
- Same cache file format as before: JSON {parent: [child,...]}.

NOTE: The baked DEFAULT_PARENT_CHILDREN is defined at the BOTTOM of this file so you can
      easily drop in a newly merged mapping later.
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
import requests
import logging
import base64
import json
import os

logger = logging.getLogger(__name__)

# -------- Endpoints & simple settings --------
_NCBI_V2_TMPL = "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon/{taxid}?parents=true&children=true"
_NCBI_V2A_TMPL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/taxonomy/taxon/{taxid}?parents=true&children=true"
_ENSEMBL_CLASSIF = "https://rest.ensembl.org/taxonomy/classification/{taxid}?content-type=application/json"

# Optional per-taxid lineage cache on disk (speeds re-runs)
LINEAGE_CACHE_DIR = os.getenv("SPECIES_TAXON_CACHE_DIR", ".species_taxon_cache")
MAX_WORKERS = int(os.getenv("SPECIES_TAXON_MAX_WORKERS", "8"))  # tune if you see throttling


def _extract_taxid(curie: str) -> Optional[int]:
    if not curie:
        return None
    m = re.search(r"(?:NCBITaxon:)?(\d+)$", curie.strip())
    return int(m.group(1)) if m else None


def _params_with_api_key(url: str) -> str:
    key = os.getenv("NCBI_API_KEY")
    if not key:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}api_key={key}"


def _api_get_json(url: str, timeout: float, retries: int, sleep: float, headers: Optional[Dict[str, str]] = None):
    """Resilient GET with exponential backoff; returns dict/list or None."""
    attempt = 0
    headers = headers or {
        "User-Agent": "AGR-species-extraction/1.2",
        "Accept": "application/json",
    }
    while True:
        attempt += 1
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    safe_url = urlparse(url)._replace(query="").geturl()
                    logger.warning("Non-JSON response from %s", safe_url)
                    return None
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt <= retries:
                    time.sleep(sleep * attempt)
                    continue
                return None
            return None
        except requests.RequestException:
            if attempt <= retries:
                time.sleep(sleep * attempt)
                continue
            return None


def _ints_from_any(x) -> List[int]:
    out: List[int] = []
    for v in x or []:
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, str) and v.isdigit():
            out.append(int(v))
    return out


def _lin_cache_path(tid: int) -> Path:
    p = Path(LINEAGE_CACHE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"lin_{tid}.json"


def _lin_cache_get(tid: int) -> Optional[dict]:
    p = _lin_cache_path(tid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _lin_cache_put(tid: int, payload: dict) -> None:
    try:
        data = json.dumps(payload).encode("utf-8")
        encoded = base64.b64encode(data).decode("ascii")
        _lin_cache_path(tid).write_text(encoded, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to cache lineage for %s: %s", tid, e)


def _parse_ncbi_node(data: dict) -> Tuple[List[int], Optional[int], str, List[Tuple[str, int]]]:
    """
    Parse a NCBI node JSON into:
      parents (list[int] as given),
      parent_tid (immediate parent if provided),
      rank (str or ''), and
      classification ranks as list of (rank, id) in fixed order.
    """
    parents: List[int] = []
    parent_tid: Optional[int] = None
    rank = ""
    cls: List[Tuple[str, int]] = []

    if not isinstance(data, dict):
        return parents, parent_tid, rank, cls

    nodes = data.get("taxonomy_nodes") or []
    if not nodes or not isinstance(nodes[0], dict):
        return parents, parent_tid, rank, cls

    tax = nodes[0].get("taxonomy") or {}
    rank = (tax.get("rank") or "").replace("_", " ").lower()

    p = tax.get("parents")
    if isinstance(p, list):
        parents = _ints_from_any(p)

    ptid = tax.get("parent_tax_id")
    if isinstance(ptid, int):
        parent_tid = ptid
    elif isinstance(ptid, str) and ptid.isdigit():
        parent_tid = int(ptid)

    # collect classification ids for usual ranks (keep order nearest->root later)
    order = ["species", "genus", "family", "order", "class", "phylum", "kingdom", "domain", "superkingdom"]
    cdict = tax.get("classification") or {}
    for r in order:
        node = cdict.get(r)
        if isinstance(node, dict):
            tid = node.get("id")
            if isinstance(tid, str) and tid.isdigit():
                tid = int(tid)
            if isinstance(tid, int):
                cls.append((r, tid))

    return parents, parent_tid, rank, cls


def _parents_nearest_first(parents: List[int], parent_tid: Optional[int]) -> List[int]:
    """Orient parents so immediate parent comes first."""
    if not parents:
        return parents[:]
    if parent_tid and parent_tid in parents:
        i = parents.index(parent_tid)
        return parents[i:] + parents[:i]
    if parents and parents[0] in (1, 131567):  # root-ish first -> reverse
        return list(reversed(parents))
    return parents[:]


def _classification_nearest_first(rank: str, cls: List[Tuple[str, int]]) -> List[int]:
    """Nearest-first classification-derived parent list (keep 'species' for below-species ranks)."""
    ids_by_rank: Dict[str, int] = {r: tid for (r, tid) in cls}
    out: List[int] = []
    below_species = any(k in (rank or "") for k in ("subspecies", "strain", "isolate", "biovar", "serovar", "pathovar"))
    order = ["species", "genus", "family", "order", "class", "phylum", "kingdom", "domain", "superkingdom"]
    for r in order:
        tid = ids_by_rank.get(r)
        if tid is None:
            continue
        if (rank == "species") and (r == "species"):
            continue  # species is self
        if (not below_species) and (r == "species"):
            continue  # species would be self/peer when at or above species
        if tid not in out:
            out.append(tid)
    return out


def _fetch_lineage_for_taxid(taxid: int, timeout: float, retries: int, sleep: float) -> dict:
    """
    Returns {
        "parent_tid": int | None,
        "parents_nf": [immediate_parent, ..., root],
        "cls_nf":     [nearest_from_classification, ...],
        "rank":       str | "",
        "anchors":    [superkingdom ids present in classification]
    }
    """
    cached = _lin_cache_get(taxid)
    if cached:
        return cached

    parent_tid: Optional[int] = None
    parents_nf: List[int] = []
    cls_nf: List[int] = []
    rank: str = ""
    anchors: List[int] = []

    # NCBI v2 -> v2alpha
    for base in (_NCBI_V2_TMPL, _NCBI_V2A_TMPL):
        url = _params_with_api_key(base.format(taxid=taxid))
        data = _api_get_json(url, timeout, retries, sleep)
        if not data:
            continue
        parents, ptid, r, cls = _parse_ncbi_node(data)
        rank = r or rank
        parent_tid = ptid if ptid is not None else parent_tid

        if parents and not parents_nf:
            parents_nf = _parents_nearest_first(parents, parent_tid)
        if cls and not cls_nf:
            cls_nf = _classification_nearest_first(rank, cls)
            for k, tid in cls:
                if k in ("domain", "superkingdom") and isinstance(tid, int):
                    anchors.append(tid)

        if parents_nf or cls_nf or parent_tid:
            break

    # Ensembl fallback if still nothing
    if not parents_nf and not cls_nf and parent_tid is None:
        data = _api_get_json(_ENSEMBL_CLASSIF.format(taxid=taxid), timeout, retries, sleep,
                             headers={"User-Agent": "AGR-species-extraction/1.2", "Accept": "application/json"})
        if isinstance(data, list) and data:
            path = data[0] if (isinstance(data[0], list)) else data
            ids: List[int] = []
            for node in path:
                tid = node.get("id") if isinstance(node, dict) else None
                if isinstance(tid, str) and tid.isdigit():
                    tid = int(tid)
                if isinstance(tid, int):
                    ids.append(tid)
            if len(ids) >= 2:
                parents_nf = ids[1:]  # immediate-first

    payload = {
        "parent_tid": parent_tid,
        "parents_nf": parents_nf,
        "cls_nf": cls_nf,
        "rank": rank or "",
        "anchors": list(dict.fromkeys(anchors))
    }
    _lin_cache_put(taxid, payload)
    return payload


def _choose_nearest_curated_ancestor(child: int, curated: Set[int], payload: dict) -> Optional[int]:
    """Pick the nearest curated ancestor from parent_tid, parents_nf, cls_nf, then anchors."""
    cand: List[int] = []
    ptid = payload.get("parent_tid")
    if isinstance(ptid, int):
        cand.append(ptid)
    cand.extend(payload.get("parents_nf") or [])
    cand.extend(payload.get("cls_nf") or [])
    for a in cand:
        if a != child and a in curated:
            return a
    for a in payload.get("anchors") or []:
        if a in curated:
            return a
    return None


def _fallback_default() -> Dict[str, Set[str]]:
    # NOTE: DEFAULT_PARENT_CHILDREN is defined at the bottom; it's looked up at call time.
    logger.info("Using baked default parent/children map (parents=%d).", len(DEFAULT_PARENT_CHILDREN))
    return {p: set(kids) for p, kids in (DEFAULT_PARENT_CHILDREN or {}).items()}


def _cache_path(cache_dir: Optional[str], cache_key: Optional[str]) -> Optional[Path]:
    if not cache_dir or not cache_key:
        return None
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"parent_children_{cache_key}.json"


def _load_cache(path: Path) -> Optional[Dict[str, Set[str]]]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: set(v) for k, v in data.items() if isinstance(v, list)}
    except Exception:
        logger.warning("Failed to read parent/children cache at %s; ignoring.", path, exc_info=True)
        return None


def _write_cache(path: Path, mapping: Dict[str, Set[str]]) -> None:
    try:
        tmp = path.with_suffix(".tmp")
        serializable = {k: sorted(list(v)) for k, v in mapping.items()}
        tmp.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.warning("Failed to write parent/children cache to %s; continuing.", path, exc_info=True)


def _merge_parent_children(
    api_map: Dict[str, Set[str]],
    default_map: Dict[str, List[str]],
    curated_curies: Optional[Set[str]] = None
) -> Dict[str, Set[str]]:
    """
    Merge API-derived and baked mappings:
    - parent set = union of keys
    - children set = union of children
    - optionally filter parents/children to curated_curies
    """
    merged: Dict[str, Set[str]] = {}
    keys = set(api_map) | set(default_map)
    for p in keys:
        kids = set()
        kids |= api_map.get(p, set())
        kids |= set(default_map.get(p, []))
        if curated_curies is not None:
            if p not in curated_curies:
                # keep parent only if at least one child is curated
                kids = {k for k in kids if k in curated_curies}
                if not kids:
                    continue  # drop empty parent
            else:
                kids = {k for k in kids if k in curated_curies}
        if kids:
            merged[p] = kids
    return merged


def get_parent_children_map(
    species_name_to_curie: Dict[str, str],
    api_timeout: float = 15.0,
    api_retries: int = 3,
    api_sleep: float = 0.3,
    *,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    force_refresh: bool = False,
    merge_with_default: bool = True,  # <<— merge online (e.g., 9 parents) with baked (e.g., 33)
) -> Dict[str, Set[str]]:
    """
    Build a parent→children map restricted to your curated set.

    Steps:
    1) If cache available and not force_refresh, return it.
    2) Build online map by linking each curated node to its nearest curated ancestor.
    3) If merge_with_default=True, union with the baked DEFAULT map (unique parent list).
    4) Write the (possibly merged) map to cache and return it.
    5) If the online map is empty (e.g., network issues), return the baked DEFAULT.

    Returns: Dict[str, Set[str]] with keys/values like "NCBITaxon:562".
    """
    try:
        # Disk cache first
        cpath = _cache_path(cache_dir, cache_key)
        if cpath and not force_refresh:
            cached = _load_cache(cpath)
            if cached:
                return cached

        # Curated set
        curated_taxids: Set[int] = set()
        for curie in species_name_to_curie.values():
            tid = _extract_taxid(curie)
            if tid is not None:
                curated_taxids.add(tid)
        if not curated_taxids:
            logger.info("No curated species taxids; using fallback default.")
            fallback = _fallback_default()
            if cpath:
                _write_cache(cpath, fallback)
            return fallback
        curated_curies: Set[str] = {f"NCBITaxon:{tid}" for tid in curated_taxids}

        # Fetch lineages in parallel -> build api_map
        api_map: Dict[str, Set[str]] = {}

        def work(tid: int) -> Tuple[int, dict]:
            return tid, _fetch_lineage_for_taxid(tid, api_timeout, api_retries, api_sleep)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(work, tid): tid for tid in curated_taxids}
            for fut in as_completed(futures):
                child, payload = fut.result()
                anc = _choose_nearest_curated_ancestor(child, curated_taxids, payload)
                if anc is None or anc == child:
                    continue
                p_curie = f"NCBITaxon:{anc}"
                c_curie = f"NCBITaxon:{child}"
                api_map.setdefault(p_curie, set()).add(c_curie)

        if not api_map:
            logger.info("API returned empty/none parent/child relationships; falling back to baked default.")
            fallback = _fallback_default()
            if cpath:
                _write_cache(cpath, fallback)
            return fallback

        # Merge with default if requested
        if merge_with_default:
            merged = _merge_parent_children(api_map, DEFAULT_PARENT_CHILDREN, curated_curies)
            if cpath:
                _write_cache(cpath, merged)
            logger.info("Merged online (parents=%d) with baked default (parents=%d) -> merged parents=%d.",
                        len(api_map), len(DEFAULT_PARENT_CHILDREN), len(merged))
            return merged

        # No merge requested: filter api_map to curated (already is), write & return
        if cpath:
            _write_cache(cpath, api_map)
        logger.info("Built parent/children via online taxonomy (parents=%d, edges=%d).",
                    len(api_map), sum(len(v) for v in api_map.values()))
        return api_map

    except Exception as e:
        logger.info("Error building parent/child map; using baked default: %s", e)
        fallback = _fallback_default()
        cpath = _cache_path(cache_dir, cache_key)
        if cpath:
            _write_cache(cpath, fallback)
        return fallback


# flake8: noqa
# ======================================================================
# BAKED DEFAULT PARENT→CHILDREN MAPPING (sorted numerically by taxid)
# ======================================================================
DEFAULT_PARENT_CHILDREN: Dict[str, List[str]] = {
  "NCBITaxon:287": ["NCBITaxon:652611", "NCBITaxon:1009714"],
  "NCBITaxon:470": ["NCBITaxon:400667"],
  "NCBITaxon:562": ["NCBITaxon:83333", "NCBITaxon:83334", "NCBITaxon:511693", "NCBITaxon:591946", "NCBITaxon:668369"],
  "NCBITaxon:5691": ["NCBITaxon:5702"],
  "NCBITaxon:629": ["NCBITaxon:630", "NCBITaxon:29486"],
  "NCBITaxon:6339": ["NCBITaxon:375939"],
  "NCBITaxon:642": ["NCBITaxon:644", "NCBITaxon:645", "NCBITaxon:648", "NCBITaxon:654", "NCBITaxon:196024"],
  "NCBITaxon:65964": ["NCBITaxon:65965", "NCBITaxon:296643"],
  "NCBITaxon:662": ["NCBITaxon:666", "NCBITaxon:672", "NCBITaxon:55601", "NCBITaxon:2025808"],
  "NCBITaxon:694009": ["NCBITaxon:2697049"],
  "NCBITaxon:7230": ["NCBITaxon:377587", "NCBITaxon:943925", "NCBITaxon:943926"],
  "NCBITaxon:7742": ["NCBITaxon:7757", "NCBITaxon:7788", "NCBITaxon:7812", "NCBITaxon:7868", "NCBITaxon:7918", "NCBITaxon:7955", "NCBITaxon:8005", "NCBITaxon:8010", "NCBITaxon:8022", "NCBITaxon:8030", "NCBITaxon:8090", "NCBITaxon:8094", "NCBITaxon:8145", "NCBITaxon:8204", "NCBITaxon:8265", "NCBITaxon:8316", "NCBITaxon:8346", "NCBITaxon:8355", "NCBITaxon:8364", "NCBITaxon:8404", "NCBITaxon:8616", "NCBITaxon:9031", "NCBITaxon:90988", "NCBITaxon:105023", "NCBITaxon:262014", "NCBITaxon:27699", "NCBITaxon:30732", "NCBITaxon:31033", "NCBITaxon:34780", "NCBITaxon:37610", "NCBITaxon:40674", "NCBITaxon:52644", "NCBITaxon:75329"],
  "NCBITaxon:81077": ["NCBITaxon:32630"],
  "NCBITaxon:88886": ["NCBITaxon:89887", "NCBITaxon:89888", "NCBITaxon:89890"],
  "NCBITaxon:90371": ["NCBITaxon:85569"],
  "NCBITaxon:10710": ["NCBITaxon:2681611"],
  "NCBITaxon:11270": ["NCBITaxon:11276", "NCBITaxon:11287", "NCBITaxon:11288", "NCBITaxon:11290", "NCBITaxon:103603", "NCBITaxon:696863", "NCBITaxon:1985699"],
  "NCBITaxon:1166": ["NCBITaxon:315271"],
  "NCBITaxon:129951": ["NCBITaxon:28285"],
  "NCBITaxon:1313": ["NCBITaxon:170187"],
  "NCBITaxon:133979": ["NCBITaxon:133980", "NCBITaxon:133983"],
  "NCBITaxon:1639": ["NCBITaxon:169963", "NCBITaxon:393133"],
  "NCBITaxon:1781": ["NCBITaxon:216594", "NCBITaxon:1131442"],
  "NCBITaxon:1827": ["NCBITaxon:1833"],
  "NCBITaxon:252598": ["NCBITaxon:285006", "NCBITaxon:307796", "NCBITaxon:462210", "NCBITaxon:464025", "NCBITaxon:471859", "NCBITaxon:538975", "NCBITaxon:538976", "NCBITaxon:545124", "NCBITaxon:559292", "NCBITaxon:574961", "NCBITaxon:580239", "NCBITaxon:580240", "NCBITaxon:643680", "NCBITaxon:658763", "NCBITaxon:721032", "NCBITaxon:764097", "NCBITaxon:764098", "NCBITaxon:764099", "NCBITaxon:764100", "NCBITaxon:889517", "NCBITaxon:929585", "NCBITaxon:929586", "NCBITaxon:929587", "NCBITaxon:929629", "NCBITaxon:947035", "NCBITaxon:947036", "NCBITaxon:947039", "NCBITaxon:947040", "NCBITaxon:1095001", "NCBITaxon:1227742", "NCBITaxon:1247190", "NCBITaxon:1337438", "NCBITaxon:1337529"],
  "NCBITaxon:30017": ["NCBITaxon:133982", "NCBITaxon:133998", "NCBITaxon:201951"],
  "NCBITaxon:30036": ["NCBITaxon:195057", "NCBITaxon:296644"],
  "NCBITaxon:326968": ["NCBITaxon:714518"],
  "NCBITaxon:36809": ["NCBITaxon:1185650", "NCBITaxon:1962118"],
  "NCBITaxon:40366": ["NCBITaxon:40372", "NCBITaxon:95109"],
  "NCBITaxon:45357": ["NCBITaxon:1231523"],
  "NCBITaxon:46703": ["NCBITaxon:3061081"],
  "NCBITaxon:5476": ["NCBITaxon:237561"],
  "NCBITaxon:9823": ["NCBITaxon:9825"],
  "NCBITaxon:10239": ["NCBITaxon:10243", "NCBITaxon:10245", "NCBITaxon:10298", "NCBITaxon:10359", "NCBITaxon:10450", "NCBITaxon:10456", "NCBITaxon:10484", "NCBITaxon:10665", "NCBITaxon:10678", "NCBITaxon:10703", "NCBITaxon:10710", "NCBITaxon:10719", "NCBITaxon:10760", "NCBITaxon:11033", "NCBITaxon:11034", "NCBITaxon:11103", "NCBITaxon:11270", "NCBITaxon:11309", "NCBITaxon:113370", "NCBITaxon:11676", "NCBITaxon:11788", "NCBITaxon:11870", "NCBITaxon:11908", "NCBITaxon:11909", "NCBITaxon:11913", "NCBITaxon:11988", "NCBITaxon:12022", "NCBITaxon:12136", "NCBITaxon:12145", "NCBITaxon:12161", "NCBITaxon:12183", "NCBITaxon:12227", "NCBITaxon:12232", "NCBITaxon:12287", "NCBITaxon:12288", "NCBITaxon:12470", "NCBITaxon:12637", "NCBITaxon:12721", "NCBITaxon:128987", "NCBITaxon:129951", "NCBITaxon:148603", "NCBITaxon:1891767", "NCBITaxon:191766", "NCBITaxon:2034342", "NCBITaxon:2034344", "NCBITaxon:2034346", "NCBITaxon:2034347", "NCBITaxon:271108", "NCBITaxon:28285", "NCBITaxon:28355", "NCBITaxon:333760", "NCBITaxon:333761", "NCBITaxon:35345", "NCBITaxon:37124", "NCBITaxon:39006", "NCBITaxon:45455", "NCBITaxon:45617", "NCBITaxon:46015", "NCBITaxon:64279", "NCBITaxon:64320", "NCBITaxon:694009", "NCBITaxon:91753", "NCBITaxon:92652", "NCBITaxon:977912", "NCBITaxon:977913", "NCBITaxon:1241918", "NCBITaxon:2907964", "NCBITaxon:2933356"]
}

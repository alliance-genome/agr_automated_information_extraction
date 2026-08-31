"""Turn ZFIN's molecular-probe label sheet into an abstract-only training set
(SCRUM-5764).

The sheet (``AGRKBID``, ``XREF``, ``Classification``) is the format
``agr_dataset_manager/dataset_upload_from_csv.py`` already expects, but its label
values are ``"Positive "`` / ``"Negative"`` while the trainer compares
``classification_value == "positive"`` exactly. This script normalizes the labels,
drops duplicate xrefs, resolves each xref to an AGRKB curie, fetches title and
abstract, and reports how many references are actually usable — references with no
abstract in the ABC cannot be classified from an abstract at all, which caps
achievable recall and is the number ZFIN needs to size manual triage.

Usage:
    python3 scripts/prepare_zfin_probe_training_set.py -f probe_labels.csv -o outdir
"""
import argparse
import csv
import json
import logging
import os
import sys

from utils.abc_utils import get_curie_from_xref, get_reference_title_and_abstract

logger = logging.getLogger(__name__)


# The three columns the sheet names. Anything else on a row is a free-text
# curator annotation — in practice Ceri's 'string "probe" in abstract' marker on
# the hard negatives, which the evaluation reports on separately. The column has
# an empty header in the export, so it is collected by exclusion rather than name.
_KNOWN_COLUMNS = {"AGRKBID", "XREF", "Classification"}


def _note_from(row: dict) -> str:
    """Return the row's curator annotation, joining any unnamed extra columns.

    A row with more fields than the header -- one extra comma, which is how an
    annotation typed past the last column arrives -- lands under
    ``csv.DictReader``'s restkey (``None``) as a *list* rather than a string, so
    both shapes have to be handled or that annotation is silently lost.
    """
    extras = []
    for key, value in row.items():
        if key in _KNOWN_COLUMNS:
            continue
        values = value if isinstance(value, list) else [value]
        extras.extend((part or "").strip() for part in values)
    return " ".join(part for part in extras if part)


def _prefixed(xref: str, mod_prefix: str) -> str:
    """Return ``xref`` with the MOD prefix the ABC's ``by_cross_reference``
    endpoint requires.

    The endpoint matches on the full curie (``ZFIN:ZDB-PUB-170126-4``); a bare
    ``ZDB-PUB-170126-4`` returns 404 for *every* reference, which reads as "none
    of these papers are in the ABC" rather than as a formatting error. Ceri's
    sheet ships bare ids, so this prefixing is what makes resolution work at all.
    An xref that already carries a prefix is passed through untouched.
    """
    return xref if ":" in xref else f"{mod_prefix}:{xref}"


def prepare(csv_file: str, output_dir: str, mod_prefix: str = "ZFIN") -> dict:
    """Write ``labelled_abstracts.json`` + ``coverage_report.txt`` into
    ``output_dir`` and return the coverage counters."""
    os.makedirs(output_dir, exist_ok=True)
    report = {"rows_read": 0, "unparseable_labels": 0, "duplicates_dropped": 0,
              "unresolved_xrefs": 0, "lookup_errors": 0, "missing_abstracts": 0,
              "usable": 0}
    records = []
    seen_xrefs = set()

    with open(csv_file, newline="") as handle:
        for row in csv.DictReader(handle):
            report["rows_read"] += 1
            xref = (row.get("XREF") or "").strip()
            label = (row.get("Classification") or "").strip().lower()
            if not xref or label not in ("positive", "negative"):
                report["unparseable_labels"] += 1
                continue
            if xref in seen_xrefs:
                report["duplicates_dropped"] += 1
                continue
            seen_xrefs.add(xref)

            curie = ((row.get("AGRKBID") or "").strip()
                     or get_curie_from_xref(_prefixed(xref, mod_prefix)))
            if not curie:
                report["unresolved_xrefs"] += 1
                logger.warning("No AGRKB curie for xref %s", xref)
                continue

            # get_reference_title_and_abstract raises on a non-200 rather than
            # returning None; one unreachable reference must not lose the rest.
            try:
                title, abstract = get_reference_title_and_abstract(curie)
            except Exception as exc:  # noqa: BLE001 - any transport/HTTP failure
                report["lookup_errors"] += 1
                logger.warning("Lookup failed for %s (%s): %s", curie, xref, exc)
                continue

            if not (abstract or "").strip():
                report["missing_abstracts"] += 1
                logger.warning("No abstract for %s (%s)", curie, xref)
                continue

            records.append({"curie": curie, "xref": xref, "label": label,
                            "title": title or "", "abstract": abstract,
                            "note": _note_from(row)})
            report["usable"] += 1

    report["positives"] = sum(1 for r in records if r["label"] == "positive")
    report["negatives"] = sum(1 for r in records if r["label"] == "negative")
    # Ceri's hard negatives: the abstract contains the word "probe" but the paper
    # is still curatable. These are the cases a keyword rule gets wrong, so the
    # evaluation reports on them separately.
    report["annotated_hard_negatives"] = sum(1 for r in records if r["note"])

    with open(os.path.join(output_dir, "labelled_abstracts.json"), "w") as handle:
        json.dump(records, handle, indent=2)

    # A CSV ready for agr_dataset_manager/dataset_upload_from_csv.py. Two things
    # make it safe to feed straight in:
    #   * AGRKBID is already resolved, so the uploader never reaches its
    #     get_curie_from_xref fallback -- the one that 404s on a bare ZDB-PUB id
    #     and then silently skips the row.
    #   * Classification is normalized to lowercase, because add_entry_to_dataset
    #     stores the value verbatim and the trainer compares == "positive".
    # XREF keeps the MOD prefix as belt and braces for the fallback path.
    upload_path = os.path.join(output_dir, "upload_ready.csv")
    with open(upload_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AGRKBID", "XREF", "Classification"])
        writer.writeheader()
        for record in records:
            writer.writerow({"AGRKBID": record["curie"],
                             "XREF": _prefixed(record["xref"], mod_prefix),
                             "Classification": record["label"]})
    lines = [f"{key}: {value}" for key, value in report.items()]
    with open(os.path.join(output_dir, "coverage_report.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("Coverage: %s", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--csv-file", required=True, help="Path to the exported label sheet")
    parser.add_argument("-o", "--output-dir", required=True, help="Where to write the outputs")
    parser.add_argument("-m", "--mod-prefix", default="ZFIN",
                        help="MOD prefix prepended to bare xrefs before resolution "
                             "(the by_cross_reference endpoint 404s without it)")
    parser.add_argument("-l", "--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, stream=sys.stdout,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    prepare(args.csv_file, args.output_dir, mod_prefix=args.mod_prefix)


if __name__ == "__main__":
    main()

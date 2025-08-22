import sys
from typing import Dict, Set, List, Tuple


def load_baseline_file(filepath: str) -> Dict[str, Dict[str, Set[str]]]:
    entity_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            agrkb_id, mod_id, entity_str = parts
            entities = set(e.strip() for e in entity_str.split('|') if e.strip())
            entity_data[agrkb_id] = {"mod_id": mod_id, "entities": entities}
    return entity_data


def load_abc_file(filepath: str) -> Dict[str, Set[str]]:
    entity_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            agrkb_id, entity_str = parts
            entities = set(e.strip() for e in entity_str.split('|') if e.strip())
            entity_data[agrkb_id] = entities
    return entity_data


def compare_entities(
    abc_data: Dict[str, Dict[str, Set[str]]],
    baseline_data: Dict[str, Set[str]],
    entity_type: str
) -> Dict[str, float]:
    shared_ids = sorted(set(abc_data) & set(baseline_data))
    all_common = all_only_abc = all_only_baseline = 0
    match_rows: List[Tuple[str, str, List[str]]] = []
    diff_rows: List[Tuple[str, str, List[str], List[str], List[str], int]] = []

    for agrkb_id in shared_ids:
        mod_id = baseline_data[agrkb_id]["mod_id"]
        abc_entities = abc_data[agrkb_id]
        baseline_entities = baseline_data[agrkb_id]["entities"]

        intersection = abc_entities & baseline_entities
        only_abc = abc_entities - baseline_entities
        only_baseline = baseline_entities - abc_entities

        all_common += len(intersection)
        all_only_abc += len(only_abc)
        all_only_baseline += len(only_baseline)

        if abc_entities == baseline_entities:
            match_rows.append((agrkb_id, mod_id, sorted(abc_entities)))
        else:
            diff_count = len(only_abc) + len(only_baseline)
            # sorted(intersection),
            diff_rows.append((
                agrkb_id,
                mod_id,
                str(len(intersection)),
                sorted(only_abc),
                sorted(only_baseline),
                diff_count
            ))

    # Write exact matches
    match_file = f"{entity_type}_matches.tsv"
    with open(match_file, "w") as f:
        for agrkb_id, mod_id, entities in match_rows:
            f.write(f"{agrkb_id}\t{mod_id}\t{entities}\n")

    # Write mismatches sorted by greatest number of differences
    diff_rows.sort(key=lambda x: x[-1], reverse=True)
    diff_file = f"{entity_type}_differences.tsv"
    with open(diff_file, "w") as f:
        for agrkb_id, mod_id, common, only_abc, only_baseline, _ in diff_rows:
            f.write(f"{agrkb_id}\t{mod_id}\t{common}\t{only_abc}\t{only_baseline}\n")

    total = all_common + all_only_abc + all_only_baseline
    stats = {
        "Total Entities in Common": all_common,
        "Percent in Common": 100 * all_common / total if total else 0.0,
        "Total Only in ABC": all_only_abc,
        "Percent Only in ABC": 100 * all_only_abc / total if total else 0.0,
        "Total Only in Baseline": all_only_baseline,
        "Percent Only in Baseline": 100 * all_only_baseline / total if total else 0.0,
        "Total Compared Papers": len(shared_ids),
        "Exact Match Count": len(match_rows),
        "Exact Match Rate": len(match_rows) / len(shared_ids) if shared_ids else 0.0,
    }
    return stats


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_species_extractions.py <entity_type> abc_file baseline_file")
        sys.exit(1)

    entity_type = sys.argv[1]
    abc_file = sys.argv[2]
    baseline_file = sys.argv[3]

    abc_data = load_abc_file(abc_file)
    baseline_data = load_baseline_file(baseline_file)

    stats = compare_entities(abc_data, baseline_data, entity_type)

    print("\n--- Statistics ---")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()

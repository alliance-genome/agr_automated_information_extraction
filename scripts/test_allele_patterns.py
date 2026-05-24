#!/usr/bin/env python3
"""
Standalone test for allele extraction patterns.
Tests the pattern matching and false positive filtering logic without Docker dependencies.

Usage:
    python3 scripts/test_allele_patterns.py
"""
import os
import sys
import json
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only the patterns and functions that don't have heavy dependencies
from utils.entity_extraction_utils import (
    ALLELE_NAME_PATTERN,
    FIGURE_PANEL_CONTEXT_RE,
    HISTONE_PROTEIN_RE,
    MEASUREMENT_RE,
    REAGENT_NAMES_LOWER,
    SUSPICIOUS_PREFIX_RE,
)


def test_allele_pattern():
    """Test ALLELE_NAME_PATTERN matches expected alleles."""
    print("\n=== Testing ALLELE_NAME_PATTERN ===")

    # Should match (lowercase start, mixed case allowed, must have letter+digit)
    should_match = [
        'e1370', 'ok1255', 'tm1949', 'n324', 'mu86',  # classic alleles
        'ttTi4348', 'ttTi5605',  # MosSCI sites (mixed case)
        'oxTi970', 'oxTi185',    # other Ti alleles
        'mgDf50', 'gkDf31',      # deficiencies
        'ieSi64', 'juSi123',     # single-copy insertions
        'wk30', 'wk7', 'wk70',   # short alleles
        'ad465', 'km21', 'p675', # more classics
    ]

    # Should NOT match
    should_not_match = [
        'GFP', 'RFP', 'YFP',     # fluorescent proteins (uppercase)
        'N2', 'CB4856',          # strain names (uppercase start)
        'H2A', 'H3',             # histones (uppercase)
        'Figure1A', 'PanelB',    # figure labels
        'BRCA1', 'TP53',         # human genes
        '123abc',                # starts with digit
    ]

    passed = 0
    failed = 0

    for allele in should_match:
        text = f"The {allele} allele was studied."
        matches = ALLELE_NAME_PATTERN.findall(text)
        if allele in matches:
            print(f"  ✓ Matched: {allele}")
            passed += 1
        else:
            print(f"  ✗ FAILED to match: {allele} (got {matches})")
            failed += 1

    for non_allele in should_not_match:
        text = f"The {non_allele} was used."
        matches = ALLELE_NAME_PATTERN.findall(text)
        if non_allele not in matches:
            print(f"  ✓ Correctly rejected: {non_allele}")
            passed += 1
        else:
            print(f"  ✗ INCORRECTLY matched: {non_allele}")
            failed += 1

    print(f"\nPattern tests: {passed} passed, {failed} failed")
    return failed == 0


def test_false_positive_patterns():
    """Test the new false positive detection patterns."""
    print("\n=== Testing False Positive Patterns ===")

    passed = 0
    failed = 0

    # Test HISTONE_PROTEIN_RE
    print("\n  Histone pattern tests:")
    histone_matches = ['h1', 'h2', 'H3', 'H4', 'H2A', 'H2B', 'h3x']
    histone_non_matches = ['h5', 'h12', 'ha1', 'hb2', 'his-1']

    for h in histone_matches:
        if HISTONE_PROTEIN_RE.match(h):
            print(f"    ✓ Matched histone: {h}")
            passed += 1
        else:
            print(f"    ✗ FAILED to match histone: {h}")
            failed += 1

    for h in histone_non_matches:
        if not HISTONE_PROTEIN_RE.match(h):
            print(f"    ✓ Correctly rejected: {h}")
            passed += 1
        else:
            print(f"    ✗ INCORRECTLY matched: {h}")
            failed += 1

    # Test MEASUREMENT_RE
    print("\n  Measurement pattern tests:")
    measurement_matches = ['t1', 't2', 'd1', 'd10']
    measurement_non_matches = ['tm1', 'ok1', 'e1370', 'abc']

    for m in measurement_matches:
        if MEASUREMENT_RE.match(m):
            print(f"    ✓ Matched measurement: {m}")
            passed += 1
        else:
            print(f"    ✗ FAILED to match measurement: {m}")
            failed += 1

    for m in measurement_non_matches:
        if not MEASUREMENT_RE.match(m):
            print(f"    ✓ Correctly rejected: {m}")
            passed += 1
        else:
            print(f"    ✗ INCORRECTLY matched: {m}")
            failed += 1

    # Test REAGENT_NAMES_LOWER
    print("\n  Reagent name tests:")
    reagent_matches = ['gfp', 'rfp', 'mcherry', 'yfp', 'tdtomato']
    reagent_non_matches = ['e1370', 'ok1255', 'tm1949']

    for r in reagent_matches:
        if r in REAGENT_NAMES_LOWER:
            print(f"    ✓ Matched reagent: {r}")
            passed += 1
        else:
            print(f"    ✗ FAILED to match reagent: {r}")
            failed += 1

    for r in reagent_non_matches:
        if r not in REAGENT_NAMES_LOWER:
            print(f"    ✓ Correctly rejected: {r}")
            passed += 1
        else:
            print(f"    ✗ INCORRECTLY matched: {r}")
            failed += 1

    # Test FIGURE_PANEL_CONTEXT_RE
    print("\n  Figure/panel context tests:")
    figure_texts = [
        ("Figure 1A shows the results", ['A']),
        ("Fig. 2B demonstrates", ['B']),
        ("Panel C shows", ['C']),
        ("as shown in Figure 3D", ['D']),
        ("see Fig 4E2", ['E2']),
    ]

    for text, expected in figure_texts:
        matches = FIGURE_PANEL_CONTEXT_RE.findall(text)
        if matches == expected:
            print(f"    ✓ Figure context: '{text}' -> {matches}")
            passed += 1
        else:
            print(f"    ✗ FAILED: '{text}' expected {expected}, got {matches}")
            failed += 1

    print(f"\nFalse positive pattern tests: {passed} passed, {failed} failed")
    return failed == 0


def test_with_sample_md_files():
    """Test with actual MD files if available."""
    print("\n=== Testing with Sample MD Files ===")

    md_dir = os.path.expanduser('~/claude_code/abc/wb_papers_subset_md/')
    allele_json = os.path.expanduser('~/claude_code/abc/cache/WB_allele.json')

    if not os.path.isdir(md_dir):
        print(f"  MD directory not found: {md_dir}")
        print("  Skipping MD file tests.")
        return True

    if not os.path.exists(allele_json):
        print(f"  Allele JSON not found: {allele_json}")
        print("  Skipping MD file tests.")
        return True

    # Load allele set
    with open(allele_json) as f:
        allele_data = json.load(f)
    if isinstance(allele_data, dict) and 'names' in allele_data:
        names = allele_data['names']
    else:
        names = allele_data
    allele_set = {n.lower() for n in names if isinstance(n, str)}
    print(f"  Loaded {len(allele_set)} curated alleles")

    # Process a few sample files
    md_files = [f for f in os.listdir(md_dir) if f.endswith('.md')][:10]
    print(f"  Processing {len(md_files)} sample MD files...")

    total_matches = 0
    total_curated = 0
    total_rejected_fp = 0

    for fname in md_files:
        path = os.path.join(md_dir, fname)
        with open(path, 'r') as f:
            content = f.read()

        # Find all pattern matches
        matches = set(ALLELE_NAME_PATTERN.findall(content))
        total_matches += len(matches)

        # Filter to curated
        curated = {m for m in matches if m.lower() in allele_set}
        total_curated += len(curated)

        # Check for false positives using new patterns
        rejected = []
        for m in curated:
            m_lower = m.lower()
            if HISTONE_PROTEIN_RE.match(m):
                rejected.append((m, 'histone'))
            elif MEASUREMENT_RE.match(m):
                rejected.append((m, 'measurement'))
            elif m_lower in REAGENT_NAMES_LOWER:
                rejected.append((m, 'reagent'))

        total_rejected_fp += len(rejected)

        if curated:
            curie = fname.replace('.md', '').replace('_', ':')
            print(f"\n  {curie}:")
            print(f"    Pattern matches: {len(matches)}")
            print(f"    Curated matches: {sorted(curated)[:10]}")
            if rejected:
                print(f"    Would reject: {rejected}")

    print(f"\n  Summary:")
    print(f"    Total pattern matches: {total_matches}")
    print(f"    Total curated matches: {total_curated}")
    print(f"    Would reject as FP: {total_rejected_fp}")

    return True


def main():
    print("=" * 60)
    print("ALLELE EXTRACTION PATTERN TESTS")
    print("=" * 60)

    all_passed = True

    if not test_allele_pattern():
        all_passed = False

    if not test_false_positive_patterns():
        all_passed = False

    if not test_with_sample_md_files():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

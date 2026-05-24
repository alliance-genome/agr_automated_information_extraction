#!/usr/bin/env python3
"""
Validate allele extraction against sample papers.

This script samples papers from a directory of MD files, extracts alleles using
the current extraction logic, and reports on what was extracted vs rejected.

Usage:
    python scripts/validate_allele_extraction.py --sample 50
    python scripts/validate_allele_extraction.py --md-dir /path/to/md/files --sample 100
    python scripts/validate_allele_extraction.py --all  # Process all papers
"""
import argparse
import json
import logging
import os
import random
import sys

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.md_utils import AllianceMarkdown  # noqa: E402
from utils.entity_extraction_utils import (  # noqa: E402
    ALLELE_NAME_PATTERN,
    filter_false_positive_alleles,
    has_allele_like_context,
    SUSPICIOUS_PREFIX_RE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def sample_papers(md_dir: str, n: int) -> list[str]:
    """Sample N random papers from the MD directory."""
    files = [f for f in os.listdir(md_dir) if f.endswith('.md')]
    if n >= len(files):
        return files
    return random.sample(files, n)


def extract_alleles_from_paper(md_path: str, allele_set: set) -> dict:
    """
    Extract alleles from a single paper and return analysis.

    Args:
        md_path: Path to the markdown file
        allele_set: Set of known allele names (lowercase)

    Returns:
        Dictionary with extraction results and analysis
    """
    md = AllianceMarkdown()
    md.load_from_file(md_path)
    fulltext = md.get_fulltext(include_attributes=True) or ""
    title = md.get_title() or ""

    # Find all pattern matches in fulltext
    all_matches = set(ALLELE_NAME_PATTERN.findall(fulltext))

    # Filter to curated alleles only
    curated_matches = {m for m in all_matches if m.lower() in allele_set}

    # Track suspicious short alleles that were rejected due to weak context
    suspicious_rejected = []
    context_passed = []
    for m in curated_matches:
        if SUSPICIOUS_PREFIX_RE.match(m.lower()):
            if has_allele_like_context(fulltext, m):
                context_passed.append(m)
            else:
                suspicious_rejected.append(m)

    # Apply false positive filter to curated matches
    filtered, rejected = filter_false_positive_alleles(list(curated_matches), fulltext)

    return {
        'title': title[:100] + '...' if len(title) > 100 else title,
        'all_pattern_matches': sorted(all_matches),
        'curated_matches': sorted(curated_matches),
        'suspicious_rejected_weak_context': sorted(suspicious_rejected),
        'suspicious_passed_context': sorted(context_passed),
        'filtered_alleles': sorted(filtered),
        'rejected_with_reasons': sorted(rejected, key=lambda x: x[0]),
        'stats': {
            'total_pattern_matches': len(all_matches),
            'curated_matches': len(curated_matches),
            'final_extracted': len(filtered),
            'rejected_false_positives': len(rejected),
        }
    }


def load_allele_set(allele_json_path: str) -> set:
    """Load the set of known allele names from the cache JSON file."""
    path = os.path.expanduser(allele_json_path)
    if not os.path.exists(path):
        logger.error("Allele JSON file not found: %s", path)
        sys.exit(1)

    with open(path) as f:
        allele_data = json.load(f)

    # Handle both formats: {"names": [...]} or just [...]
    if isinstance(allele_data, dict) and 'names' in allele_data:
        names = allele_data['names']
    elif isinstance(allele_data, list):
        names = allele_data
    else:
        logger.error("Unexpected allele JSON format")
        sys.exit(1)

    return {n.lower() for n in names if isinstance(n, str)}


def main():
    parser = argparse.ArgumentParser(
        description='Validate allele extraction against sample papers'
    )
    parser.add_argument(
        '--md-dir',
        default='~/claude_code/abc/wb_papers_subset_md/',
        help='Directory containing MD files to process'
    )
    parser.add_argument(
        '--allele-json',
        default='~/claude_code/abc/cache/WB_allele.json',
        help='Path to JSON file containing curated allele names'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=50,
        help='Number of papers to sample (default: 50)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all papers instead of sampling'
    )
    parser.add_argument(
        '--output',
        default='allele_extraction_validation.json',
        help='Output JSON file for detailed results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible sampling'
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Load allele list
    logger.info("Loading allele list from %s", args.allele_json)
    allele_set = load_allele_set(args.allele_json)
    logger.info("Loaded %d curated alleles", len(allele_set))

    # Get papers to process
    md_dir = os.path.expanduser(args.md_dir)
    if not os.path.isdir(md_dir):
        logger.error("MD directory not found: %s", md_dir)
        sys.exit(1)

    if args.all:
        papers = [f for f in os.listdir(md_dir) if f.endswith('.md')]
        logger.info("Processing all %d papers", len(papers))
    else:
        papers = sample_papers(md_dir, args.sample)
        logger.info("Sampled %d papers", len(papers))

    # Process each paper
    results = {}
    total_extracted = 0
    total_rejected = 0
    total_curated_matches = 0

    for i, fname in enumerate(sorted(papers), 1):
        curie = fname.replace('.md', '').replace('_', ':')
        md_path = os.path.join(md_dir, fname)

        try:
            result = extract_alleles_from_paper(md_path, allele_set)
            results[curie] = result
            total_extracted += result['stats']['final_extracted']
            total_rejected += result['stats']['rejected_false_positives']
            total_curated_matches += result['stats']['curated_matches']

            if i % 100 == 0 or i == len(papers):
                logger.info("Processed %d/%d papers", i, len(papers))

        except Exception as e:
            logger.warning("Failed to process %s: %s", fname, e)
            results[curie] = {'error': str(e)}

    # Save detailed results
    output_path = os.path.expanduser(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Detailed results saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("ALLELE EXTRACTION VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Papers analyzed: {len(results)}")
    print(f"Total curated allele matches: {total_curated_matches}")
    print(f"Total alleles extracted (after filtering): {total_extracted}")
    print(f"Total false positives rejected: {total_rejected}")
    if total_curated_matches > 0:
        precision = total_extracted / total_curated_matches * 100
        print(f"Retention rate: {precision:.1f}%")
    print("=" * 60)

    # Show rejection reasons breakdown
    rejection_reasons = {}
    for curie, result in results.items():
        if 'error' in result:
            continue
        for allele, reason in result.get('rejected_with_reasons', []):
            # Extract reason category
            category = reason.split('(')[0].strip()
            rejection_reasons[category] = rejection_reasons.get(category, 0) + 1

    if rejection_reasons:
        print("\nREJECTION REASONS BREAKDOWN:")
        print("-" * 40)
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Show some examples of extracted alleles
    print("\nSAMPLE EXTRACTIONS (first 5 papers with results):")
    print("-" * 40)
    shown = 0
    for curie, result in results.items():
        if 'error' in result:
            continue
        if result['stats']['final_extracted'] > 0:
            print(f"\n{curie}:")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  Extracted: {', '.join(result['filtered_alleles'][:10])}")
            if len(result['filtered_alleles']) > 10:
                print(f"    ... and {len(result['filtered_alleles']) - 10} more")
            if result['rejected_with_reasons']:
                print(f"  Rejected: {result['rejected_with_reasons'][:3]}")
            shown += 1
            if shown >= 5:
                break


if __name__ == '__main__':
    main()

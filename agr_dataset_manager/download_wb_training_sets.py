#!/usr/bin/env python3
"""
Script to download training sets from WormBase Curation Status Page
and save them in CSV format compatible with dataset_upload_from_csv.py
"""

import argparse
import csv
import logging
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Available data types from WormBase
AVAILABLE_DATATYPES = [
    'antibody', 'catalyticact', 'geneint', 'geneprod', 'genereg',
    'humandisease', 'newmutant', 'otherexpr', 'overexpr', 'rnai',
    'transporter'
]


class WormBaseDownloader:
    """Class to handle downloading training sets from WormBase"""

    def __init__(self, username: str, password: str):
        """Initialize downloader with credentials"""
        self.username = username
        self.password = password
        self.base_url = "https://caltech-curation.textpressolab.com/priv/cgi-bin/curation_status.cgi"
        self.session = requests.Session()
        self.auth = HTTPBasicAuth(username, password)

    def build_url(self, datatype: str, classification: str) -> str:
        """Build URL for specific datatype and classification"""
        method = "allval%20pos" if classification == "positive" else "allval%20neg"
        params = {
            'action': 'listCurationStatisticsPapersPage',
            'select_datatypesource': 'caltech',
            'select_curator': 'two324',
            'listDatatype': datatype,
            'method': method,
            'checkbox_cfp': 'on',
            'checkbox_afp': 'on',
            'checkbox_str': 'on',
            'checkbox_nnc': 'on',
            'checkbox_svm': 'on'
        }
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}?{query_string}"

    def extract_papers(self, html_content: str) -> List[str]:
        """Extract paper IDs from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        papers = []

        # Find all links that contain WBPaper IDs
        pattern = re.compile(r'WBPaper(\d{8})')
        for link in soup.find_all('a', href=True):
            if 'specific_papers=WBPaper' in link['href']:
                match = pattern.search(link['href'])
                if match:
                    paper_id = f"WBPaper{match.group(1)}"
                    if paper_id not in papers:
                        papers.append(paper_id)

        # Also check the textarea content if present
        textarea = soup.find('textarea', {'name': 'specific_papers'})
        if textarea and textarea.text:
            textarea_papers = textarea.text.strip().split()
            for paper in textarea_papers:
                paper = paper.strip()
                if paper.startswith('WBPaper'):
                    if paper not in papers:
                        papers.append(paper)
                elif paper.isdigit() and len(paper) == 8:
                    paper_id = f"WBPaper{paper}"
                    if paper_id not in papers:
                        papers.append(paper_id)

        return papers

    def download_datatype(self, datatype: str) -> Tuple[List[str], List[str]]:
        """Download positive and negative papers for a datatype"""
        positive_papers = []
        negative_papers = []

        # Download positive papers
        try:
            logger.info(f"Downloading positive papers for {datatype}")
            url = self.build_url(datatype, "positive")
            response = self.session.get(url, auth=self.auth, timeout=30)
            response.raise_for_status()
            positive_papers = self.extract_papers(response.text)
            logger.info(f"Found {len(positive_papers)} positive papers for {datatype}")
        except Exception as e:
            logger.error(f"Failed to download positive papers for {datatype}: {e}")

        # Download negative papers
        try:
            logger.info(f"Downloading negative papers for {datatype}")
            url = self.build_url(datatype, "negative")
            response = self.session.get(url, auth=self.auth, timeout=30)
            response.raise_for_status()
            negative_papers = self.extract_papers(response.text)
            logger.info(f"Found {len(negative_papers)} negative papers for {datatype}")
        except Exception as e:
            logger.error(f"Failed to download negative papers for {datatype}: {e}")

        return positive_papers, negative_papers

    def save_to_csv(self, datatype: str, positive_papers: List[str],
                    negative_papers: List[str], output_dir: str):
        """Save papers to CSV file in the required format"""
        # Create output directory for datatype if it doesn't exist
        datatype_dir = os.path.join(output_dir, datatype)
        os.makedirs(datatype_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{datatype}_{timestamp}.csv"
        filepath = os.path.join(datatype_dir, filename)

        # Write CSV file
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['AGRKBID', 'XREF', 'Classification'])

            # Write positive papers
            for paper in positive_papers:
                # AGRKBID is empty, XREF contains the WBPaper ID
                writer.writerow(['', paper, 'positive'])

            # Write negative papers
            for paper in negative_papers:
                writer.writerow(['', paper, 'negative'])

        logger.info(f"Saved {len(positive_papers) + len(negative_papers)} papers to {filepath}")
        return filepath


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download training sets from WormBase Curation Status Page"
    )
    parser.add_argument(
        '--datatypes',
        nargs='*',
        choices=AVAILABLE_DATATYPES + ['all'],
        default=['all'],
        help='Data types to download (space-separated). Use "all" for all types.'
    )
    parser.add_argument(
        '--output-dir',
        default='/data/training_sets/WB',
        help='Output directory for CSV files (default: /data/training_sets/WB)'
    )
    parser.add_argument(
        '--username',
        help='WormBase username (or set WB_USERNAME environment variable)'
    )
    parser.add_argument(
        '--password',
        help='WormBase password (or set WB_PASSWORD environment variable)'
    )
    parser.add_argument(
        '--list-datatypes',
        action='store_true',
        help='List all available data types and exit'
    )

    args = parser.parse_args()

    # Handle list datatypes option
    if args.list_datatypes:
        print("Available data types:")
        for dt in AVAILABLE_DATATYPES:
            print(f"  {dt}")
        return

    # Get credentials
    username = args.username or os.getenv('WB_USERNAME')
    password = args.password or os.getenv('WB_PASSWORD')

    if not username or not password:
        logger.error("Username and password are required. Provide via command line or set WB_USERNAME and WB_PASSWORD environment variables")
        sys.exit(1)

    # Determine which datatypes to download
    if not args.datatypes:
        logger.error("No datatypes specified. Use --datatypes to specify which types to download, or use 'all'")
        sys.exit(1)

    datatypes_to_download = AVAILABLE_DATATYPES if 'all' in args.datatypes else args.datatypes

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        logger.error(f"Output directory {args.output_dir} does not exist. Please create it first.")
        sys.exit(1)

    # Initialize downloader
    downloader = WormBaseDownloader(username, password)

    # Track results
    results = {}

    # Download each datatype
    for datatype in datatypes_to_download:
        logger.info(f"\nProcessing datatype: {datatype}")
        try:
            positive_papers, negative_papers = downloader.download_datatype(datatype)
            if positive_papers or negative_papers:
                filepath = downloader.save_to_csv(
                    datatype, positive_papers, negative_papers, args.output_dir
                )
                results[datatype] = {
                    'positive': len(positive_papers),
                    'negative': len(negative_papers),
                    'file': filepath
                }
            else:
                logger.warning(f"No papers found for {datatype}")
                results[datatype] = {
                    'positive': 0,
                    'negative': 0,
                    'file': None
                }
        except Exception as e:
            logger.error(f"Failed to process {datatype}: {e}")
            results[datatype] = {'error': str(e)}

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for datatype, info in results.items():
        if 'error' in info:
            print(f"{datatype}: ERROR - {info['error']}")
        elif info['file']:
            print(f"{datatype}: {info['positive']} positive, {info['negative']} negative -> {info['file']}")
        else:
            print(f"{datatype}: No papers found")
    print("="*60)


if __name__ == "__main__":
    main()
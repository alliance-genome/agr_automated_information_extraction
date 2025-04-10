import logging
import argparse

logger = logging.getLogger(__name__)

helptext = r"""
    This script has been split into 2 new scripts.
    agr_document_classifier_classify.py
    and
    agr_document_classifier_trainer.py
    
    Please run one of these instead.
"""

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=helptext)
    parser.add_argument("-m", "--mode", type=str, choices=['train', 'classify'], default="classify",
                        help="Mode of operation: train or classify")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.mode == "classify":
        logger.error("Please use agr_document_classifier_classify.py. This has replaced this script.")
    else:
        logger.error("Please use agr_document_classifier_trainer.py. This has replaced this script.")



if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""CLI shim so bash callers can dispatch send_report() without re-implementing SMTP.

Usage:
    _send_report_shim.py <subject> <body>

Reads CRONTAB_EMAIL / SENDER_EMAIL / SENDER_PASSWORD from the environment,
exactly like the existing Python callers in export_fb_tets.py.
"""
import sys

from agr_literature_service.lit_processing.utils.report_utils import send_report


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: _send_report_shim.py <subject> <body>", file=sys.stderr)
        return 2
    send_report(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())

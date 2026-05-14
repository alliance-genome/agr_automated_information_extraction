#!/usr/bin/env bash
# Daily freshness check for the four FB ABC textmining output files in SVN.
# If any file's last commit is older than STALE_THRESHOLD_DAYS (default 3),
# emails curators.
#
# Designed to run inside the agr_document_classifier container so it has access
# to svn 1.14 and to send_report. The cron entry on the GoCD agent should be:
#
#   30 13 * * * docker run --rm \
#     -e CRONTAB_EMAIL -e SENDER_EMAIL -e SENDER_PASSWORD \
#     -e SVN_REPO_URL=https://svn.flybase.org/.../curation_status \
#     -v "$HOME/.subversion:/root/.subversion" \
#     agr_document_classifier \
#     ./bin/check_textmining_freshness.sh
#
# Required env vars:
#   SVN_REPO_URL                                   - parent URL of the
#                                                    textmining_*.txt files
#   CRONTAB_EMAIL, SENDER_EMAIL, SENDER_PASSWORD   - for alert dispatch
#
# Optional:
#   STALE_THRESHOLD_DAYS    age cutoff in days (default: 3)
#
# Always exits 0 — alerting is the side effect; we don't want a stale-files
# alarm to also fail the cron entry itself.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SVN_REPO_URL:-}" ]]; then
    echo "ERROR: SVN_REPO_URL must be set" >&2
    exit 0
fi

STALE_THRESHOLD_DAYS="${STALE_THRESHOLD_DAYS:-3}"

FILES=(
    "textmining_positive_ABC.txt"
    "textmining_negative_ABC.txt"
    "textmining_positive_ABC_using_score.txt"
    "textmining_negative_ABC_using_score.txt"
)

threshold_seconds=$(( STALE_THRESHOLD_DAYS * 86400 ))
now_epoch=$(date -u +%s)
stale_report=""

for f in "${FILES[@]}"; do
    url="${SVN_REPO_URL%/}/$f"
    iso=$(svn info --show-item last-changed-date "$url" 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$iso" ]]; then
        stale_report+="MISSING/UNREADABLE: $url"$'\n'
        continue
    fi
    # svn emits e.g. 2026-05-14T12:48:26.123456Z; GNU date handles that directly.
    mtime=$(date -u -d "$iso" +%s 2>/dev/null)
    if [[ -z "$mtime" ]]; then
        stale_report+="UNPARSEABLE DATE for $url: $iso"$'\n'
        continue
    fi
    age=$(( now_epoch - mtime ))
    if (( age > threshold_seconds )); then
        days=$(( age / 86400 ))
        stale_report+="STALE (${days}d old, last commit $iso): $url"$'\n'
    fi
done

if [[ -z "$stale_report" ]]; then
    exit 0
fi

body="The following FB ABC textmining files under $SVN_REPO_URL
have not been re-committed within the last $STALE_THRESHOLD_DAYS days.
The ExportFBClassifiers pipeline may be wedged.

$stale_report
Suggested checks:
  * GoCD pipeline 'ExportFBClassifiers' run history
  * Latest run_export_and_commit.sh log on the agent
  * svn log -l 5 $SVN_REPO_URL"

if ! python3 "$SCRIPT_DIR/_send_report_shim.py" \
        "FB ABC textmining files are stale (>${STALE_THRESHOLD_DAYS}d)" \
        "$body" 2>&1; then
    echo "WARNING: freshness alert dispatch failed" >&2
fi

exit 0

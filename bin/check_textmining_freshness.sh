#!/usr/bin/env bash
# Daily freshness check for the four FB ABC textmining output files.
# If any file's mtime is older than STALE_THRESHOLD_DAYS (default 3), emails curators.
#
# Designed to run as a cron entry on the FlyBase GoCD agent (which is the only
# place that can stat the real SVN working copy directly).
#
# Optional env vars:
#   CURATION_STATUS_DIR     host path of the SVN working copy
#                           (default: /var/go/alliance_gocd/go-agent-flybase-3/pipelines/ExportFBClassifiers/curation_status)
#   STALE_THRESHOLD_DAYS    age cutoff in days (default: 3)
#
# Always exits 0 — alerting is the side effect; we don't want a stale-files alarm
# to also fail the cron entry itself.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATION_STATUS_DIR="${CURATION_STATUS_DIR:-/var/go/alliance_gocd/go-agent-flybase-3/pipelines/ExportFBClassifiers/curation_status}"
STALE_THRESHOLD_DAYS="${STALE_THRESHOLD_DAYS:-3}"

FILES=(
    "textmining_positive_ABC.txt"
    "textmining_negative_ABC.txt"
    "textmining_positive_ABC_using_score.txt"
    "textmining_negative_ABC_using_score.txt"
    ".last_success"
)

# stat varies between GNU (Linux) and BSD (macOS); try both.
file_mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null
}

format_epoch() {
    date -u -d "@$1" +%FT%TZ 2>/dev/null || date -u -r "$1" +%FT%TZ 2>/dev/null
}

threshold_seconds=$(( STALE_THRESHOLD_DAYS * 86400 ))
now_epoch=$(date +%s)
stale_report=""

for f in "${FILES[@]}"; do
    path="$CURATION_STATUS_DIR/$f"
    if [[ ! -e "$path" ]]; then
        stale_report+="MISSING: $path"$'\n'
        continue
    fi
    mtime=$(file_mtime "$path")
    if [[ -z "$mtime" ]]; then
        stale_report+="UNREADABLE: $path"$'\n'
        continue
    fi
    age=$(( now_epoch - mtime ))
    if (( age > threshold_seconds )); then
        days=$(( age / 86400 ))
        human=$(format_epoch "$mtime")
        stale_report+="STALE (${days}d old, last modified $human): $path"$'\n'
    fi
done

if [[ -z "$stale_report" ]]; then
    exit 0
fi

body="The following FB ABC textmining files in $CURATION_STATUS_DIR on $(hostname)
have not been refreshed within the last $STALE_THRESHOLD_DAYS days.
The ExportFBClassifiers pipeline may be wedged.

$stale_report
Suggested checks:
  * GoCD pipeline 'ExportFBClassifiers' run history
  * svn status in $CURATION_STATUS_DIR (look for 'C' entries)
  * Latest run_export_and_commit.sh log on this agent"

if ! python3 "$SCRIPT_DIR/_send_report_shim.py" \
        "FB ABC textmining files are stale (>${STALE_THRESHOLD_DAYS}d)" \
        "$body" 2>&1; then
    echo "WARNING: freshness alert dispatch failed" >&2
fi

exit 0

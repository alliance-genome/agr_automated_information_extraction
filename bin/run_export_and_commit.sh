#!/usr/bin/env bash
# Self-healing wrapper for the FlyBase ABC textmining export + SVN commit step.
#
# This script is intended to run INSIDE the agr_document_classifier container.
# The container owns the SVN working copy end-to-end: it checks it out on every
# run (the WC is ephemeral; no host bind-mount), the export scripts write to it,
# and the wrapper commits back. Freshness monitoring no longer depends on a
# host-side file — it queries SVN directly (see bin/check_textmining_freshness.sh).
#
# The GoCD task only needs to mount the agent's ~/.subversion/ for cached
# credentials:
#
#   docker run --rm \
#     -e BLUE_PASSWORD -e CRONTAB_EMAIL -e SENDER_EMAIL -e SENDER_PASSWORD \
#     -e SVN_REPO_URL=https://svn.flybase.org/.../curation_status \
#     -e CURATION_STATUS_DIR=/curation_status \
#     -v "$HOME/.subversion:/root/.subversion" \
#     agr_document_classifier \
#     ./bin/run_export_and_commit.sh
#
# CURATION_STATUS_DIR must be /curation_status: the export scripts hardcode
# /curation_status/textmining_*.txt as their output path.
#
# Flow:
#   1. If /curation_status has no .svn/, run `svn checkout $SVN_REPO_URL` to
#      create the working copy. Otherwise `svn update`; on E155036 (format too
#      old), run `svn upgrade` and retry once.
#   2. Auto-resolve any pre-existing C-state entries with --accept=theirs-full
#      (so the export writes onto a clean file, not one full of conflict markers).
#   3. Run the two export scripts directly via python3 (override EXPORT_RUNNER
#      to swap in your own invocation).
#   4. svn add --force; svn commit.
#   5. On commit failure with E155015: re-resolve with --accept=mine-full, retry once.
#   6. On any unrecoverable failure: send_report via the Python shim, exit non-zero.
#   7. On success: exit 0. Freshness is monitored separately via the wrapper's
#      SVN commits (see bin/check_textmining_freshness.sh).
#
# Required env vars (typically set by the GoCD pipeline):
#   CRONTAB_EMAIL, SENDER_EMAIL, SENDER_PASSWORD   - for alert dispatch
#   BLUE_PASSWORD                                  - DB access for the export scripts.
#                                                    Other BLUE_* vars (USERNAME, HOST,
#                                                    PORT, DATABASE) have prod defaults
#                                                    baked into the Python scripts.
#   SVN_REPO_URL                                   - the SVN URL to check out from on
#                                                    first run (when .svn/ is missing).
#                                                    Unused once the working copy
#                                                    exists but cheap to keep set.
#   SVN_PASSWORD                                   - consumed by svn via the cached
#                                                    credentials bind-mounted from the
#                                                    agent (not read by this script
#                                                    directly)
#
# Optional overrides:
#   CURATION_STATUS_DIR  path of the SVN working copy inside the container. Defaults
#                        to "$PWD/curation_status"; the GoCD invocation above sets it
#                        explicitly to /curation_status (the bind-mount target).
#   EXPORT_RUNNER        command that runs both export scripts; receives no args
#                        (default: invoke export_fb_tets.py and export_fb_tets_using_score.py
#                        via python3)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture $PWD before we cd anywhere, so the default is the caller's CWD.
CURATION_STATUS_DIR="${CURATION_STATUS_DIR:-$PWD/curation_status}"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

send_alert() {
    local subject="$1"
    local body="$2"
    log "ALERT: $subject"
    if ! python3 "$SCRIPT_DIR/_send_report_shim.py" "$subject" "$body" 2>&1; then
        log "WARNING: send_report shim failed; alert may not have been delivered"
    fi
}

# Resolves all C-state entries in the working copy with the given --accept value.
# Echoes the list of files it acted on (so callers can include it in alerts).
resolve_conflicts() {
    local accept="$1"
    local conflicted
    conflicted=$(svn status "$CURATION_STATUS_DIR" 2>/dev/null | awk '/^C/ {print $2}')
    if [[ -z "$conflicted" ]]; then
        return 0
    fi
    log "Conflicted entries detected; resolving with --accept=$accept:"
    log "$conflicted"
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        if ! svn resolve --accept="$accept" "$f"; then
            log "WARNING: svn resolve --accept=$accept failed for $f"
        fi
    done <<< "$conflicted"
    echo "$conflicted"
}

default_export_runner() {
    python3 /usr/src/app/export_fb_tets.py || return $?
    python3 /usr/src/app/export_fb_tets_using_score.py
}

run_export() {
    if [[ -n "${EXPORT_RUNNER:-}" ]]; then
        bash -c "$EXPORT_RUNNER"
    else
        default_export_runner
    fi
}

# ---- main ----

mkdir -p "$CURATION_STATUS_DIR"
if ! cd "$CURATION_STATUS_DIR"; then
    send_alert "FB ABC textmining: working copy unreachable" \
               "Cannot cd into CURATION_STATUS_DIR=$CURATION_STATUS_DIR on $(hostname)."
    exit 2
fi

if [[ ! -d "$CURATION_STATUS_DIR/.svn" ]]; then
    if [[ -z "${SVN_REPO_URL:-}" ]]; then
        send_alert "FB ABC textmining: SVN_REPO_URL not set on first run" \
                   "Host: $(hostname)
$CURATION_STATUS_DIR has no .svn/ directory and SVN_REPO_URL is unset, so the
wrapper cannot bootstrap the working copy. Set SVN_REPO_URL in the GoCD task."
        exit 2
    fi
    log "No .svn/ found in $CURATION_STATUS_DIR; running svn checkout $SVN_REPO_URL"
    checkout_out=$(svn checkout "$SVN_REPO_URL" "$CURATION_STATUS_DIR" 2>&1)
    checkout_rc=$?
    log "$checkout_out"
    if (( checkout_rc != 0 )); then
        send_alert "FB ABC textmining: svn checkout failed" \
                   "Host: $(hostname)
Working copy: $CURATION_STATUS_DIR
SVN_REPO_URL: $SVN_REPO_URL

svn checkout output:
$checkout_out"
        exit "$checkout_rc"
    fi
fi

log "Running svn update in $CURATION_STATUS_DIR"
update_out=$(svn update 2>&1)
update_rc=$?
log "$update_out"
if (( update_rc != 0 )) && grep -q "E155036" <<< "$update_out"; then
    log "Working copy is in an older format; running svn upgrade and retrying"
    upgrade_out=$(svn upgrade "$CURATION_STATUS_DIR" 2>&1)
    log "$upgrade_out"
    update_out=$(svn update 2>&1)
    update_rc=$?
    log "$update_out"
fi
if (( update_rc != 0 )); then
    send_alert "FB ABC textmining: svn update failed" \
               "Host: $(hostname)
Working copy: $CURATION_STATUS_DIR

svn update output:
$update_out"
    exit "$update_rc"
fi

resolve_conflicts "theirs-full" >/dev/null

log "Running export scripts"
run_export
export_rc=$?
if (( export_rc != 0 )); then
    log "Export runner exited $export_rc"
    # The Python scripts call send_report() themselves on exception. Re-alert here
    # in case the failure was the runner itself (e.g. EXPORT_RUNNER override broken).
    send_alert "FB ABC textmining: export scripts failed" \
               "Host: $(hostname)
Working copy: $CURATION_STATUS_DIR
EXPORT_RUNNER exit code: $export_rc

If the export scripts themselves crashed they already emailed you separately."
    exit "$export_rc"
fi

log "Running svn add --force (idempotent for tracked files)"
svn add --force "$CURATION_STATUS_DIR" 2>&1 | while IFS= read -r line; do log "  $line"; done

commit_msg="automated textmining ABC update $(date -u +%FT%TZ)"
log "Running svn commit: $commit_msg"
commit_out=$(svn commit -m "$commit_msg" "$CURATION_STATUS_DIR" 2>&1)
commit_rc=$?
log "$commit_out"

if (( commit_rc == 0 )); then
    log "Commit succeeded"
    exit 0
fi

if grep -q "E155015" <<< "$commit_out"; then
    log "Detected E155015 conflict on commit; auto-resolving with --accept=mine-full and retrying once"
    resolved_files=$(resolve_conflicts "mine-full")
    retry_out=$(svn commit -m "$commit_msg (post-resolve retry)" "$CURATION_STATUS_DIR" 2>&1)
    retry_rc=$?
    log "$retry_out"
    if (( retry_rc == 0 )); then
        log "Retry commit succeeded after auto-resolve"
        # Still inform humans that self-heal kicked in, so a curator can sanity-check
        # the committed content. Non-fatal but worth eyes on.
        send_alert "FB ABC textmining: svn conflict auto-resolved" \
                   "Host: $(hostname)
Working copy: $CURATION_STATUS_DIR

Initial commit failed with E155015. The wrapper ran svn resolve --accept=mine-full
on the following files and retried successfully:

$resolved_files

Recommend spot-checking the committed file contents to confirm no curator edits were lost."
        exit 0
    fi
    send_alert "FB ABC textmining: svn commit failed after auto-resolve retry" \
               "Host: $(hostname)
Working copy: $CURATION_STATUS_DIR

Initial commit error:
$commit_out

Files we attempted to resolve with --accept=mine-full:
$resolved_files

Retry commit error:
$retry_out"
    exit "$retry_rc"
fi

# Non-conflict failure (auth, network, repo down, locked working copy, etc.)
send_alert "FB ABC textmining: svn commit failed" \
           "Host: $(hostname)
Working copy: $CURATION_STATUS_DIR

svn commit output (exit $commit_rc):
$commit_out"
exit "$commit_rc"

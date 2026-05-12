#!/usr/bin/env bash
# Self-healing wrapper for the FlyBase ABC textmining export + SVN commit step.
#
# Flow:
#   1. svn update; auto-resolve any pre-existing C-state entries with --accept=theirs-full
#      (so the export writes onto a clean file, not one full of conflict markers)
#   2. Run the two export scripts (inside the agr_document_classifier docker image
#      by default; override EXPORT_RUNNER to swap in your own invocation)
#   3. svn add --force; svn commit
#   4. On commit failure with E155015: re-resolve with --accept=mine-full, retry once
#   5. On any unrecoverable failure: send_report via the Python shim, exit non-zero
#   6. On success: touch .last_success sentinel for the freshness checker
#
# Required env vars (typically set by the GoCD pipeline):
#   CRONTAB_EMAIL, SENDER_EMAIL, SENDER_PASSWORD   - for alert dispatch
#   BLUE_PASSWORD                                  - forwarded into docker for the
#                                                    export scripts' DB access. The
#                                                    other BLUE_* vars (USERNAME,
#                                                    HOST, PORT, DATABASE) have prod
#                                                    defaults baked into the Python
#                                                    scripts and are not required.
#   SVN_PASSWORD                                   - consumed by svn via cached
#                                                    credentials on the GoCD agent
#                                                    (not read by this script directly)
#
# Optional overrides:
#   CURATION_STATUS_DIR  host path of the SVN working copy. Defaults to
#                        "$PWD/curation_status" — when this script is invoked
#                        as a GoCD pipeline task, $PWD is the pipeline working
#                        directory (e.g. /var/go/<agent>/pipelines/ExportFBClassifiers),
#                        so the default resolves correctly regardless of which
#                        agent the job lands on.
#   EXPORT_RUNNER        command that runs both export scripts; receives no args
#                        (default: docker run agr_document_classifier ... for both scripts)
#   DOCKER_IMAGE         docker image name used by the default EXPORT_RUNNER
#                        (default: agr_document_classifier)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture $PWD before we cd anywhere, so the default is the GoCD task's CWD.
CURATION_STATUS_DIR="${CURATION_STATUS_DIR:-$PWD/curation_status}"
DOCKER_IMAGE="${DOCKER_IMAGE:-agr_document_classifier}"

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
    docker run --rm \
        -e BLUE_PASSWORD \
        -v "$CURATION_STATUS_DIR:/curation_status" \
        "$DOCKER_IMAGE" \
        python3 /usr/src/app/export_fb_tets.py
    local rc1=$?
    if (( rc1 != 0 )); then return "$rc1"; fi

    docker run --rm \
        -e BLUE_PASSWORD \
        -v "$CURATION_STATUS_DIR:/curation_status" \
        "$DOCKER_IMAGE" \
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

if ! cd "$CURATION_STATUS_DIR"; then
    send_alert "FB ABC textmining: working copy unreachable" \
               "Cannot cd into CURATION_STATUS_DIR=$CURATION_STATUS_DIR on $(hostname)."
    exit 2
fi

log "Running svn update in $CURATION_STATUS_DIR"
update_out=$(svn update 2>&1)
update_rc=$?
log "$update_out"
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
    # in case the failure was the docker invocation itself (image missing, mount denied).
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
    date -u +%FT%TZ > "$CURATION_STATUS_DIR/.last_success"
    log "Commit succeeded; .last_success updated"
    exit 0
fi

if grep -q "E155015" <<< "$commit_out"; then
    log "Detected E155015 conflict on commit; auto-resolving with --accept=mine-full and retrying once"
    resolved_files=$(resolve_conflicts "mine-full")
    retry_out=$(svn commit -m "$commit_msg (post-resolve retry)" "$CURATION_STATUS_DIR" 2>&1)
    retry_rc=$?
    log "$retry_out"
    if (( retry_rc == 0 )); then
        date -u +%FT%TZ > "$CURATION_STATUS_DIR/.last_success"
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

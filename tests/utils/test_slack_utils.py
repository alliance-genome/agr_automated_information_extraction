"""Tests for the Slack notification helper used by the entity extraction job.

The alert is delivered by emailing the scrum-blueteam-private channel's
"send email to channel" address, reusing agr_literature_service.send_report.
send_slack_notification must be best-effort: a failure to send must never raise
(so it cannot crash the job or mask the original error).
"""

from unittest.mock import patch

from utils import slack_utils
from utils.slack_utils import (
    send_slack_notification,
    format_traceback_html,
    format_skipped_jobs_html,
    build_entity_run_summary_html,
    DEFAULT_SLACK_CHANNEL_EMAIL,
)


def test_sends_email_to_default_channel_address():
    with patch.dict("os.environ", {}, clear=True):
        with patch(
            "agr_literature_service.lit_processing.utils.report_utils.send_report"
        ) as mock_send:
            assert send_slack_notification("subj", "body") is True
            mock_send.assert_called_once_with(
                "subj", "body", email=DEFAULT_SLACK_CHANNEL_EMAIL
            )


def test_channel_address_overridable_via_env():
    override = "custom-channel@example.slack.com"
    with patch.dict("os.environ", {"SLACK_CHANNEL_EMAIL": override}, clear=True):
        with patch(
            "agr_literature_service.lit_processing.utils.report_utils.send_report"
        ) as mock_send:
            assert send_slack_notification("subj", "body") is True
            mock_send.assert_called_once_with("subj", "body", email=override)


def test_returns_false_and_never_raises_on_failure():
    with patch.dict("os.environ", {}, clear=True):
        with patch(
            "agr_literature_service.lit_processing.utils.report_utils.send_report",
            side_effect=RuntimeError("smtp down"),
        ):
            assert send_slack_notification("subj", "body") is False


def test_returns_false_when_library_import_fails(monkeypatch):
    # Simulate agr_literature_service being unavailable at call time.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("agr_literature_service"):
            raise ImportError("no agr_literature_service")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert send_slack_notification("subj", "body") is False


def test_format_traceback_html_escapes_and_wraps():
    out = format_traceback_html("Error <tag> & 'x'")
    assert out.startswith("<pre>") and out.endswith("</pre>")
    assert "&lt;tag&gt;" in out
    assert "&amp;" in out


def test_slack_utils_module_has_no_requests_dependency():
    # Delivery is via email (send_report), not an HTTP webhook.
    assert not hasattr(slack_utils, "requests")


def test_format_skipped_jobs_html_empty_returns_blank():
    assert format_skipped_jobs_html([]) == ""


def test_format_skipped_jobs_html_summarizes_counts_and_reasons():
    skipped = [
        {"mod_abbreviation": "WB", "topic": "ATP:0000005", "jobs": 12,
         "reason": "extraction model not found"},
        {"mod_abbreviation": "ZFIN", "topic": "ATP:0000006", "jobs": 3},
    ]
    out = format_skipped_jobs_html(skipped)
    assert "Skipped 15 job(s) across 2 mod/topic" in out
    assert "mod: WB, topic: ATP:0000005 — 12 job(s) skipped (extraction model not found)" in out
    # missing 'reason' falls back to the default
    assert "mod: ZFIN, topic: ATP:0000006 — 3 job(s) skipped (model not found)" in out


def test_build_entity_run_summary_reports_all_three_categories():
    skipped = [{"mod_abbreviation": "WB", "topic": "ATP:0000005", "jobs": 4,
                "reason": "extraction model not found"}]
    out = build_entity_run_summary_html(total_failed=2, total_jobs=10,
                                        total_md_skipped=3, skipped_jobs=skipped)
    assert "2 of 10 per-item job(s) failed (fulltext fetch error)" in out
    assert "3 reference(s) skipped (MD file load error)" in out
    assert "Skipped 4 job(s)" in out


def test_build_entity_run_summary_omits_zero_categories():
    # A run with only MD-load skips must not claim any failures.
    out = build_entity_run_summary_html(total_failed=0, total_jobs=10,
                                        total_md_skipped=5, skipped_jobs=[])
    assert "failed" not in out
    assert "5 reference(s) skipped (MD file load error)" in out

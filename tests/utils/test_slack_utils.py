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

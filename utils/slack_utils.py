import html
import logging
import os

logger = logging.getLogger(__name__)

# Email-to-channel address for the "scrum-blueteam-private" Slack channel.
# Slack's "send email to a channel" feature posts inbound email into the channel.
# Overridable via the SLACK_CHANNEL_EMAIL env var.
DEFAULT_SLACK_CHANNEL_EMAIL = (
    "scrum-blueteam-privat-aaaapxhnp7z6ohf3yeaagi3kha@alliance-project.slack.com"
)


def send_slack_notification(subject: str, message: str) -> bool:
    """Post an alert to the scrum-blueteam-private Slack channel via its email address.

    Reuses agr_literature_service's send_report, which sends HTML email using the
    SENDER_EMAIL / SENDER_PASSWORD / REPLY_TO environment variables. Never raises:
    a notification failure must not crash the job or mask the original error.
    Returns True if the email was dispatched, False otherwise.
    """
    recipient = os.environ.get("SLACK_CHANNEL_EMAIL", DEFAULT_SLACK_CHANNEL_EMAIL)
    try:
        from agr_literature_service.lit_processing.utils.report_utils import send_report
        send_report(subject, message, email=recipient)
        return True
    except Exception as e:
        logger.error(f"Failed to send Slack notification email: {e}")
        return False


def format_traceback_html(tb: str) -> str:
    """Wrap a traceback string as HTML-safe <pre> for the email body."""
    return f"<pre>{html.escape(tb)}</pre>"

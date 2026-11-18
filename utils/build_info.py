"""Which commit this image was built from (SCRUM-6878).

The Dockerfile's build_info stage reads the commit from the build context's .git
into BUILD_INFO_DIR, so it is known however the image is built. GIT_SHA and
BUILD_DATE take precedence when set (build arguments). Anything not found is
reported as "unknown" rather than guessed. Each job logs build_info_line() when it
starts, so its log shows which code ran it.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

UNKNOWN = "unknown"
BUILD_INFO_DIR = Path("/etc/abc_build_info")


def _lookup(env_name: str, file_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if value and value != UNKNOWN:
        return value
    try:
        value = (BUILD_INFO_DIR / file_name).read_text().strip()
    except OSError:
        value = ""
    return value or UNKNOWN


def get_build_info() -> Dict[str, str]:
    return {
        "commit": _lookup("GIT_SHA", "commit"),
        "build_date": _lookup("BUILD_DATE", "build_date"),
    }


def build_info_line() -> str:
    info = get_build_info()
    return f"agr_automated_information_extraction commit {info['commit']}, built {info['build_date']}"


def log_build_info(logger: Optional[logging.Logger] = None) -> None:
    (logger or logging.getLogger(__name__)).info(build_info_line())

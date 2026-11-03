"""bin/run_export_and_commit.sh only touches FlyBase's SVN when FB_SVN_COMMIT=true.

The production GoCD pipeline sets it. Any other run (stage, a test, someone at a shell) runs the
export with no svn call at all and no SVN credentials, and the export prints what it would write.
"""
import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "bin" / "run_export_and_commit.sh"
EXPORT_OUTPUT = "261030\t\t12345\tnocur:high\tcam"


def _run(tmp_path, fb_svn_commit=None):
    calls = tmp_path / "svn_calls.log"
    calls.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    svn = bin_dir / "svn"
    # A stand-in svn that records its arguments.
    svn.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{calls}"\nexit 0\n')
    svn.chmod(svn.stat().st_mode | stat.S_IEXEC)
    working_copy = tmp_path / "curation_status"
    (working_copy / ".svn").mkdir(parents=True)

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CURATION_STATUS_DIR": str(working_copy),
        "EXPORT_RUNNER": f"printf '{EXPORT_OUTPUT}\\n'",
    }
    if fb_svn_commit is not None:
        env["FB_SVN_COMMIT"] = fb_svn_commit
    if fb_svn_commit == "true":
        env.update({"SVN_USERNAME": "u", "SVN_PASSWORD": "p"})
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    return result, calls.read_text()


def test_dry_run_without_the_gocd_variable_never_calls_svn(tmp_path):
    # No SVN credentials are set either: a dry run must not need them.
    result, calls = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == ""
    assert "dry run" in result.stdout
    assert EXPORT_OUTPUT in result.stdout


def test_only_the_exact_value_true_commits(tmp_path):
    result, calls = _run(tmp_path, fb_svn_commit="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == ""


def test_commits_when_the_gocd_variable_is_true(tmp_path):
    result, calls = _run(tmp_path, fb_svn_commit="true")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "update" in calls
    assert "add --force" in calls
    assert " commit " in f" {calls} "

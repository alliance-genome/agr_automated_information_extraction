"""Tests for reporting which commit a job runs (SCRUM-6878)."""
import logging
from pathlib import Path

from utils import build_info
from utils.build_info import build_info_line, get_build_info, log_build_info

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_reads_the_files_from_the_image_build(monkeypatch, tmp_path):
    (tmp_path / "commit").write_text("e3d658e7\n")
    (tmp_path / "build_date").write_text("2026-11-18T13:30:00Z\n")
    monkeypatch.setattr(build_info, "BUILD_INFO_DIR", tmp_path)
    # the Dockerfile's ARG defaults leave the env at "unknown" when no build argument is passed
    monkeypatch.setenv("GIT_SHA", "unknown")
    monkeypatch.setenv("BUILD_DATE", "unknown")
    assert get_build_info() == {"commit": "e3d658e7", "build_date": "2026-11-18T13:30:00Z"}


def test_build_arguments_take_precedence(monkeypatch, tmp_path):
    (tmp_path / "commit").write_text("e3d658e7\n")
    monkeypatch.setattr(build_info, "BUILD_INFO_DIR", tmp_path)
    monkeypatch.setenv("GIT_SHA", "12345678")
    monkeypatch.setenv("BUILD_DATE", "2026-11-18T14:00:00Z")
    assert get_build_info() == {"commit": "12345678", "build_date": "2026-11-18T14:00:00Z"}


def test_unknown_outside_an_image(monkeypatch, tmp_path):
    monkeypatch.setattr(build_info, "BUILD_INFO_DIR", tmp_path / "missing")
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("BUILD_DATE", raising=False)
    assert build_info_line() == "agr_automated_information_extraction commit unknown, built unknown"


def test_logs_one_info_line(monkeypatch, caplog):
    monkeypatch.setenv("GIT_SHA", "e3d658e7")
    monkeypatch.setenv("BUILD_DATE", "2026-11-18T13:30:00Z")
    with caplog.at_level(logging.INFO):
        log_build_info(logging.getLogger("job"))
    assert [r.getMessage() for r in caplog.records] == [
        "agr_automated_information_extraction commit e3d658e7, built 2026-11-18T13:30:00Z"]


def test_dockerfile_records_the_commit():
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "FROM python:3.11-alpine3.20 AS build_info" in text
    assert "COPY .git /tmp/repo.git" in text
    assert f"COPY --from=build_info /build_info {build_info.BUILD_INFO_DIR}" in text
    # COPY .git fails the build if a .dockerignore ever excludes it
    dockerignore = REPO_ROOT / ".dockerignore"
    if dockerignore.exists():
        patterns = {line.strip() for line in dockerignore.read_text().splitlines()}
        assert not {".git", ".git/", "/.git", "**/.git"} & patterns


def test_every_job_copied_into_the_image_logs_its_commit():
    # the job scripts the Dockerfile ADDs to /usr/src/app (not helpers or the old split-up pipeline)
    jobs = [
        "agr_document_classifier/agr_document_classifier_classify.py",
        "agr_document_classifier/agr_document_classifier_trainer.py",
        "agr_document_classifier/agr_priority_classifier_balanced.py",
        "agr_document_classifier/agr_antibody_string_matching_classifier.py",
        "agr_dataset_manager/dataset_downloader.py",
        "agr_dataset_manager/dataset_upload_from_csv.py",
        "agr_entity_extractor/agr_entity_extraction_pipeline.py",
        "agr_entity_extractor/agr_entity_extraction_pipeline_fast.py",
        "agr_entity_extractor/agr_species_extraction_pipeline.py",
        "agr_entity_extractor/fit_and_upload_tfidf_vectorizer.py",
        "agr_entity_extractor/fit_and_upload_tfidf_vectorizer_fast.py",
        "agr_entity_extractor/upload_string_matching_extractor.py",
        "export_data/export_fb_tets.py",
        "export_data/export_fb_tets_using_score.py",
    ]
    missing = [job for job in jobs
               if "log_build_info()" not in (REPO_ROOT / job).read_text()
               and "print(build_info_line())" not in (REPO_ROOT / job).read_text()]
    assert missing == []

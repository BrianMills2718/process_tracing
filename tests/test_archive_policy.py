"""Tests for OKF archive/wiki policy boundaries."""

from __future__ import annotations

from pathlib import Path


def test_current_repo_wiki_uses_okf_log_location():
    bundle = Path("process-tracing-current")

    assert (bundle / "CLAUDE.md").is_file()
    assert (bundle / "wiki" / "index.md").is_file()
    assert (bundle / "wiki" / "log.md").is_file()
    assert not (bundle / "log.md").exists()


def test_archive_policy_moves_retired_material_outside_projects():
    text = Path("docs/ARCHIVE_POLICY.md").read_text(encoding="utf-8")

    assert "~/archive/process_tracing" in text
    assert "wiki/log.md" in text
    assert "must not search `~/archive/process_tracing` during normal implementation" in text


def test_archive_bundle_uses_okf_log_location():
    bundle = Path("/home/brian/archive/process_tracing")

    assert (bundle / "CLAUDE.md").is_file()
    assert (bundle / "wiki" / "index.md").is_file()
    assert (bundle / "wiki" / "log.md").is_file()
    assert not (bundle / "log.md").exists()


"""Tests for active architecture documentation required by design-plan."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.plans(3)
def test_architecture_doc_has_required_design_plan_diagrams():
    text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "## Boundary Diagram" in text
    assert "## Domain Model Diagram" in text
    assert "## Data-Flow And Contract Diagram" in text
    assert "```mermaid\nflowchart LR" in text
    assert "```mermaid\nclassDiagram" in text
    assert "```mermaid\nsequenceDiagram" in text
    assert "Typed contracts:" in text
    assert "## Backward Runtime Pass" in text

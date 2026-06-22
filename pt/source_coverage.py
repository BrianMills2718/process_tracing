"""Deterministic source-packet coverage checks.

Coverage is provenance plumbing, not semantic evidence classification. The
packet supplies exact text markers for each source; this module checks whether
those markers appear in the assembled input text and in extracted evidence
quotes.
"""

from __future__ import annotations

import re
from typing import Literal

from pt.schemas import ExtractionResult
from pt.source_packet import (
    SourceCandidate,
    SourceCoverageItem,
    SourceCoverageReport,
    SourcePacket,
)


def build_source_coverage(
    packet: SourcePacket,
    input_text: str,
    extraction: ExtractionResult,
) -> SourceCoverageReport:
    """Build a packet-source coverage report from input text and evidence."""

    items: list[SourceCoverageItem] = []
    assigned_evidence_ids: set[str] = set()

    for index, source in enumerate(packet.source_candidates, start=1):
        source_id = _source_id(source, index)
        markers = _markers(source)
        input_marker_hits = sum(_count_marker(input_text, marker) for marker in markers)
        evidence_ids: list[str] = []
        if markers:
            for evidence in extraction.evidence:
                if any(_contains_marker(evidence.source_text, marker) for marker in markers):
                    evidence_ids.append(evidence.id)
                    assigned_evidence_ids.add(evidence.id)

        covered_in_input = input_marker_hits > 0
        covered_in_evidence = bool(evidence_ids)
        status: Literal[
            "covered",
            "input_only",
            "evidence_only",
            "missing",
            "unconfigured",
        ]
        if not markers:
            status = "unconfigured"
        elif covered_in_input and covered_in_evidence:
            status = "covered"
        elif covered_in_input:
            status = "input_only"
        elif covered_in_evidence:
            status = "evidence_only"
        else:
            status = "missing"

        items.append(
            SourceCoverageItem(
                source_id=source_id,
                title=source.title,
                source_group=source.source_group,
                source_kind=source.source_kind,
                text_markers=markers,
                input_marker_hits=input_marker_hits,
                evidence_ids=evidence_ids,
                evidence_count=len(evidence_ids),
                covered_in_input=covered_in_input,
                covered_in_evidence=covered_in_evidence,
                status=status,
            )
        )

    evidence_ids = [evidence.id for evidence in extraction.evidence]
    return SourceCoverageReport(
        source_count=len(items),
        sources_with_input_markers=sum(1 for item in items if item.covered_in_input),
        sources_with_evidence=sum(1 for item in items if item.covered_in_evidence),
        evidence_count=len(evidence_ids),
        assigned_evidence_count=len(assigned_evidence_ids),
        unassigned_evidence_ids=[
            evidence_id for evidence_id in evidence_ids if evidence_id not in assigned_evidence_ids
        ],
        missing_source_ids=[item.source_id for item in items if item.status == "missing"],
        input_only_source_ids=[item.source_id for item in items if item.status == "input_only"],
        unconfigured_source_ids=[
            item.source_id for item in items if item.status == "unconfigured"
        ],
        items=items,
    )


def _markers(source: SourceCandidate) -> list[str]:
    """Return unique configured markers in deterministic order."""

    seen: set[str] = set()
    markers: list[str] = []
    for marker in source.text_markers:
        cleaned = " ".join(marker.split())
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            markers.append(cleaned)
    return markers


def _source_id(source: SourceCandidate, index: int) -> str:
    """Return a stable source id, falling back to a title slug."""

    if source.source_id and source.source_id.strip():
        return _slug(source.source_id)
    slug = _slug(source.title)
    return slug or f"source_{index}"


def _slug(value: str) -> str:
    """Convert a source label into a stable lowercase identifier."""

    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_")


def _contains_marker(text: str, marker: str) -> bool:
    """Return whether ``marker`` appears in ``text`` case-insensitively."""

    return marker.lower() in text.lower()


def _count_marker(text: str, marker: str) -> int:
    """Count case-insensitive marker occurrences in text."""

    return text.lower().count(marker.lower())

"""Source-packet contracts for process-tracing research design.

The packet records the source base, observability assumptions, rival
interpretations, and remaining source gaps that should govern inference before
the pipeline scores evidence. It is metadata about research design, not proof
that every source has already been collected or extracted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourcePacketError(Exception):
    """Raised when a source-packet file cannot be loaded or validated."""


class SourceCandidate(BaseModel):
    """One proposed or included source for a process-tracing source packet."""

    model_config = ConfigDict(extra="ignore")

    source_id: str | None = Field(
        default=None,
        description=(
            "Optional stable source identifier used for packet coverage, such as "
            "'source_a' or 'brumaire_decree'."
        ),
    )
    title: str = Field(description="Human-readable source title.")
    source_group: str | None = Field(
        default=None,
        description=(
            "Optional analyst-defined group for independence/source-lineage "
            "tracking, such as official proclamations, legislative records, "
            "hostile newspapers, memoirs, or historiography."
        ),
    )
    source_kind: str = Field(
        description=(
            "Broad source kind, such as primary legal text, proclamation, "
            "memoir, newspaper, archive, secondary narrative, or historiography."
        )
    )
    date_coverage: str = Field(description="Date or period covered by the source.")
    locator: str | None = Field(
        default=None,
        description="URL, citation, archive locator, file path, or null when not yet known.",
    )
    provenance_note: str = Field(description="Where this source comes from and why it is admissible.")
    reliability_note: str = Field(description="Biases, limitations, authorship issues, or source-production risks.")
    expected_observability: str = Field(
        description="What traces this source genre should and should not reveal for the research question."
    )
    relevance_to_question: str = Field(description="Why this source helps discriminate among rival explanations.")
    text_markers: list[str] = Field(
        default_factory=list,
        description=(
            "Exact provenance markers expected in the assembled input text and "
            "source-grounded evidence quotes, such as 'Source A' or a citation label."
        ),
    )


class RivalInterpretation(BaseModel):
    """A rival scholarly or source-grounded interpretation to preserve."""

    model_config = ConfigDict(extra="ignore")

    interpretation: str = Field(description="The rival interpretation or mechanism.")
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="Source titles or source classes that support or represent this interpretation.",
    )
    discriminating_implication: str = Field(
        description="What evidence would distinguish this interpretation from important rivals."
    )


class SourceGap(BaseModel):
    """A missing source class or trace that limits the packet."""

    model_config = ConfigDict(extra="ignore")

    missing_source_class: str = Field(description="Concrete source class still missing.")
    why_it_matters: str = Field(description="How the gap could cap inference quality.")
    expected_location: str = Field(description="Where an analyst or agent should look next.")
    priority: Literal["high", "medium", "low"] = Field(description="Collection priority.")


class PreSpecifiedTest(BaseModel):
    """A process-tracing test the source packet expects the later pipeline to run."""

    model_config = ConfigDict(extra="ignore")

    test_name: str = Field(description="Short name for the pre-specified diagnostic test.")
    target_rival_pair: str = Field(
        description="Hypothesis pair or rival interpretations this test should discriminate."
    )
    expected_trace: str = Field(description="Trace that should be observable if the favored mechanism is true.")
    contrary_trace: str = Field(description="Trace or absence that would favor a rival interpretation.")
    source_classes: list[str] = Field(
        default_factory=list,
        description="Source classes where the expected or contrary trace should appear.",
    )


class SourcePacketDraft(BaseModel):
    """Typed draft emitted by the assistant and accepted by the pipeline contract."""

    model_config = ConfigDict(extra="ignore")

    case_name: str = Field(description="Case or event being studied.")
    research_question: str = Field(description="Process-tracing research question.")
    focal_window: str = Field(description="Focal temporal window for the outcome and proximate traces.")
    outcome: str = Field(description="Outcome the source packet is designed to explain.")
    source_candidates: list[SourceCandidate] = Field(
        min_length=1,
        description="Candidate or included sources in the packet.",
    )
    rival_interpretations: list[RivalInterpretation] = Field(
        default_factory=list,
        description="Rival interpretations the packet should preserve before testing.",
    )
    observability_notes: list[str] = Field(
        default_factory=list,
        description="Cross-source notes about what the corpus can and cannot reveal.",
    )
    known_gaps: list[SourceGap] = Field(
        default_factory=list,
        description="Missing source classes or traces that still limit inference.",
    )
    pre_specified_tests: list[PreSpecifiedTest] = Field(
        default_factory=list,
        description="Diagnostic tests pre-specified before likelihood scoring.",
    )
    proposed_next_steps: list[str] = Field(
        default_factory=list,
        description="Concrete next actions for improving the packet or running the pipeline.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Caveats that should cap claims if not resolved.",
    )


class SourcePacketSummary(BaseModel):
    """Compact source-packet metadata stored with pipeline results."""

    case_name: str = Field(description="Case or event being studied.")
    research_question: str = Field(description="Process-tracing research question.")
    focal_window: str = Field(description="Focal temporal window for proximate traces.")
    outcome: str = Field(description="Outcome the packet is designed to explain.")
    source_count: int = Field(description="Number of source candidates or included sources.")
    source_groups: list[str] = Field(description="Distinct analyst-defined source groups.")
    source_kinds: list[str] = Field(description="Distinct source kinds represented.")
    date_coverage: list[str] = Field(description="Distinct date-coverage labels represented.")
    known_gap_count: int = Field(description="Number of known missing-source gaps.")
    high_priority_gap_count: int = Field(description="Number of high-priority missing-source gaps.")
    high_priority_gaps: list[str] = Field(description="Names of high-priority missing source classes.")
    pre_specified_test_count: int = Field(description="Number of pre-specified diagnostic tests.")
    limitations: list[str] = Field(description="Packet-level limitations that should cap claims.")
    source_packet_path: str | None = Field(
        default=None,
        description="Path to the source-packet JSON, when loaded from a file.",
    )


class SourceCoverageItem(BaseModel):
    """Coverage of one source-packet source in the assembled corpus and evidence."""

    source_id: str = Field(description="Stable source identifier for this coverage row.")
    title: str = Field(description="Source title from the packet.")
    source_group: str | None = Field(description="Packet source group, when provided.")
    source_kind: str = Field(description="Packet source kind.")
    text_markers: list[str] = Field(description="Exact provenance markers used for coverage matching.")
    input_marker_hits: int = Field(description="Count of marker occurrences in the input text.")
    evidence_ids: list[str] = Field(description="Evidence IDs whose source text contains a marker.")
    evidence_count: int = Field(description="Number of extracted evidence items linked to this source.")
    covered_in_input: bool = Field(description="True when at least one marker appears in the input text.")
    covered_in_evidence: bool = Field(description="True when at least one evidence quote carries a marker.")
    status: Literal[
        "covered",
        "input_only",
        "evidence_only",
        "missing",
        "unconfigured",
    ] = Field(description="Coverage status for this source.")


class SourceCoverageReport(BaseModel):
    """Deterministic coverage check for packet sources against input and extraction."""

    source_count: int = Field(description="Number of packet sources checked.")
    sources_with_input_markers: int = Field(description="Sources with at least one marker in the input text.")
    sources_with_evidence: int = Field(description="Sources linked to at least one extracted evidence item.")
    evidence_count: int = Field(description="Total extracted evidence items in the final result.")
    assigned_evidence_count: int = Field(description="Evidence items linked to at least one packet source marker.")
    unassigned_evidence_ids: list[str] = Field(description="Evidence IDs with no packet-source marker in source_text.")
    missing_source_ids: list[str] = Field(description="Configured source IDs with no input marker and no evidence.")
    input_only_source_ids: list[str] = Field(description="Sources present in input markers but absent from extracted evidence.")
    unconfigured_source_ids: list[str] = Field(description="Sources without explicit markers, so coverage is weak.")
    items: list[SourceCoverageItem] = Field(description="Per-source coverage rows.")


class SourcePacket(SourcePacketDraft):
    """Validated source-packet contract accepted by the process-tracing pipeline."""

    def to_summary(self, source_packet_path: str | None = None) -> SourcePacketSummary:
        """Return compact metadata suitable for storing in ``result.json``."""

        high_priority_gaps = [
            gap.missing_source_class
            for gap in self.known_gaps
            if gap.priority == "high"
        ]
        return SourcePacketSummary(
            case_name=self.case_name,
            research_question=self.research_question,
            focal_window=self.focal_window,
            outcome=self.outcome,
            source_count=len(self.source_candidates),
            source_groups=_unique_sorted(
                source.source_group
                for source in self.source_candidates
                if source.source_group
            ),
            source_kinds=_unique_sorted(source.source_kind for source in self.source_candidates),
            date_coverage=_unique_sorted(source.date_coverage for source in self.source_candidates),
            known_gap_count=len(self.known_gaps),
            high_priority_gap_count=len(high_priority_gaps),
            high_priority_gaps=high_priority_gaps,
            pre_specified_test_count=len(self.pre_specified_tests),
            limitations=list(self.limitations),
            source_packet_path=source_packet_path,
        )

    def to_prompt_context(self) -> str:
        """Render compact source-packet context for hypothesis generation."""

        source_lines = "\n".join(
            (
                f"- {source.title}; group={source.source_group or 'unspecified'}; "
                f"id={source.source_id or 'unspecified'}; "
                f"kind={source.source_kind}; coverage={source.date_coverage}; "
                f"locator={source.locator or 'not specified'}; "
                f"text_markers={', '.join(source.text_markers) or 'not specified'}; "
                f"observability={source.expected_observability}; "
                f"reliability={source.reliability_note}; "
                f"relevance={source.relevance_to_question}"
            )
            for source in self.source_candidates
        )
        rival_lines = "\n".join(
            (
                f"- {rival.interpretation}; sources={', '.join(rival.supporting_sources) or 'not specified'}; "
                f"discriminator={rival.discriminating_implication}"
            )
            for rival in self.rival_interpretations
        ) or "- None specified."
        gap_lines = "\n".join(
            (
                f"- {gap.priority}: {gap.missing_source_class}; "
                f"why={gap.why_it_matters}; look_next={gap.expected_location}"
            )
            for gap in self.known_gaps
        ) or "- None specified."
        test_lines = "\n".join(
            (
                f"- {test.test_name}; pair={test.target_rival_pair}; "
                f"expected={test.expected_trace}; contrary={test.contrary_trace}; "
                f"sources={', '.join(test.source_classes) or 'not specified'}"
            )
            for test in self.pre_specified_tests
        ) or "- None specified."
        observability = "\n".join(f"- {note}" for note in self.observability_notes) or "- None specified."
        limitations = "\n".join(f"- {limitation}" for limitation in self.limitations) or "- None specified."
        return (
            f"Case: {self.case_name}\n"
            f"Research question: {self.research_question}\n"
            f"Focal window: {self.focal_window}\n"
            f"Outcome: {self.outcome}\n\n"
            "Sources:\n"
            f"{source_lines}\n\n"
            "Rival interpretations to preserve:\n"
            f"{rival_lines}\n\n"
            "Known source gaps:\n"
            f"{gap_lines}\n\n"
            "Pre-specified diagnostic tests:\n"
            f"{test_lines}\n\n"
            "Observability notes:\n"
            f"{observability}\n\n"
            "Packet limitations:\n"
            f"{limitations}"
        )


def load_source_packet(path: Path | str) -> SourcePacket:
    """Load a source packet JSON file or assistant artifact JSON file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SourcePacketError(f"source-packet file not found: {path}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SourcePacketError(f"source-packet file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise SourcePacketError("source-packet JSON must be an object")

    payload = data.get("draft") if "draft" in data else data
    if not isinstance(payload, dict):
        raise SourcePacketError("source-packet artifact must contain an object-valued 'draft'")
    try:
        return SourcePacket.model_validate(payload)
    except Exception as exc:
        raise SourcePacketError(f"source-packet validation failed: {exc}") from exc


def _unique_sorted(values: Iterable[str | None]) -> list[str]:
    """Return unique non-empty string values in deterministic order."""

    unique: set[str] = set()
    for value in values:
        if isinstance(value, str) and value.strip():
            unique.add(value.strip())
    return sorted(unique, key=str.lower)

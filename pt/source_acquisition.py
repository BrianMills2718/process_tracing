"""Evidence-acquisition targets derived from a process-tracing result.

The planner converts inferential weak points in an existing trace into concrete
source-search targets. It does not retrieve sources itself; retrieval belongs in
the script or agent layer so the core analysis contract remains deterministic.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from pt.schemas import AbsenceEvaluation, Hypothesis, ProcessTracingResult
from pt.source_packet import SourceGap, SourceGapDisposition, SourcePacket


TargetKind = Literal[
    "source_gap",
    "damaging_absence",
    "sensitivity_discriminator",
    "driver_corroboration",
]


class AcquisitionTarget(BaseModel):
    """One source-acquisition target for the next research iteration."""

    target_id: str = Field(description="Stable target identifier.")
    kind: TargetKind = Field(description="Why this source target matters.")
    priority_score: int = Field(ge=0, le=100, description="Deterministic priority score.")
    evidence_need: str = Field(description="Missing trace or observation that would clarify inference.")
    inferential_payoff: str = Field(description="How acquiring this evidence would change the trace.")
    target_source_class: str = Field(description="Source class or genre to search.")
    related_hypotheses: list[str] = Field(description="Hypothesis IDs most affected by this target.")
    related_evidence_ids: list[str] = Field(description="Existing evidence IDs this target audits or corroborates.")
    search_queries: list[str] = Field(description="Concrete web-retrieval queries for this target.")
    stop_rule: str = Field(description="Condition for stopping this acquisition attempt.")


class AcquisitionPlan(BaseModel):
    """Ranked evidence-acquisition agenda for one process-tracing result."""

    case_name: str = Field(description="Case name inferred from source packet or research question.")
    rationale: str = Field(description="Why these targets are the next best iteration.")
    targets: list[AcquisitionTarget] = Field(description="Targets sorted by descending priority.")


def build_acquisition_plan(
    result: ProcessTracingResult,
    *,
    source_packet: SourcePacket | None = None,
    max_targets: int = 8,
) -> AcquisitionPlan:
    """Build a ranked acquisition agenda from trace uncertainty and source gaps."""

    if max_targets < 1:
        raise ValueError("max_targets must be >= 1")

    case_name = (
        source_packet.case_name
        if source_packet is not None
        else result.source_packet.case_name
        if result.source_packet is not None
        else _case_from_question(result.hypothesis_space.research_question)
    )
    targets: list[AcquisitionTarget] = []
    targets.extend(_source_gap_targets(result, source_packet, case_name))
    targets.extend(_damaging_absence_targets(result, source_packet, case_name))
    targets.extend(_sensitivity_targets(result, case_name))
    targets.extend(_driver_corroboration_targets(result, case_name))

    deduped = _dedupe_targets(targets)
    kind_rank: dict[TargetKind, int] = {
        "source_gap": 0,
        "damaging_absence": 1,
        "sensitivity_discriminator": 2,
        "driver_corroboration": 3,
    }
    ranked = sorted(
        deduped,
        key=lambda target: (-target.priority_score, kind_rank[target.kind], target.target_id),
    )[:max_targets]
    rationale = (
        "Targets are ranked by claim-scope blockers first, then damaging "
        "absence tests, then posterior/prior sensitivity, then corroboration "
        "of influential driver evidence."
    )
    return AcquisitionPlan(case_name=case_name, rationale=rationale, targets=ranked)


def _source_gap_targets(
    result: ProcessTracingResult,
    packet: SourcePacket | None,
    case_name: str,
) -> list[AcquisitionTarget]:
    if packet is None:
        return []

    disposition_by_gap = {
        disposition.missing_source_class: disposition
        for disposition in packet.source_gap_dispositions
    }
    source_ids_by_gap = {
        source.source_id or source.title: source.title
        for source in packet.source_candidates
    }
    targets: list[AcquisitionTarget] = []
    for index, gap in enumerate(packet.known_gaps, start=1):
        disposition = disposition_by_gap.get(gap.missing_source_class)
        if _gap_is_resolved(disposition):
            continue
        score = 100 if gap.priority == "high" else 82 if gap.priority == "medium" else 55
        if disposition is not None and disposition.status == "partially_mitigated":
            score -= 4
        evidence_need = disposition.expected_trace if disposition else gap.why_it_matters
        related_sources = [
            source_ids_by_gap[source_id]
            for source_id in (disposition.relevant_source_ids if disposition else [])
            if source_id in source_ids_by_gap
        ]
        queries = _gap_queries(case_name, gap, disposition)
        targets.append(
            AcquisitionTarget(
                target_id=f"acq_gap_{index}",
                kind="source_gap",
                priority_score=score,
                evidence_need=evidence_need,
                inferential_payoff=(
                    "Tests whether current claim-scope cap should be lifted, "
                    "kept, or converted into an accepted limitation."
                ),
                target_source_class=gap.missing_source_class,
                related_hypotheses=_all_hypothesis_ids(result),
                related_evidence_ids=[],
                search_queries=queries,
                stop_rule=(
                    "Stop when direct sources are acquired, a credible archive route is "
                    "recorded, or at least three independent searches produce only "
                    "adjacent/retrospective substitutes."
                ),
            )
        )
        if related_sources:
            targets[-1].inferential_payoff += (
                " Existing adjacent source coverage to audit: "
                + "; ".join(related_sources)
                + "."
            )
    return targets


def _damaging_absence_targets(
    result: ProcessTracingResult,
    packet: SourcePacket | None,
    case_name: str,
) -> list[AcquisitionTarget]:
    targets: list[AcquisitionTarget] = []
    source_classes = _packet_source_classes(packet)
    for index, absence in enumerate(
        (item for item in result.absence.evaluations if item.severity == "damaging"),
        start=1,
    ):
        targets.append(
            AcquisitionTarget(
                target_id=f"acq_absence_{index}",
                kind="damaging_absence",
                priority_score=88 if absence.would_be_extractable else 72,
                evidence_need=absence.missing_evidence,
                inferential_payoff=(
                    "Converts an absence-of-evidence penalty into either observed "
                    "counterevidence or a better-scoped absence claim."
                ),
                target_source_class=", ".join(source_classes) or "sources likely to contain the missing trace",
                related_hypotheses=[absence.hypothesis_id],
                related_evidence_ids=[],
                search_queries=_absence_queries(case_name, absence, source_classes),
                stop_rule=(
                    "Stop when the missing trace is found in a source of expected scope "
                    "or when source scope shows the absence should no longer be treated "
                    "as damaging."
                ),
            )
        )
    return targets


def _sensitivity_targets(result: ProcessTracingResult, case_name: str) -> list[AcquisitionTarget]:
    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    runner_up_id = result.bayesian.ranking[1] if len(result.bayesian.ranking) > 1 else None
    if top_id is None or runner_up_id is None:
        return []

    prior_sensitive = (
        result.bayesian.prior_sensitivity is not None
        and not result.bayesian.prior_sensitivity.stable_under_prior_perturbation
    )
    top_sensitivity = next(
        (item for item in result.bayesian.sensitivity if item.hypothesis_id == top_id),
        None,
    )
    range_width = (
        top_sensitivity.posterior_high - top_sensitivity.posterior_low
        if top_sensitivity is not None
        else 0.0
    )
    if not prior_sensitive and (top_sensitivity is None or top_sensitivity.rank_stable and range_width < 0.20):
        return []

    top = _hypothesis_by_id(result, top_id)
    runner_up = _hypothesis_by_id(result, runner_up_id)
    top_label = _hypothesis_label(top, top_id)
    runner_label = _hypothesis_label(runner_up, runner_up_id)
    score = 86 if prior_sensitive else 78
    if range_width >= 0.35:
        score += 6
    return [
        AcquisitionTarget(
            target_id="acq_sensitivity_top_vs_runner_up",
            kind="sensitivity_discriminator",
            priority_score=min(score, 95),
            evidence_need=(
                f"Independent trace that distinguishes {top_label} from {runner_label}."
            ),
            inferential_payoff=(
                "Reduces prior sensitivity and tests whether the current winner is a "
                "real mechanism or an artifact of broad priors/weakly discriminating evidence."
            ),
            target_source_class="independent primary or near-contemporaneous sources",
            related_hypotheses=[top_id, runner_up_id],
            related_evidence_ids=_top_driver_ids(result, top_id),
            search_queries=_discriminator_queries(case_name, top, runner_up),
            stop_rule=(
                "Stop when a source supplies a trace predicted by one mechanism and "
                "unexpected under the rival, or when retrieved sources remain generic "
                "background with no likelihood separation."
            ),
        )
    ]


def _driver_corroboration_targets(result: ProcessTracingResult, case_name: str) -> list[AcquisitionTarget]:
    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    if top_id is None:
        return []
    top_drivers = _top_driver_ids(result, top_id)
    if not top_drivers:
        return []
    evidence_by_id = {evidence.id: evidence for evidence in result.extraction.evidence}
    driver_text = [
        evidence_by_id[evidence_id].description
        for evidence_id in top_drivers
        if evidence_id in evidence_by_id
    ]
    if not driver_text:
        return []
    return [
        AcquisitionTarget(
            target_id="acq_driver_corroboration",
            kind="driver_corroboration",
            priority_score=62,
            evidence_need=(
                "Independent corroboration for top-driver evidence: "
                + "; ".join(driver_text[:3])
            ),
            inferential_payoff=(
                "Tests whether the strongest updates are source-lineage artifacts or "
                "independently observed process traces."
            ),
            target_source_class="independent accounts or primary records not sharing the same source lineage",
            related_hypotheses=[top_id],
            related_evidence_ids=top_drivers,
            search_queries=[
                _clean_query(f"{case_name} {' '.join(driver_text[:2])} primary source"),
                _clean_query(f"{case_name} independent account {' '.join(driver_text[:2])}"),
            ],
            stop_rule=(
                "Stop when top-driver claims are corroborated by an independent source "
                "or when dependence should be strengthened in the Bayesian test."
            ),
        )
    ]


def _gap_queries(
    case_name: str,
    gap: SourceGap,
    disposition: SourceGapDisposition | None,
) -> list[str]:
    trace = disposition.expected_trace if disposition else gap.why_it_matters
    return [
        _clean_query(f"{case_name} {gap.missing_source_class} {gap.expected_location}"),
        _clean_query(f"{case_name} {trace} primary source"),
        _clean_query(f"{case_name} correspondence memoir archive {gap.missing_source_class}"),
    ]


def _absence_queries(
    case_name: str,
    absence: AbsenceEvaluation,
    source_classes: list[str],
) -> list[str]:
    source_hint = " ".join(source_classes[:3])
    return [
        _clean_query(f"{case_name} {absence.missing_evidence}"),
        _clean_query(f"{case_name} {absence.missing_evidence} {source_hint}"),
    ]


def _discriminator_queries(
    case_name: str,
    top: Hypothesis | None,
    runner_up: Hypothesis | None,
) -> list[str]:
    top_terms = _hypothesis_terms(top)
    runner_terms = _hypothesis_terms(runner_up)
    return [
        _clean_query(f"{case_name} {top_terms} {runner_terms} primary source"),
        _clean_query(f"{case_name} {top_terms} {runner_terms} correspondence memoir"),
    ]


def _gap_is_resolved(disposition: SourceGapDisposition | None) -> bool:
    return disposition is not None and disposition.status in {"acquired", "accepted_limit"}


def _packet_source_classes(packet: SourcePacket | None) -> list[str]:
    if packet is None:
        return []
    classes = {
        source.source_group or source.source_kind
        for source in packet.source_candidates
        if source.source_group or source.source_kind
    }
    return sorted(classes, key=str.lower)


def _top_driver_ids(result: ProcessTracingResult, hypothesis_id: str) -> list[str]:
    posterior = next(
        (item for item in result.bayesian.posteriors if item.hypothesis_id == hypothesis_id),
        None,
    )
    return list(posterior.top_drivers) if posterior else []


def _all_hypothesis_ids(result: ProcessTracingResult) -> list[str]:
    return [hypothesis.id for hypothesis in result.hypothesis_space.hypotheses]


def _hypothesis_by_id(result: ProcessTracingResult, hypothesis_id: str) -> Hypothesis | None:
    return next(
        (hypothesis for hypothesis in result.hypothesis_space.hypotheses if hypothesis.id == hypothesis_id),
        None,
    )


def _hypothesis_label(hypothesis: Hypothesis | None, fallback_id: str) -> str:
    if hypothesis is None:
        return fallback_id
    return f"{fallback_id} ({hypothesis.description})"


def _hypothesis_terms(hypothesis: Hypothesis | None) -> str:
    if hypothesis is None:
        return ""
    text = f"{hypothesis.description} {hypothesis.causal_mechanism}"
    words = re.findall(r"\b[^\W\d_][\w'-]{3,}\b", text, flags=re.UNICODE)
    stop = {
        "that",
        "this",
        "with",
        "from",
        "through",
        "because",
        "rather",
        "would",
        "could",
        "hypothesis",
        "mechanism",
    }
    selected: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in stop or lower in selected:
            continue
        selected.append(lower)
        if len(selected) >= 5:
            break
    return " ".join(selected)


def _case_from_question(question: str) -> str:
    words = question.split()
    return " ".join(words[:5]).strip(" ?") or "process tracing case"


def _clean_query(query: str) -> str:
    return " ".join(query.replace("\n", " ").split())


def _dedupe_targets(targets: list[AcquisitionTarget]) -> list[AcquisitionTarget]:
    seen: set[tuple[str, str]] = set()
    deduped: list[AcquisitionTarget] = []
    for target in targets:
        key = (target.kind, target.evidence_need.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    return deduped

"""Typed source-design state for iterative process-tracing source planning.

The source packet defines the initial scope contract. The source-design state
adds the mutable iteration loop around that packet: acquisition actions,
retrieved candidates, review decisions, and gap disposition updates. It stays
distinct from the evidence trace so retrieval can be reviewed before any
candidate is treated as admitted source material.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pt.schemas import ProcessTracingResult
from pt.source_acquisition import (
    AcquisitionAction,
    AcquisitionPlan,
    ActionStatus,
    build_acquisition_plan,
)
from pt.source_packet import (
    SourceGapDisposition,
    SourcePacket,
    SourcePacketDraft,
)


class SourceDesignError(Exception):
    """Raised when a source-design state transition cannot be validated."""


ReviewDecisionType = Literal[
    "admit_as_source",
    "reject_as_adjacent",
    "needs_followup",
    "reject",
]
RetrievedStatus = Literal["pending", "reviewed", "admitted", "rejected", "adjacent", "needs_followup"]


class RetrievedCandidate(BaseModel):
    """A retrieved source candidate awaiting or completing review."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str = Field(description="Stable candidate identifier.")
    action_id: str = Field(description="Acquisition action that produced this candidate.")
    title: str = Field(description="Candidate title.")
    url: str | None = Field(default=None, description="Candidate URL or archive locator.")
    source_kind: str | None = Field(default=None, description="Broad source kind for the candidate.")
    retrieved_status: RetrievedStatus = Field(description="Review state for the candidate.")
    note: str = Field(description="Short review note or retrieval summary.")


class ReviewDecision(BaseModel):
    """A review outcome for one retrieved candidate."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str = Field(description="Reviewed candidate identifier.")
    action_id: str = Field(description="Acquisition action that produced the candidate.")
    missing_source_class: str = Field(
        description="Source class or gap this decision should update."
    )
    decision: ReviewDecisionType = Field(description="Typed review outcome.")
    rationale: str = Field(description="Why this decision was made.")
    admitted_source_id: str | None = Field(
        default=None,
        description="Optional admitted source identifier when the candidate is accepted.",
    )


class SourceDesignState(SourcePacketDraft):
    """Mutable source-design artifact for one case and iteration."""

    iteration: int = Field(
        default=1,
        ge=1,
        description="Monotonic design iteration number for the current source-design cycle.",
    )
    acquisition_actions: list[AcquisitionAction] = Field(
        default_factory=list,
        description="Typed acquisition actions derived from the current trace and packet gaps.",
    )
    retrieved_candidates: list[RetrievedCandidate] = Field(
        default_factory=list,
        description="Retrieved candidates awaiting or completing review.",
    )
    review_log: list[ReviewDecision] = Field(
        default_factory=list,
        description="Durable review history for retrieval candidates.",
    )

    def to_source_packet(self) -> SourcePacket:
        """Return the packet-shaped subset of the design state."""

        return SourcePacket.model_validate(self.model_dump())

    @classmethod
    def from_source_packet(
        cls,
        packet: SourcePacket,
        *,
        acquisition_plan: AcquisitionPlan | None = None,
        iteration: int = 1,
    ) -> SourceDesignState:
        """Create a design state from a validated source packet."""

        design = cls.model_validate({**packet.model_dump(), "iteration": iteration})
        if acquisition_plan is not None:
            design.acquisition_actions = acquisition_plan.to_action_records()
        return design

    @classmethod
    def from_result(
        cls,
        result: ProcessTracingResult,
        *,
        source_packet: SourcePacket,
        max_targets: int = 8,
        iteration: int = 1,
    ) -> SourceDesignState:
        """Create a source-design state from a trace result and source packet."""

        plan = build_acquisition_plan(result, source_packet=source_packet, max_targets=max_targets)
        return cls.from_source_packet(source_packet, acquisition_plan=plan, iteration=iteration)

    def with_acquisition_plan(
        self,
        plan: AcquisitionPlan,
        *,
        status: ActionStatus = "proposed",
    ) -> SourceDesignState:
        """Return a copy of the state with action records replaced from a plan."""

        updated = self.model_copy(deep=True)
        updated.acquisition_actions = plan.to_action_records(status=status)
        return updated

    def record_review_decision(self, decision: ReviewDecision) -> SourceDesignState:
        """Return a copy of the state with a review decision and gap update applied."""

        action = next(
            (item for item in self.acquisition_actions if item.action_id == decision.action_id),
            None,
        )
        if action is None:
            raise SourceDesignError(
                f"unknown acquisition action for review decision: {decision.action_id}"
            )
        if action.target_source_class != decision.missing_source_class:
            raise SourceDesignError(
                "review decision does not match action source class: "
                f"{decision.missing_source_class!r} != {action.target_source_class!r}"
            )

        updated = self.model_copy(deep=True)
        updated.review_log.append(decision)
        updated.source_gap_dispositions = _update_gap_dispositions(
            updated.source_gap_dispositions,
            decision,
            action.target_source_class,
            action.evidence_need,
        )
        if decision.admitted_source_id:
            updated.proposed_next_steps.append(
                f"candidate {decision.candidate_id} reviewed for {decision.missing_source_class}"
            )
        return updated

    def refresh_from_result(self, result: ProcessTracingResult) -> SourceDesignState:
        """Return a copy of the state refreshed from a real pipeline result."""

        updated = self.model_copy(deep=True)
        updated.iteration = self.iteration + 1
        if result.source_packet is not None:
            updated.limitations = list(result.source_packet.limitations)
            updated.source_gap_dispositions = [
                disposition.model_copy(deep=True)
                for disposition in result.source_packet.source_gap_dispositions
            ]
        updated.proposed_next_steps = list(result.synthesis.suggested_further_tests)
        return updated


def build_source_design_state(
    result: ProcessTracingResult,
    *,
    source_packet: SourcePacket,
    max_targets: int = 8,
    iteration: int = 1,
) -> SourceDesignState:
    """Build a source-design state from a result and source packet."""

    return SourceDesignState.from_result(
        result,
        source_packet=source_packet,
        max_targets=max_targets,
        iteration=iteration,
    )


def _update_gap_dispositions(
    dispositions: Iterable[SourceGapDisposition],
    decision: ReviewDecision,
    target_source_class: str,
    evidence_need: str,
) -> list[SourceGapDisposition]:
    """Apply one review decision to the matching gap disposition."""

    updated: list[SourceGapDisposition] = []
    matched = False
    status_by_decision: dict[ReviewDecisionType, str] = {
        "admit_as_source": "acquired",
        "reject_as_adjacent": "partially_mitigated",
        "needs_followup": "unresolved",
        "reject": "unavailable",
    }
    for disposition in dispositions:
        if disposition.missing_source_class != decision.missing_source_class:
            updated.append(disposition)
            continue
        matched = True
        new_disposition = disposition.model_copy(deep=True)
        new_disposition.status = status_by_decision[decision.decision]  # type: ignore[assignment]
        if decision.decision == "admit_as_source" and decision.admitted_source_id:
            if decision.admitted_source_id not in new_disposition.relevant_source_ids:
                new_disposition.relevant_source_ids.append(decision.admitted_source_id)
        if decision.candidate_id not in new_disposition.search_actions:
            new_disposition.search_actions.append(
                f"reviewed {decision.candidate_id}: {decision.decision}"
            )
        new_disposition.expected_trace = new_disposition.expected_trace or evidence_need
        new_disposition.claim_implications = (
            f"{new_disposition.claim_implications} | review: {decision.rationale}"
            if new_disposition.claim_implications
            else decision.rationale
        )
        new_disposition.disposition_reason = decision.rationale
        updated.append(new_disposition)

    if not matched:
        updated.append(
            SourceGapDisposition(
                missing_source_class=decision.missing_source_class,
                status=status_by_decision[decision.decision],  # type: ignore[arg-type]
                relevant_source_ids=[decision.admitted_source_id]
                if decision.admitted_source_id
                else [],
                expected_trace=evidence_need,
                claim_implications=decision.rationale,
                search_actions=[f"reviewed {decision.candidate_id}: {decision.decision}"],
                disposition_reason=decision.rationale,
            )
        )
    return updated

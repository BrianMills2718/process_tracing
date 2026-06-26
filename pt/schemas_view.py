"""View-layer Pydantic models for the stage visual audit panels.

These types are the contracts between ViewRenderer and the workbench UI.
They are derived from the backward runtime pass in Plan 005, Slice 005d —
each field traces back to a contract need, not a schema-first invention.

ViewRenderer projects a raw stage artifact (TestingResult, BayesianResult, etc.)
into a ViewPayload.  The workbench handler serialises the payload to JSON and
the browser renders the corresponding panel.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Matrix panel (stage: test) ────────────────────────────────────────────────

class ClusterOverlay(BaseModel):
    """One dependence cluster for grouping matrix rows visually."""

    cluster_id: str = Field(description="Stable ID for this cluster group.")
    evidence_ids: list[str] = Field(
        description="Evidence IDs that belong to this cluster."
    )
    strength: float = Field(
        description="Scalar dependence strength [0,1] from the TestingResult cluster."
    )
    reason_snippet: str = Field(
        description="Short human-readable reason for the cluster, truncated to 120 chars."
    )


class MatrixRow(BaseModel):
    """One row of the evidence × hypothesis likelihood matrix."""

    evidence_id: str = Field(description="Evidence item ID.")
    lr_vector: dict[str, float] = Field(
        description="Hypothesis-id → relative_likelihood mapping for this evidence item."
    )
    relevance: float = Field(
        description="Relevance score from the TestingResult likelihood entry."
    )
    below_threshold: bool = Field(
        description="True when relevance < 0.4 — LR is forced to 1.0 and cell is greyed."
    )
    diagnostic_type: str = Field(
        description="Van Evera diagnostic label (hoop, smoking_gun, doubly_decisive, straw_in_the_wind) "
        "for the most-discriminating hypothesis on this row."
    )
    justification_snippet: str = Field(
        description="First 140 chars of the LLM justification for this likelihood vector."
    )
    cluster_id: Optional[str] = Field(
        default=None,
        description="Cluster ID this row belongs to, or None if unclustered."
    )
    source_quote_snippet: Optional[str] = Field(
        default=None,
        description="First 120 chars of source_text from the ExtractionResult evidence item."
    )


class MatrixViewPayload(BaseModel):
    """Full payload for rendering the evidence × hypothesis matrix panel."""

    stage_id: Literal["test"] = "test"
    hypotheses: list[str] = Field(
        description="Ordered list of hypothesis IDs (column headers, left to right)."
    )
    rows: list[MatrixRow] = Field(
        description="Evidence rows sorted by max(abs(log(lr))) descending so most-discriminating appear first."
    )
    cluster_overlays: list[ClusterOverlay] = Field(
        description="Dependence cluster groups for row-banding in the matrix."
    )
    below_threshold_count: int = Field(
        description="Count of rows with relevance < 0.4 (forced to uninformative LR=1.0)."
    )
    total_count: int = Field(description="Total evidence item count.")


# ── Support panel (stage: update) ─────────────────────────────────────────────

class SupportBar(BaseModel):
    """One hypothesis's posterior bar with sensitivity band."""

    hypothesis_id: str = Field(description="Hypothesis ID.")
    label: str = Field(
        description="Human-readable hypothesis label from HypothesisSpace."
    )
    posterior: float = Field(description="Computed final posterior (comparative support).")
    posterior_low: float = Field(
        description="Lower bound of the sensitivity range (±50% perturbation of top drivers)."
    )
    posterior_high: float = Field(
        description="Upper bound of the sensitivity range."
    )
    robustness: str = Field(
        description="Mechanical robustness class: 'robust', 'moderate', or 'fragile'."
    )
    rank_stable: bool = Field(
        description="False when this hypothesis's rank changes under driver perturbation."
    )
    top_driver_ids: list[str] = Field(
        description="Top 3 evidence IDs that drive this hypothesis's posterior."
    )


class SupportViewPayload(BaseModel):
    """Full payload for rendering the support bars + sensitivity bands panel."""

    stage_id: Literal["update"] = "update"
    bars: list[SupportBar] = Field(
        description="One bar per hypothesis, sorted by posterior descending."
    )
    prior_sensitivity_stable: bool = Field(
        description="True when the top hypothesis ranking is stable under ±2× prior perturbation."
    )
    perturbation_factor: float = Field(
        description="The prior perturbation factor used (typically 2.0)."
    )
    fragile_warning: bool = Field(
        description="True when any hypothesis with posterior > 0.5 has robustness == 'fragile'. "
        "Triggers the prominent warning banner in the UI."
    )
    rank_instability_warning: bool = Field(
        description="True when any hypothesis has rank_stable == False. "
        "Triggers a separate rank-instability chip."
    )


# ── Provenance panel (stage: synthesize) ──────────────────────────────────────

class ProvenanceRow(BaseModel):
    """One row of the evidence provenance panel — trace from evidence to source."""

    evidence_id: str = Field(description="Evidence item ID.")
    source_quote_snippet: str = Field(
        description="First 120 chars of the evidence source_text from ExtractionResult."
    )
    source_marker: Optional[str] = Field(
        default=None,
        description="Source packet marker (e.g. 'Source B') if the evidence is traceable, else None."
    )
    favored_hypothesis_id: str = Field(
        description="Hypothesis ID with the highest relative_likelihood for this evidence."
    )
    peak_lr: float = Field(
        description="Max relative_likelihood across all hypotheses for this evidence "
        "(positive = favors a hypothesis, near 1.0 = uninformative)."
    )


class ProvenanceViewPayload(BaseModel):
    """Full payload for the evidence provenance panel."""

    stage_id: Literal["synthesize"] = "synthesize"
    rows: list[ProvenanceRow] = Field(
        description="Provenance rows sorted by abs(peak_lr) descending."
    )
    items_with_marker: int = Field(
        description="Count of evidence items traceable to a source packet marker."
    )
    items_total: int = Field(description="Total evidence item count.")


# ── Delta panel (stage: refine) ───────────────────────────────────────────────

class PosteriorShift(BaseModel):
    """Before/after posterior comparison for one hypothesis after refinement."""

    hypothesis_id: str = Field(description="Hypothesis ID.")
    before: float = Field(description="Posterior from pre_refine/bayesian.json.")
    after: float = Field(description="Posterior from bayesian.json after refinement re-run.")
    delta: float = Field(
        description="after − before.  Positive = this hypothesis gained support after refinement."
    )


class DeltaViewPayload(BaseModel):
    """Full payload for the refinement delta board panel."""

    stage_id: Literal["refine"] = "refine"
    new_evidence_count: int = Field(description="Count of new evidence items added by refinement.")
    reinterpreted_count: int = Field(description="Count of reinterpreted evidence items.")
    spurious_count: int = Field(description="Count of evidence items flagged as spurious by refinement.")
    hypothesis_refined_count: int = Field(
        description="Count of hypotheses with at least one refinement action."
    )
    posterior_shifts: list[PosteriorShift] = Field(
        description="Per-hypothesis before/after posterior comparison. "
        "Empty if pre_refine/ artifacts are not available."
    )
    pre_refine_available: bool = Field(
        description="True when pre_refine/bayesian.json is present and shifts are populated."
    )


# ── Union and routing ─────────────────────────────────────────────────────────

ViewPayload = MatrixViewPayload | SupportViewPayload | ProvenanceViewPayload | DeltaViewPayload

STAGE_VIEW_MAPPING: dict[str, str] = {
    "test": "matrix",
    "update": "support",
    "synthesize": "provenance",
    "refine": "delta",
}

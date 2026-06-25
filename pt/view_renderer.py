"""Project raw stage artifacts into view-layer payloads for the workbench UI.

Each `build_*_payload` function reads the persisted JSON artifacts from a run
directory and returns the corresponding ViewPayload. Called by the GET
/api/runs/{run_id}/stages/{stage_id}/artifact endpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from pt.schemas import (
    BayesianResult,
    ExtractionResult,
    HypothesisSpace,
    RefinementResult,
    TestingResult,
)
from pt.schemas_view import (
    ClusterOverlay,
    DeltaViewPayload,
    MatrixRow,
    MatrixViewPayload,
    PosteriorShift,
    ProvenanceRow,
    ProvenanceViewPayload,
    SupportBar,
    SupportViewPayload,
    ViewPayload,
)


def build_view_payload(run_dir: Path, stage_id: str) -> Optional[ViewPayload]:
    """Route stage_id to the appropriate view builder. Returns None for unmapped stages."""
    if stage_id == "test":
        return build_matrix_payload(run_dir)
    if stage_id == "update":
        return build_support_payload(run_dir)
    if stage_id == "synthesize":
        return build_provenance_payload(run_dir)
    if stage_id == "refine":
        return build_delta_payload(run_dir)
    return None


def build_matrix_payload(run_dir: Path) -> MatrixViewPayload:
    """Project TestingResult + HypothesisSpace + ExtractionResult → MatrixViewPayload."""
    testing = TestingResult.model_validate(
        json.loads((run_dir / "testing.json").read_bytes())
    )
    hs = HypothesisSpace.model_validate(
        json.loads((run_dir / "hypothesis_space.json").read_bytes())
    )
    extraction = ExtractionResult.model_validate(
        json.loads((run_dir / "extraction.json").read_bytes())
    )

    hyp_ids = [h.id for h in hs.hypotheses]
    evidence_lookup = {e.id: e for e in extraction.evidence}

    # Build cluster membership map: evidence_id → cluster_N label
    cluster_for: dict[str, str] = {}
    for i, cluster in enumerate(testing.dependence_clusters):
        cid = f"cluster_{i}"
        for eid in cluster.evidence_ids:
            cluster_for[eid] = cid

    rows: list[MatrixRow] = []
    for el in testing.evidence_likelihoods:
        lr_vec = {
            hl.hypothesis_id: hl.relative_likelihood
            for hl in el.hypothesis_likelihoods
        }
        below = el.relevance < 0.4
        max_log_lr = max(
            (abs(math.log(max(v, 1e-9))) for v in lr_vec.values()), default=0.0
        )
        if max_log_lr > math.log(9):
            diag = "doubly_decisive"
        elif max_log_lr > math.log(3):
            diag = "smoking_gun"
        elif max_log_lr > math.log(1.5):
            diag = "hoop"
        else:
            diag = "straw_in_the_wind"

        ev = evidence_lookup.get(el.evidence_id)
        rows.append(
            MatrixRow(
                evidence_id=el.evidence_id,
                lr_vector=lr_vec,
                relevance=el.relevance,
                below_threshold=below,
                diagnostic_type=diag,
                justification_snippet=el.justification[:140],
                cluster_id=cluster_for.get(el.evidence_id),
                source_quote_snippet=ev.source_text[:120] if ev else None,
            )
        )

    # Sort most-discriminating first (largest max |log LR| across the vector)
    rows.sort(
        key=lambda r: max(
            (abs(math.log(max(v, 1e-9))) for v in r.lr_vector.values()), default=0.0
        ),
        reverse=True,
    )

    cluster_overlays = [
        ClusterOverlay(
            cluster_id=f"cluster_{i}",
            evidence_ids=c.evidence_ids,
            strength=c.dependence_strength,
            reason_snippet=c.reason[:120],
        )
        for i, c in enumerate(testing.dependence_clusters)
    ]

    below_count = sum(1 for r in rows if r.below_threshold)
    return MatrixViewPayload(
        hypotheses=hyp_ids,
        rows=rows,
        cluster_overlays=cluster_overlays,
        below_threshold_count=below_count,
        total_count=len(rows),
    )


def build_support_payload(run_dir: Path) -> SupportViewPayload:
    """Project BayesianResult + HypothesisSpace → SupportViewPayload."""
    bayesian = BayesianResult.model_validate(
        json.loads((run_dir / "bayesian.json").read_bytes())
    )
    hs = HypothesisSpace.model_validate(
        json.loads((run_dir / "hypothesis_space.json").read_bytes())
    )

    label_for = {h.id: h.description[:60] for h in hs.hypotheses}
    sens_for = {s.hypothesis_id: s for s in bayesian.sensitivity}

    bars: list[SupportBar] = []
    for hp in sorted(bayesian.posteriors, key=lambda p: p.final_posterior, reverse=True):
        s = sens_for.get(hp.hypothesis_id)
        bars.append(
            SupportBar(
                hypothesis_id=hp.hypothesis_id,
                label=label_for.get(hp.hypothesis_id, hp.hypothesis_id),
                posterior=hp.final_posterior,
                posterior_low=s.posterior_low if s else hp.final_posterior,
                posterior_high=s.posterior_high if s else hp.final_posterior,
                robustness=hp.robustness,
                rank_stable=s.rank_stable if s else True,
                top_driver_ids=hp.top_drivers,
            )
        )

    fragile_warning = any(b.posterior > 0.5 and b.robustness == "fragile" for b in bars)
    rank_instability_warning = any(not b.rank_stable for b in bars)
    ps = bayesian.prior_sensitivity
    return SupportViewPayload(
        bars=bars,
        prior_sensitivity_stable=ps.stable_under_prior_perturbation if ps else True,
        perturbation_factor=ps.perturbation_factor if ps else 2.0,
        fragile_warning=fragile_warning,
        rank_instability_warning=rank_instability_warning,
    )


def build_provenance_payload(run_dir: Path) -> ProvenanceViewPayload:
    """Project TestingResult + ExtractionResult (+ optional source_coverage) → ProvenanceViewPayload."""
    testing = TestingResult.model_validate(
        json.loads((run_dir / "testing.json").read_bytes())
    )
    extraction = ExtractionResult.model_validate(
        json.loads((run_dir / "extraction.json").read_bytes())
    )

    # Optional: map evidence_id → packet source marker from result.json source_coverage
    marker_for: dict[str, str] = {}
    result_path = run_dir / "result.json"
    if result_path.exists():
        try:
            sc = json.loads(result_path.read_bytes()).get("source_coverage") or {}
            for item in sc.get("items", []):
                markers = item.get("text_markers") or [item.get("source_id", "")]
                label = markers[0] if markers else None
                if label:
                    for eid in item.get("evidence_ids", []):
                        marker_for[eid] = label
        except Exception:
            pass

    evidence_lookup = {e.id: e for e in extraction.evidence}

    rows: list[ProvenanceRow] = []
    for el in testing.evidence_likelihoods:
        lr_vals = {hl.hypothesis_id: hl.relative_likelihood for hl in el.hypothesis_likelihoods}
        if lr_vals:
            favored = max(lr_vals, key=lambda k: lr_vals[k])
            peak_lr = lr_vals[favored]
        else:
            favored = ""
            peak_lr = 1.0
        ev = evidence_lookup.get(el.evidence_id)
        rows.append(
            ProvenanceRow(
                evidence_id=el.evidence_id,
                source_quote_snippet=(ev.source_text[:120] if ev else el.evidence_id),
                source_marker=marker_for.get(el.evidence_id),
                favored_hypothesis_id=favored,
                peak_lr=peak_lr,
            )
        )

    rows.sort(
        key=lambda r: abs(math.log(max(r.peak_lr, 1e-9))),
        reverse=True,
    )
    return ProvenanceViewPayload(
        rows=rows,
        items_with_marker=sum(1 for r in rows if r.source_marker is not None),
        items_total=len(rows),
    )


def build_delta_payload(run_dir: Path) -> DeltaViewPayload:
    """Project RefinementResult + optional pre_refine/bayesian.json → DeltaViewPayload."""
    refinement = RefinementResult.model_validate(
        json.loads((run_dir / "refinement.json").read_bytes())
    )

    pre_refine_dir = run_dir / "pre_refine"
    pre_refine_available = (pre_refine_dir / "bayesian.json").exists()
    posterior_shifts: list[PosteriorShift] = []

    if pre_refine_available:
        pre_bay = BayesianResult.model_validate(
            json.loads((pre_refine_dir / "bayesian.json").read_bytes())
        )
        post_bay = BayesianResult.model_validate(
            json.loads((run_dir / "bayesian.json").read_bytes())
        )
        pre_map = {p.hypothesis_id: p.final_posterior for p in pre_bay.posteriors}
        post_map = {p.hypothesis_id: p.final_posterior for p in post_bay.posteriors}
        for hid, before in pre_map.items():
            after = post_map.get(hid, before)
            posterior_shifts.append(
                PosteriorShift(hypothesis_id=hid, before=before, after=after, delta=after - before)
            )
        posterior_shifts.sort(key=lambda s: abs(s.delta), reverse=True)

    hyp_refined_ids = {r.hypothesis_id for r in refinement.hypothesis_refinements}
    return DeltaViewPayload(
        new_evidence_count=len(refinement.new_evidence),
        reinterpreted_count=len(refinement.reinterpreted_evidence),
        spurious_count=len(refinement.spurious_extractions),
        hypothesis_refined_count=len(hyp_refined_ids),
        posterior_shifts=posterior_shifts,
        pre_refine_available=pre_refine_available,
    )

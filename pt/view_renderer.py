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

from pt.schemas import ExtractionResult, HypothesisSpace, TestingResult
from pt.schemas_view import (
    ClusterOverlay,
    MatrixRow,
    MatrixViewPayload,
    ViewPayload,
)


def build_view_payload(run_dir: Path, stage_id: str) -> Optional[ViewPayload]:
    """Route stage_id to the appropriate view builder. Returns None for unmapped stages."""
    if stage_id == "test":
        return build_matrix_payload(run_dir)
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

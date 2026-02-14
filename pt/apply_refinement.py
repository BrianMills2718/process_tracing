"""Apply a RefinementResult delta to extraction and hypothesis data.

Pure Python, no LLM. Returns deep copies — originals are preserved for audit.
"""

from __future__ import annotations

from pt.schemas import (
    CausalEdge,
    Evidence,
    ExtractionResult,
    HypothesisSpace,
    RefinementResult,
)


def apply_refinement(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    refinement: RefinementResult,
    *,
    verbose: bool = True,
) -> tuple[ExtractionResult, HypothesisSpace]:
    """Apply refinement delta, return updated (extraction, hypothesis_space).

    Application order:
    1. Remove spurious evidence items
    2. Remove spurious causal edges
    3. Add new evidence (validate no ID collisions)
    4. Apply reinterpretations (update evidence_type, optionally description)
    5. Add new causal edges
    6. Apply hypothesis refinements
    """
    ext = extraction.model_copy(deep=True)
    hs = hypothesis_space.model_copy(deep=True)

    # Build lookup for quick access
    ev_by_id = {e.id: e for e in ext.evidence}
    h_by_id = {h.id: h for h in hs.hypotheses}

    # 1. Remove spurious evidence
    spurious_ev_ids = {
        s.item_id for s in refinement.spurious_extractions if s.item_type == "evidence"
    }
    if spurious_ev_ids:
        ext.evidence = [e for e in ext.evidence if e.id not in spurious_ev_ids]
        if verbose:
            print(f"  Refinement: removed {len(spurious_ev_ids)} spurious evidence items")

    # 2. Remove spurious causal edges
    spurious_edge_ids = {
        s.item_id for s in refinement.spurious_extractions if s.item_type == "causal_edge"
    }
    if spurious_edge_ids:
        before = len(ext.causal_edges)
        ext.causal_edges = [
            e for e in ext.causal_edges
            if f"{e.source_id}->{e.target_id}" not in spurious_edge_ids
        ]
        removed = before - len(ext.causal_edges)
        if verbose:
            print(f"  Refinement: removed {removed} spurious causal edges")

    # 3. Add new evidence (validate no ID collisions)
    existing_ids = {e.id for e in ext.evidence}
    for ne in refinement.new_evidence:
        if ne.id in existing_ids:
            raise ValueError(f"New evidence ID '{ne.id}' collides with existing evidence")
        ext.evidence.append(Evidence(
            id=ne.id,
            description=ne.description,
            source_text=ne.source_text,
            evidence_type=ne.evidence_type,
            approximate_date=ne.approximate_date,
        ))
        existing_ids.add(ne.id)
    if refinement.new_evidence and verbose:
        print(f"  Refinement: added {len(refinement.new_evidence)} new evidence items")

    # 4. Apply reinterpretations
    # Rebuild lookup after removals/additions
    ev_by_id = {e.id: e for e in ext.evidence}
    for ri in refinement.reinterpreted_evidence:
        ev = ev_by_id.get(ri.evidence_id)
        if ev is None:
            if verbose:
                print(f"  Refinement warning: reinterpretation target '{ri.evidence_id}' not found, skipping")
            continue
        ev.evidence_type = ri.new_type
        if ri.updated_description:
            ev.description = ri.updated_description
    if refinement.reinterpreted_evidence and verbose:
        print(f"  Refinement: reinterpreted {len(refinement.reinterpreted_evidence)} evidence items")

    # 5. Add new causal edges
    for nce in refinement.new_causal_edges:
        ext.causal_edges.append(CausalEdge(
            source_id=nce.source_id,
            target_id=nce.target_id,
            relationship=nce.relationship,
        ))
    if refinement.new_causal_edges and verbose:
        print(f"  Refinement: added {len(refinement.new_causal_edges)} new causal edges")

    # 6. Apply hypothesis refinements
    for hr in refinement.hypothesis_refinements:
        h = h_by_id.get(hr.hypothesis_id)
        if h is None:
            if verbose:
                print(f"  Refinement warning: hypothesis '{hr.hypothesis_id}' not found, skipping")
            continue

        if hr.refinement_type == "merge_suggestion":
            if verbose:
                print(f"  Refinement: MERGE SUGGESTION for {hr.hypothesis_id} — {hr.description}")
                print(f"    (Not auto-applied. Review and merge manually if appropriate.)")
            continue

        if hr.refinement_type == "sharpen_mechanism":
            if hr.updated_causal_mechanism:
                h.causal_mechanism = hr.updated_causal_mechanism
            if verbose:
                print(f"  Refinement: sharpened mechanism for {hr.hypothesis_id}")

        elif hr.refinement_type == "add_prediction":
            h.observable_predictions.extend(hr.new_predictions)
            if verbose:
                print(f"  Refinement: added {len(hr.new_predictions)} predictions to {hr.hypothesis_id}")

        elif hr.refinement_type == "reframe":
            if hr.updated_causal_mechanism:
                h.causal_mechanism = hr.updated_causal_mechanism
            if hr.description:
                h.description = hr.description
            if verbose:
                print(f"  Refinement: reframed {hr.hypothesis_id}")

    return ext, hs

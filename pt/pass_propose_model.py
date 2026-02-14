"""Pass: Propose a causal model from N extraction results (data-driven workflow).

Single LLM call examines all extractions and proposes a CausalModelSpec.
"""

from __future__ import annotations

from pathlib import Path

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import ExtractionResult
from pt.schemas_multi import CausalModelSpec

PROMPTS_DIR = Path(__file__).parent / "prompts"


def propose_causal_model(
    extractions: dict[str, ExtractionResult],
    *,
    model: str | None = None,
) -> CausalModelSpec:
    """Propose a causal model from multiple case extractions.

    Args:
        extractions: Map of case_id -> ExtractionResult.
        model: LLM model override.

    Returns:
        A proposed CausalModelSpec derived from the extraction data.
    """
    # Build case summaries
    case_blocks = []
    for case_id, ext in extractions.items():
        mechanisms = [f"    - {m.id}: {m.description}" for m in ext.mechanisms]
        mech_block = "\n".join(mechanisms) if mechanisms else "    (none)"

        hypotheses = [f"    - {h.id}: {h.description}" for h in ext.hypotheses_in_text]
        hyp_block = "\n".join(hypotheses) if hypotheses else "    (none)"

        edges = [f"    - {e.source_id} -> {e.target_id}: {e.relationship}" for e in ext.causal_edges[:15]]
        edge_block = "\n".join(edges) if edges else "    (none)"

        case_blocks.append(
            f"### {case_id}\n"
            f"Summary: {ext.summary}\n"
            f"Mechanisms:\n{mech_block}\n"
            f"Hypotheses in text:\n{hyp_block}\n"
            f"Causal edges (top 15):\n{edge_block}"
        )
    cases_text = "\n\n".join(case_blocks)

    messages = render_prompt(
        PROMPTS_DIR / "propose_model.yaml",
        num_cases=len(extractions),
        cases_text=cases_text,
    )

    kwargs: dict = {}
    if model is not None:
        kwargs["model"] = model
    result = call_llm(messages[0]["content"], CausalModelSpec, **kwargs)

    # Validate
    errors = result.validate_dag()
    if errors:
        raise ValueError(f"Proposed model has validation errors: {errors}")

    return result

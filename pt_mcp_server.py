#!/usr/bin/env python3
"""
Process Tracing MCP Server

Exposes Van Evera process tracing pipeline as MCP tools.
Wraps pt/pipeline.py for use by Claude Code.
"""

import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Add project root to path for pt package imports
sys.path.insert(0, str(Path(__file__).parent))

mcp = FastMCP("process-tracing")


@mcp.tool()
def run_process_trace(
    text: str,
    model: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Run Van Evera process tracing analysis on a text.

    Performs systematic causal inference using a 4-pass LLM pipeline:
    1. Extract: evidence, actors, events, mechanisms, causal edges
    2. Hypothesize: competing causal explanations with observable predictions
    3. Test: diagnostic tests (hoop, smoking gun, doubly decisive, straw-in-wind)
    4. Synthesize: Bayesian-updated verdicts and analytical narrative

    This is computationally expensive (~2-5 min, multiple LLM calls).
    Use for serious hypothesis testing, not casual queries.

    Args:
        text: The text to analyze (historical account, news article, research report).
              Should be substantial (500+ words) for meaningful analysis.
        model: LLM model to use (default: PT_MODEL env var or gemini-2.5-flash).
               Supports any LiteLLM model ID.
        output_dir: Optional directory path to save results (result.json + report.html)
    """
    from pt.pipeline import run_pipeline

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result = run_pipeline(
        text=text,
        model=model,
        verbose=False,
        review=False,
        output_dir=output_dir,
    )

    # Serialize the ProcessTracingResult
    result_dict = result.model_dump()

    # Save to output_dir if specified
    if output_dir:
        result_path = os.path.join(output_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

    # Return a summary rather than the full result (which can be huge)
    summary = {
        "status": "complete",
        "research_question": result.hypothesis_space.research_question,
        "hypotheses": [
            {
                "id": h.id,
                "description": h.description,
                "source": h.source,
            }
            for h in result.hypothesis_space.hypotheses
        ],
        "bayesian_ranking": result.bayesian.ranking,
        "posteriors": {
            p.hypothesis_id: round(p.final_posterior, 4)
            for p in result.bayesian.posteriors
        },
        "verdicts": [
            {
                "hypothesis_id": v.hypothesis_id,
                "status": v.status,
                "key_evidence_for": v.key_evidence_for[:3] if v.key_evidence_for else [],
                "key_evidence_against": v.key_evidence_against[:3] if v.key_evidence_against else [],
                "robustness": v.robustness,
            }
            for v in result.synthesis.verdicts
        ],
        "narrative_preview": result.synthesis.analytical_narrative[:1000],
        "evidence_count": len(result.extraction.evidence),
        "output_dir": output_dir,
    }

    return json.dumps(summary, indent=2, default=str)


@mcp.tool()
def get_trace_results(output_dir: str) -> str:
    """Read saved process tracing results from a previous run.

    Args:
        output_dir: Path to the output directory containing result.json
    """
    result_path = os.path.join(output_dir, "result.json")
    if not os.path.exists(result_path):
        return json.dumps({"error": f"No result.json found at {result_path}"})

    with open(result_path, "r") as f:
        result = json.load(f)

    # Return the synthesis section (most useful for research)
    synthesis = result.get("synthesis", {})
    return json.dumps({
        "research_question": result.get("hypothesis_space", {}).get("research_question"),
        "verdicts": synthesis.get("verdicts", []),
        "analytical_narrative": synthesis.get("analytical_narrative"),
        "comparative_analysis": synthesis.get("comparative_analysis"),
        "limitations": synthesis.get("limitations"),
        "suggested_further_tests": synthesis.get("suggested_further_tests"),
    }, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()

"""Pass 3b: Absence-of-evidence evaluation (failed hoop tests)."""

from __future__ import annotations

import json

from pt.llm import call_llm
from pt.schemas import (
    AbsenceResult,
    ExtractionResult,
    HypothesisSpace,
    TestingResult,
)


PROMPT = """\
You are looking for "the dog that didn't bark" in a Van Evera process tracing analysis.

The pipeline has already tested hypotheses against evidence that IS in the text. Your job is the complement: find evidence that SHOULD be in the text but ISN'T. In Van Evera's framework, when a hypothesis predicts observable evidence and that evidence is absent, it's a failed hoop test — one of the strongest signals against a hypothesis.

## Goal

For each hypothesis, identify predictions where the extracted evidence contains NOTHING that addresses them. These are gaps where the hypothesis expected to leave a trace in the record but didn't.

The core question for each gap: **given the kind of text this is, would this evidence appear if it existed?** A Wikipedia overview covers major events, legislation, and prominent leaders. It won't contain private correspondence, econometric data, or archival records. Only flag absences the text WOULD contain — otherwise the absence tells you about the text's genre, not the hypothesis's validity.

Think of yourself as a dissertation committee member reading the evidence list and asking: "If this hypothesis is right, where is the evidence for X? The text covers this topic — why isn't X mentioned?" Apply this standard with equal rigor to every hypothesis, whether it appears strong or weak from the evidence. Every hypothesis has at least one prediction the text doesn't address — find it.

The hardest part of this task is avoiding false positives. Before flagging any prediction as absent, guard against three failure modes:

1. **Evidence that partially addresses the prediction already exists.** Scan the extracted evidence list carefully — if any item addresses the prediction even partially, it is NOT absent. The testing pass already handled it. Do not demand more granularity than the text provides.
2. **The text's genre cannot provide this evidence regardless of the hypothesis's truth.** A Wikipedia overview will never contain private correspondence, meeting minutes, or econometric data. If the prediction demands specialized sources, the absence is uninformative — it tells you about the genre, not the hypothesis.
3. **The prediction is counterfactual or comparative, not directly observable.** "If X hadn't happened, Y wouldn't have occurred" is not evidence any text could contain. Skip these entirely.

When you do identify a genuine absence, describe what specific evidence is missing — do not simply restate the prediction. Explain WHY the absence matters for this hypothesis's causal mechanism.

## Severity

- "damaging": The hypothesis's core mechanism requires this evidence. Without it, the causal story has a hole.
- "notable": Important but not essential. Raises questions without being decisive.
- "minor": Peripheral. Worth noting but doesn't change the assessment.

Reserve "damaging" for cases where the hypothesis's central claim is undermined. If extracted evidence partially covers the prediction, the severity is at most "notable."

## Key judgment: `would_be_extractable`

Would this evidence appear in a text of this genre if it existed in the historical record? Set to false if the missing evidence would require specialized sources beyond this text's scope.

## Data

### Text summary (genre and scope)
{text_summary}

### Extracted evidence (what IS in the text)
{evidence_json}

### Hypotheses and their predictions
{hypotheses_json}

### Testing summary (which predictions had relevant evidence)
This shows which predictions were linked to relevant evidence during testing. Use it as a starting point, but always verify against the extracted evidence list — some evidence may address a prediction without being formally linked to it.
{testing_summary_json}
"""


def _build_testing_summary(testing: TestingResult) -> list[dict]:
    """Summarize which predictions had relevant evidence evaluations.

    Deliberately omits hypothesis-level support/opposition counts to avoid
    leaking hypothesis strength to the absence evaluator (prevents confirmation bias).
    """
    summary = []
    for ht in testing.hypothesis_tests:
        pred_hits: dict[str, int] = {}
        for ev in ht.evidence_evaluations:
            if ev.prediction_id and ev.relevance >= 0.6:
                pred_hits[ev.prediction_id] = pred_hits.get(ev.prediction_id, 0) + 1
        summary.append({
            "hypothesis_id": ht.hypothesis_id,
            "predictions_with_relevant_evidence": pred_hits,
        })
    return summary


def run_absence(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    testing: TestingResult,
    *,
    model: str | None = None,
) -> AbsenceResult:
    """Evaluate absence of evidence for all hypotheses."""
    kwargs = {"model": model} if model else {}
    testing_summary = _build_testing_summary(testing)

    return call_llm(
        PROMPT.format(
            text_summary=extraction.summary,
            evidence_json=json.dumps(
                [e.model_dump() for e in extraction.evidence], indent=2
            ),
            hypotheses_json=json.dumps(
                hypothesis_space.model_dump()["hypotheses"], indent=2
            ),
            testing_summary_json=json.dumps(testing_summary, indent=2),
        ),
        AbsenceResult,
        **kwargs,
    )

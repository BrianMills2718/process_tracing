"""Pass 3: Diagnostic tests and evidence evaluation (heart of Van Evera)."""

from __future__ import annotations

import json

from pt.llm import call_llm
from pt.schemas import (
    Evidence,
    ExtractionResult,
    Hypothesis,
    HypothesisSpace,
    HypothesisTestResult,
    TestingResult,
)


PROMPT = """\
You are performing DIAGNOSTIC TESTING for Van Evera process tracing on ONE hypothesis.

## Van Evera's Four Tests
| Test | Necessity | Sufficiency | Passing | Failing |
|------|-----------|-------------|---------|---------|
| Hoop | High | Low | Survives | ELIMINATED |
| Smoking gun | Low | High | STRONGLY confirmed | Not much affected |
| Doubly decisive | High | High | Confirmed + rivals eliminated | ELIMINATED |
| Straw in the wind | Low | Low | Slightly helps | Slightly hurts |

## Step 1: Classify each prediction
For each prediction, reason about necessity and sufficiency, then assign the test type.

## Step 2: Evaluate EVERY evidence item

YOU MUST EVALUATE EVERY SINGLE EVIDENCE ITEM. No skipping.

For each, provide:
- `finding`: "pass", "fail", or "ambiguous"
- `p_e_given_h`: probability if hypothesis is TRUE (0.01–0.99)
- `p_e_given_not_h`: probability if hypothesis is FALSE (0.01–0.99)
- `justification`: 1-2 sentences explaining the specific reasoning

## MANDATORY RULES FOR HONEST PROBABILITY ASSIGNMENT

Rule A — NO DUMPING GROUND FOR RELEVANT EVIDENCE: Among evidence items you rate as `relevance` >= 0.6, you may NOT assign P(E|H)=0.5 and P(E|~H)=0.5 to more than 5 items. For relevant evidence, think harder about direction.

However, evidence with LOW relevance (relevance < 0.6) — such as events from decades before the outcome, or facts about unrelated causal domains — SHOULD be assigned LR near 1.0 (e.g., P(E|H)=0.5, P(E|~H)=0.5). Do NOT force a directional lean on evidence that genuinely cannot discriminate. Population growth figures tell you nothing about whether elites conspired; background economic conditions tell you nothing about whether ideology drove the outcome. Forcing a direction on such evidence creates false compound effects that crush narrowly-correct hypotheses.

Rule B — EVERY HYPOTHESIS HAS WEAKNESSES: You MUST identify at least 5 evidence items where the likelihood ratio is BELOW 1.0 (i.e., p_e_given_not_h > p_e_given_h). No hypothesis is supported by all evidence. If you cannot find 5, you are not thinking critically enough about what alternative hypotheses would predict.

Rule C — VARIED ESTIMATES: Use the full range of probabilities. Your estimates should NOT cluster on a few round numbers. Think about each evidence item individually. 0.35 and 0.72 are valid estimates — not everything is 0.5, 0.6, 0.7, 0.8, or 0.9.

Rule D — EVIDENCE AGAINST: When evidence better supports an ALTERNATIVE hypothesis than this one, the finding should be "fail" and p_e_given_not_h should be HIGHER than p_e_given_h. For example, if the evidence shows economic crisis driving events, and this hypothesis claims ideology was the driver, that evidence weighs against this hypothesis.

Rule E — NO CIRCULAR EVIDENCE:
- If evidence RESTATES or CLOSELY PARAPHRASES the hypothesis, the LR MUST be near 1.0 (set P(E|H) and P(E|~H) to the same value, e.g. 0.50/0.50). A historian's claim that "X caused Y" tells you nothing about whether X actually caused Y — it is the claim itself, not independent evidence for the claim. Set `relevance` to 0.1 for such items.
- Evidence marked as `evidence_type: "interpretive"` (historian arguments, scholarly claims) gets max LR = 5.0. These are scholarly opinions, not smoking guns.
- Only direct empirical evidence (actions, decisions, measurable outcomes) can have LR > 10.

Rule F — SPEAKER-ATTRIBUTED EVIDENCE (for debate/multi-speaker texts):
- When evidence items are attributed to specific speakers, evaluate the FACTUAL CONTENT of the claim, not who said it. A fact is equally informative regardless of which speaker states it.
- Do NOT systematically favor one speaker's evidence over another's. If Speaker A provides 20 items supporting H1 and Speaker B provides 10 items supporting H2, this does NOT mean H1 is twice as likely — it means Speaker A talked more about that topic.
- Points where opposing speakers AGREE on a factual claim are STRONGER evidence than uncontested claims by one speaker.
- A speaker's ASSESSMENT or PREDICTION (e.g., "I think they're losing") is interpretive and gets max LR = 5.0, even if it's attributed to an expert.

## Calibration: LR = p_e_given_h / p_e_given_not_h
- LR > 10: very strong for H (ONLY for direct empirical evidence) | LR 3–10: moderate | LR 1–3: weak
- LR ≈ 1: uninformative
- LR 0.3–1: weak against | LR 0.1–0.3: moderate against | LR < 0.1: strong against

## Relevance scoring

For each evidence evaluation, assign `relevance` (0.0–1.0). This reflects BOTH temporal proximity AND causal-domain fit with the hypothesis:

Temporal dimension:
- Evidence from decades before the outcome: 0.3–0.5
- Evidence from years before the outcome: 0.5–0.7
- Evidence from the same period as the outcome: 0.8–1.0
- Evidence from immediately before/during the outcome: 0.9–1.0

Causal-domain dimension:
- Evidence about the SAME causal domain as the hypothesis (e.g., elite politics for an elite-conspiracy hypothesis): no penalty
- Evidence about a DIFFERENT causal domain (e.g., demographic trends for an elite-conspiracy hypothesis): reduce by 0.2–0.4
- Evidence that is circular (restates the hypothesis): set to 0.1

Use the LOWER of the two dimensions. Evidence that is temporally close but causally irrelevant, or causally relevant but temporally distant, should get a moderate score (0.4–0.6).

## Hypothesis under evaluation

{hypothesis_json}

## ALL evidence items (evaluate EVERY one)

{evidence_json}

## Alternative hypotheses (this is what ~H means — evidence supporting THESE weighs against the hypothesis above)

{all_hypotheses_brief}

## Pairwise discrimination check

For each pair of alternative hypotheses listed above that seem similar to the hypothesis under evaluation, identify at least 3 evidence items where your LR assignments DIVERGE by a factor of 2+. If you cannot find 3 such items, your evaluations are not discriminating enough—go back and think harder about what distinguishes this hypothesis from its closest rival.
"""


def _test_one_hypothesis(
    hypothesis: Hypothesis,
    evidence: list[Evidence],
    all_hypotheses: list[Hypothesis],
    *,
    model: str | None = None,
) -> HypothesisTestResult:
    """Run diagnostic tests for a single hypothesis."""
    kwargs = {"model": model} if model else {}
    brief_hypotheses = [{"id": h.id, "description": h.description} for h in all_hypotheses if h.id != hypothesis.id]
    return call_llm(
        PROMPT.format(
            hypothesis_json=json.dumps(hypothesis.model_dump(), indent=2),
            evidence_json=json.dumps([e.model_dump() for e in evidence], indent=2),
            all_hypotheses_brief=json.dumps(brief_hypotheses, indent=2),
        ),
        HypothesisTestResult,
        **kwargs,
    )


def run_test(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    *,
    model: str | None = None,
) -> TestingResult:
    """Run diagnostic tests for all hypotheses (one LLM call per hypothesis)."""
    results: list[HypothesisTestResult] = []
    evidence = extraction.evidence
    all_hypotheses = hypothesis_space.hypotheses

    for i, h in enumerate(all_hypotheses, 1):
        print(f"  Testing hypothesis {i}/{len(all_hypotheses)}: {h.id}")
        result = _test_one_hypothesis(h, evidence, all_hypotheses, model=model)
        result.hypothesis_id = h.id

        # Report coverage and balance
        n_evals = len(result.evidence_evaluations)
        n_for = sum(1 for ee in result.evidence_evaluations if ee.p_e_given_h > ee.p_e_given_not_h)
        n_against = sum(1 for ee in result.evidence_evaluations if ee.p_e_given_h < ee.p_e_given_not_h)
        n_neutral = n_evals - n_for - n_against
        print(f"    {n_evals}/{len(evidence)} evaluated | for={n_for} against={n_against} neutral={n_neutral}")
        if n_against == 0:
            print(f"    WARNING: zero disconfirming evidence — likely biased evaluation")

        results.append(result)

    return TestingResult(hypothesis_tests=results)

"""Pass 4: Written synthesis and verdicts."""

from __future__ import annotations

import json

from pt.llm import call_llm
from pt.schemas import (
    BayesianResult,
    ExtractionResult,
    HypothesisSpace,
    SynthesisResult,
    TestingResult,
)


PROMPT = """\
You are writing the SYNTHESIS phase of a Van Evera process tracing analysis.

You have the complete results of extraction, hypothesis formulation, diagnostic testing, and Bayesian updating.

## Your task

### 1. Verdicts
For each hypothesis:
- `status`: "strongly_supported", "supported", "weakened", "eliminated", or "indeterminate"
- `key_evidence_for`: Evidence IDs where LR > 1 (supports this hypothesis)
- `key_evidence_against`: Evidence IDs where LR < 1 (weighs against this hypothesis). EVERY hypothesis should have SOME evidence against it unless the case is overwhelming.
- `reasoning`: 2-3 sentences on why this verdict follows from the specific tests that passed/failed
- `steelman`: **MANDATORY for ALL hypotheses, especially eliminated ones.** Write 3-5 sentences presenting the STRONGEST possible case for this hypothesis, as if you were its most passionate advocate. What is the best evidence? What reasoning makes it plausible? Even if the posterior is 0.001, a fair analysis shows the reader why a reasonable person might believe this hypothesis. If you cannot steelman a hypothesis, the hypothesis space is poorly designed.
- `posterior_robustness`: Set to "robust" if the posterior was driven by a few decisive diagnostic tests (smoking guns, hoop failures). Set to "fragile" if the posterior was driven by accumulation of many small LR effects from borderline-relevant evidence — because those small effects could easily go the other way. A hypothesis eliminated by one decisive test is robustly eliminated; a hypothesis eliminated by 30 items each contributing LR=0.8 is fragile.

### 2. Comparative analysis
A substantial paragraph (8-12 sentences) comparing hypotheses:
- Which specific evidence items DISCRIMINATE most between hypotheses? (high LR for one, low for another)
- Which hypotheses make similar predictions and which diverge?
- What are the key "smoking gun" or "hoop" findings that drive the ranking?
- Which hypothesis was most damaged by the evidence and why?

### 3. Analytical narrative
4-6 paragraphs of genuine analysis (NOT a summary of the text):
- What did the process tracing REVEAL that wasn't obvious from reading the text?
- Walk through the evidence trail: which tests were most informative and why?
- Discuss how RELEVANCE WEIGHTING affected the results: which evidence items were down-weighted for being temporally distant or causally off-domain, and how did this change the ranking compared to naive counting?
- Explain where the evidence surprised you—where a hypothesis you might expect to do well was actually undermined
- Reference specific likelihood ratios and what they mean
- Address counterarguments: why might the top-ranked hypothesis still be wrong?
- End with what the analysis actually tells us about causation, not just correlation

### 4. Limitations
Be specific: what evidence was missing? What couldn't be distinguished? Where were probability estimates most uncertain?

### 5. Suggested further tests
Specific evidence that would most change the ranking. Focus on tests that would DISCRIMINATE between the top 2-3 hypotheses.

## Data

### Extraction
{extraction_json}

### Hypothesis space
{hypotheses_json}

### Diagnostic testing
{testing_json}

### Bayesian posteriors
{bayesian_json}
"""


def run_synthesize(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    testing: TestingResult,
    bayesian: BayesianResult,
    *,
    model: str | None = None,
) -> SynthesisResult:
    """Generate final synthesis from all pipeline results."""
    kwargs = {"model": model} if model else {}
    return call_llm(
        PROMPT.format(
            extraction_json=json.dumps(extraction.model_dump(), indent=2),
            hypotheses_json=json.dumps(hypothesis_space.model_dump(), indent=2),
            testing_json=json.dumps(testing.model_dump(), indent=2),
            bayesian_json=json.dumps(bayesian.model_dump(), indent=2),
        ),
        SynthesisResult,
        **kwargs,
    )

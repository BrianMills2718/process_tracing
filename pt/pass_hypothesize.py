"""Pass 2: Build hypothesis space with rivals."""

from __future__ import annotations

import json

from pt.llm import call_llm
from pt.schemas import ExtractionResult, HypothesisSpace


PROMPT = """\
You are performing the HYPOTHESIS FORMULATION phase of Van Evera process tracing.

## Rule 1: SINGULAR research question

The research question MUST identify ONE specific outcome and ask what caused it. NOT a compound question.

BAD: "Why did X radicalize AND conclude with Y?" (two different outcomes)
GOOD: "Why did the Revolution radicalize into the Terror rather than stabilizing as a constitutional monarchy?"
GOOD: "Why did the Directory collapse into Napoleon's coup rather than evolving into a stable republic?"

Pick the outcome that the text's evidence best allows you to adjudicate. The hypotheses must ALL explain THIS SAME outcome via different causal paths.

## Rule 2: HYPOTHESES AS COMPETING "DECISIVE FACTOR" CLAIMS

Each hypothesis must claim: "Factor X was the DECISIVE cause — without X, the outcome would NOT have occurred, even if other contributing factors were present."

This means hypotheses must be in TENSION with each other. If H1 is correct (structural forces were decisive), then H2 (individual agency was decisive) must be WRONG — because H1 claims the outcome would have happened regardless of individual choices. Conversely, if H2 is right, then individuals' choices were necessary and the structural conditions alone were insufficient.

Every hypothesis must:
- Claim to be the NECESSARY and/or SUFFICIENT cause of the specific outcome
- Explain the SAME specific outcome (from the research question)
- Propose a DIFFERENT causal mechanism
- Make predictions that CONTRADICT at least one other hypothesis's predictions

If H1 and H2 can both be fully true without contradiction, they are not competing hypotheses — combine them into one or reframe them as competing claims about which factor was decisive.

## Rule 3: Decompose compound explanations

If the text says "A + B + C caused X," test A, B, and C as separate hypotheses. Which one did the DECISIVE causal work — the one without which X would not have happened? That's what process tracing is for.

## Rule 4: DISTINGUISHABILITY CHECK

For each PAIR of hypotheses, you must be able to name at least THREE pieces of evidence that would support one but NOT the other. If you cannot find 3 discriminating items for a pair, the hypotheses are not distinct enough — MERGE them into one or sharpen the causal mechanism until they diverge.

Each hypothesis's `observable_predictions` must include at least 3 predictions that explicitly name a rival hypothesis they contradict. For example: "If H1 is true but H3 is false, we would expect to see X" or "Unlike H2, this hypothesis predicts Y."

Common traps to AVOID:
- Two hypotheses that both involve the same key actor (e.g., Napoleon) are probably not distinct. The distinction must be in the CAUSAL MECHANISM, not the personnel.
- **CHAIN HYPOTHESES**: If H1 is a precondition for H2, or H2 is the ideological framing of H1's material cause, they are NOT rivals — they are steps in the SAME causal story. Example: "salutary neglect created autonomy expectations" (precondition) + "new taxes triggered resistance" (trigger) + "constitutional principles mobilized opposition" (framing) are THREE FACETS OF ONE EXPLANATION, not three competing hypotheses. MERGE them into one and find genuinely different causal mechanisms.
- **Test**: If all three hypotheses could be TRUE simultaneously without contradiction, they are not rivals. Real rivals are in tension: if elite conspiracy drove the outcome, then popular mobilization did NOT; if structural forces were sufficient, then individual agency was NOT necessary.

## Rule 5: NO TAUTOLOGICAL HYPOTHESES

A hypothesis must explain WHY the crisis resolved in this SPECIFIC way, not merely describe the crisis. Test: if the hypothesis is true by definition given the outcome, it is tautological and useless.

BAD: "The Directory collapsed due to institutional instability" — this DESCRIBES the collapse. It doesn't explain why instability led to a military coup rather than another constitutional revision, a royalist restoration, or a Jacobin revival.
GOOD: "Repeated purges of elected officials destroyed the legislature's legitimacy, making extra-constitutional seizure of power the path of least resistance for ambitious elites."

The fix: replace the label ("instability") with the specific MECHANISM (purges → legitimacy loss → extra-constitutional action).

## Rule 6: NO HYPOTHESIS-FROM-EVIDENCE CIRCULARITY

Do NOT create a hypothesis that directly quotes or closely paraphrases an interpretive evidence item. If the text says "Historian X argues that A caused B," you may NOT use "A caused B" as a hypothesis — because then evaluating that evidence against that hypothesis is circular (the evidence IS the hypothesis).

Instead, decompose the historian's claim into a testable mechanism: What specific observable predictions does "A caused B" make that differ from alternative explanations?

## Rule 7: MUTUAL EXCLUSION SELF-CHECK

Before finalizing your hypotheses, apply this test to EVERY pair:

"If H_A is the decisive cause, does that make H_B unnecessary or wrong?"

- If YES for at least one direction → they are genuine rivals. Keep both.
- If NO in both directions (both can be fully true and decisive simultaneously) → they are NOT rivals. MERGE or reframe.

Examples of genuine rivalry:
- "Structural economic forces made revolution inevitable" vs. "Without Samuel Adams's organizing, grievances would have dissipated" — if structural forces were sufficient, Adams was unnecessary; if Adams was necessary, structural forces alone were insufficient.
- "Popular mobilization from below drove the revolution" vs. "Elite manipulation from above manufactured consent for revolution" — these make opposite predictions about where initiative originated.

Examples of FAKE rivalry (fail the mutual exclusion test):
- "British taxes angered colonists" vs. "Colonists had an ideology of liberty" vs. "Radical leaders organized resistance" — all three can be simultaneously true. They are three links in one causal chain: taxes provided the grievance, ideology framed it, leaders organized it. MERGE into one hypothesis.

## Instructions

1. **Research question**: One specific causal question about one outcome.

2. **Text hypotheses** (2-3): Distinct causal claims from the text. Each is a SINGLE factor, not a laundry list. Set `source` to "text".

3. **Rival hypotheses** (2-3): Genuinely different explanations. Each should predict evidence the text hypotheses would NOT predict. Consider:
   - Structural vs. agency explanations
   - Material vs. ideational causes
   - Top-down (elite) vs. bottom-up (popular) drivers
   - Domestic vs. international factors
   Set `source` to "generated".

   MANDATORY: At least one hypothesis (text or generated) must be an AGENCY hypothesis — specific named individuals making deliberate choices that caused the outcome. Not "the military became important" but "Person X recruited Person Y and executed Plan Z." If the text names an architect of the outcome, that person's deliberate actions must be a hypothesis.

4. For EACH hypothesis (5-6 total):
   - `theoretical_basis`: 1-2 sentences
   - `causal_mechanism`: Specific step-by-step chain (1-2 sentences)
   - `observable_predictions`: 4-6 predictions that DISTINGUISH this from others:
     * What would we see if THIS is true but OTHERS are false?
     * What should be ABSENT if this is true?
     * What QUANTITATIVE pattern would we expect?

Use IDs: h1, h2... and pred_h1_01, pred_h1_02...

## Extraction results

{extraction_json}
"""


def run_hypothesize(extraction: ExtractionResult, *, model: str | None = None) -> HypothesisSpace:
    """Build hypothesis space from extraction results."""
    kwargs = {"model": model} if model else {}
    return call_llm(
        PROMPT.format(extraction_json=json.dumps(extraction.model_dump(), indent=2)),
        HypothesisSpace,
        **kwargs,
    )

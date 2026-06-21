# Output Quality Rubric

This rubric grades a generated process-tracing result and report from an
academic process-tracing standpoint. It grades both the visible report and the
underlying evidentiary basis. A polished report cannot receive an `A` if the
input corpus, diagnostic tests, temporal sequence, or hypothesis design would
not survive a PhD-level methods critique.

## Acceptance Criteria

- `A`: 90-100. PhD-review-ready under the available corpus. No active academic
  evidence cap. Claims remain cautious, but the source base, discriminators,
  temporal sequence, and diagnostic tests are strong enough for scholarly review.
- `B`: 80-89. Strong exploratory output. The report is clear, but one academic
  weakness still prevents publication-strength causal inference.
- `C`: 70-79. Useful for hypothesis generation and audit planning. The structure
  is sound, but the evidentiary basis is too thin, weak, broad, or background
  heavy for a PhD-level causal claim.
- `D`: 60-69. Major weaknesses. The output may be readable but has serious
  method risks in source scope, diagnosticity, temporality, or synthesis.
- `F`: below 60. Not usable.

## Rubric by Pipeline Output

| Category | Points | A-grade standard |
|---|---:|---|
| Contract integrity | 15 | Extraction, hypotheses, testing vectors, dependence clusters, Bayesian posteriors, and report artifacts are structurally consistent; no evidence, hypothesis, or vector is silently dropped. |
| Comparative-support discipline | 15 | Bayesian support is framed as comparative over the listed hypothesis set, not truth probability; synthesis verdicts are calibrated to posteriors or explicitly labeled as secondary mechanisms. |
| Temporal and causal proximity | 15 | The report gives a focal year, proximate/background evidence shares, background top-driver warnings, and a chronological event/evidence timeline before network interpretation. |
| Robustness and fragility | 15 | High-support fragile winners are not treated as settled results; sensitivity interval, rank stability, prior stability, and fragility warnings are visible. |
| Evidence weighting and dependence | 15 | Raw counts are distinguished from effective evidence; dependence clusters, weak-evidence accumulation, and source-lineage risks are visible. |
| Hypothesis discrimination | 10 | Broad or absorptive winners are flagged; rival hypotheses have explicit discriminators rather than overlapping umbrella mechanisms. |
| Source-scope and absence calibration | 10 | Absence findings are tied to source genre and archive expectations; broad overview texts cannot support strong damaging absence claims without caveats. |
| Report usability and safety | 5 | HTML is safe, top-ranked hypotheses are visually connected to support, hidden isolated nodes are disclosed as not discarded, and PhD-level recommendations are visible in the report. |

## Academic Evidence Caps

These caps override the surface score. A report that satisfies every display
requirement can still be capped when the underlying scholarly evidence is weak.

| Cap | Trigger | Required improvement |
|---:|---|---|
| 78 | Single-source or single-text limitation acknowledged | Add a source packet with primary documents, rival secondary accounts, and source metadata. |
| 76 | No evidence item reaches moderate diagnostic strength | Pre-specify hoop, smoking-gun, and discriminating tests; seek traces unlikely under rivals. |
| 80 | Less than 20% of evidence is proximate to the focal outcome | Collect dated evidence from the decisive decision window and score it separately. |
| 82 | Top drivers are background-context items | Separate enabling conditions from proximate mechanism traces. |
| 84 | High-support winner is fragile | Treat as provisional ranking; seek fewer but stronger discriminating traces. |
| 84 | Winning hypothesis is broad/absorptive | Split into narrower mechanisms or add pairwise discriminators. |
| 86 | Synthesis verdicts overstate low posteriors | Recalibrate verdict labels to comparative support. |
| 88 | More than half of extracted evidence has no displayed graph edge | Classify unlinked evidence as background, discarded, or pending-test evidence. |

## Scope Limits

The audit grades academic readiness of the generated output, not historical
truth. With a limited input text, the optimal output may be a clear `C`: useful
for hypothesis generation, explicit about its limits, and precise about what
must be collected next. Do not raise the grade by hiding limitations; raise it
by improving source scope, diagnostic evidence, temporal process evidence, and
hypothesis discrimination.

## Iteration Protocol

1. Run `make audit-result RESULT=RESULT REPORT=REPORT`.
2. Read both category recommendations and academic evidence caps.
3. Fix report-only failures first when the JSON already contains the needed
   information.
4. If a cap reflects missing evidence, stop trying to polish the report and
   collect the recommended source material.
5. Rerun extraction, testing, Bayesian update, report generation, and the audit.
6. Stop only when the output reaches `A` or the remaining cap requires external
   evidence not present in the current input corpus.

# Output Quality Rubric

This rubric grades a generated process-tracing result and report. An `A` means
the output is review-ready: it may still be uncertain, but its uncertainty,
scope limits, and causal-evidence weaknesses are visible enough that a reader is
not misled by the ranking.

## Acceptance Criteria

- `A`: 90-100. Review-ready. Major causal, temporal, robustness, and reporting
  caveats are surfaced in the report.
- `B`: 80-89. Useful but not review-ready. One or more important caveats are
  present only in raw data or prose, not clearly surfaced.
- `C`: 70-79. Analytically risky. The output may have valid structure but can
  mislead on timing, posterior meaning, or evidence strength.
- `D`: 60-69. Major weaknesses. The reader cannot trust the headline without
  manually auditing the JSON.
- `F`: below 60. Not usable.

## Rubric

| Category | Points | A-grade standard |
|---|---:|---|
| Contract integrity | 15 | Evidence, hypotheses, vectors, clusters, and report artifacts are structurally consistent; IDs are exact; no silent drops. |
| Comparative-support discipline | 15 | Report labels support as comparative, not truth probability; verdict labels are calibrated to posterior support or explicitly caveated as secondary mechanisms. |
| Temporal and causal proximity | 15 | Report shows focal year, proximate evidence share, background evidence share, and whether top drivers are proximate to the outcome. |
| Robustness and fragility | 15 | High-support fragile results are prominently warned; sensitivity interval, rank stability, and prior stability are visible. |
| Evidence weighting and dependence | 15 | Report distinguishes raw counts from effective/weighted evidence; dependence clusters and weak-evidence accumulation are visible. |
| Hypothesis discrimination | 10 | Broad or absorptive winning hypotheses are flagged when they overlap rival mechanisms; remaining discriminators are named. |
| Source-scope and absence calibration | 10 | Damaging absence findings are caveated when the source is broad and may not contain the missing micro-evidence. |
| Report usability and safety | 5 | HTML is safe to open, collapsibles work with model-provided IDs, the top-ranked hypothesis is visually connected to its supporting evidence, and the audit is visible without reading JSON. |

## Scope Limits

The audit grades reporting discipline and internal evidence handling. An `A`
means the report is review-ready, not that the causal claim is historically true
or externally validated. Source quality, missing archives, and hypothesis-space
coverage still require substantive review.

## Iteration Protocol

1. Run `make audit-result RESULT=RESULT REPORT=REPORT`.
2. Fix the highest-point failing category first.
3. Regenerate the report from the same `result.json` when the fix is report-only.
4. Rerun the audit.
5. Stop only when the grade is `A` or a genuine blocker prevents further progress.

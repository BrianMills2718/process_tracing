# Output Quality Rubric

This rubric grades a generated process-tracing result and report from an
academic process-tracing standpoint. It separates two judgments that must not be
collapsed: (1) the critique of the analysis **given the accepted sources**, and
(2) the claim-scope caveat about whether those sources are enough for broader
publication-strength inference. A polished report cannot receive an `A` if the
accepted-source analysis, diagnostic tests, temporal sequence, or hypothesis
design would not survive a PhD-level methods critique; unresolved source gaps
cap claims beyond the current corpus rather than making the conditional critique
meaningless.

## Two Audit Tracks

Every audit reports two related but distinct scores:

- **Given-source grade**: quality of the process-tracing analysis conditional on
  the accepted source material. The grader must see the source packet and
  packet-source coverage summary before applying this grade.
- **Claim-scope grade**: how far the resulting claims can travel beyond the
  accepted corpus. Unresolved source gaps lower this track, but they are not by
  themselves failures of the given-source critique.

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
| Source-scope and absence calibration | 10 | Absence findings are tied to source genre and archive expectations; accepted source packets and packet-source coverage are visible in the report, and broad overview texts cannot support strong damaging absence claims without caveats. |
| Report usability and safety | 5 | HTML is safe, top-ranked hypotheses are visually connected to support, hidden isolated nodes are disclosed as not discarded, and PhD-level recommendations, evidence triage, and optimality gate are visible in the report. |

## Academic Evidence Caps

These caps override the surface score. A report that satisfies every display
requirement can still be capped when the underlying scholarly evidence is weak.
Diagnostic strength is measured as the post-cap, relevance-discounted pairwise
LR spread for each evidence item (`max(LR) / min(LR)` across hypotheses), not as
one hypothesis's centered LR against the vector mean. This keeps the gate aligned
with Van Evera-style discrimination between rivals.

Source-scope caps are claim-strength caps, not ordinary critique rows. They say
"do not generalize beyond the accepted corpus yet"; they should not be rendered
as if the given-source analysis failed merely because additional sources would
improve publication readiness. Packet-source coverage is different: if an
accepted packet source is not represented in extracted evidence, the grader does
not know that source material was actually used, so that remains a given-source
critique cap.

| Cap | Trigger | Required improvement |
|---:|---|---|
| 78 | Single-source or single-text limitation acknowledged and no source packet is stored | Add a source packet with primary documents, rival secondary accounts, source metadata, observability notes, and source gaps. |
| 82 | Source packet exists but is thin, has high-priority gaps, or carries unresolved packet limitations | Extend or repair the packet before publication-strength causal claims. |
| 82 | Source packet exists but packet-source coverage is missing, unconfigured, or incomplete | Add exact text markers, assemble the corpus from packet sources, and verify every accepted packet source appears in extracted evidence or is explicitly dispositioned as a gap. |
| 88 | Source packet exists but synthesis/report still describe stale single-source limits | Regenerate or repair synthesis so source-scope limitations are based on the accepted packet. |
| 76 | No evidence item reaches moderate diagnostic strength | Pre-specify hoop, smoking-gun, and discriminating tests; seek traces unlikely under rivals. |
| 80 | Less than 20% of evidence is proximate to the focal outcome | Collect dated evidence from the decisive decision window and score it separately. |
| 82 | Top drivers are background-context items | Separate enabling conditions from proximate mechanism traces. |
| 84 | High-support winner is fragile | Treat as provisional ranking; seek fewer but stronger discriminating traces. |
| 84 | Winning hypothesis is broad/absorptive | Split into narrower mechanisms or add pairwise discriminators. |
| 86 | Synthesis verdicts overstate low posteriors | Recalibrate verdict labels to comparative support; this should normally be caught by `pt/verdict_calibration.py`. |
| 88 | More than half of extracted evidence has no displayed graph edge | Classify unlinked evidence as background, discarded, or pending-test evidence. |

## Scope Limits

The audit grades academic readiness of the generated output, not historical
truth. With a limited input text, the optimal output may be a clear `C`: useful
for hypothesis generation, explicit about its limits, and precise about what
must be collected next. Do not raise the grade by hiding limitations; raise it
by improving source scope, diagnostic evidence, temporal process evidence, and
hypothesis discrimination.

## Optimality Gate

Each audit must say whether the output is optimal for the accepted-source
critique and separately whether claim-scope caps remain:

- `optimal_given_accepted_sources`: no conditional caps remain; the output can
  be read as PhD-review-ready conditional on the accepted source material.
- `not_optimal` + `repair_report_or_model`: the JSON contains enough information,
  but the report/model presentation or synthesis needs correction.
- `not_optimal` + `design_stronger_conditional_tests`: the given-source analysis
  needs stronger discriminators, dependence review, or pre-specified tests.
- `claim_scope_acceptance_criteria`: source gaps or packet limitations that cap
  broader claims beyond the accepted corpus.

The HTML report must include an evidence triage table that classifies extracted
items as top drivers, displayed discriminators, background weak signals,
low-relevance items, near-neutral inventory, or tested-but-unlinked evidence.

## Iteration Protocol

1. Run `make audit-result RESULT=RESULT REPORT=REPORT`.
2. Read `source_material_context`, then read `conditional_caps` and
   `claim_scope_caps` separately.
3. Fix report-only failures first when the JSON already contains the needed
   information.
4. If a conditional cap reflects weak discriminators or fragility, design
   stronger tests/traces within the accepted-source critique. If a claim-scope
   cap reflects missing sources, keep current conclusions conditional and only
   collect or disposition those sources before broader claims.
5. Rerun extraction, testing, Bayesian update, report generation, and the audit.
6. Stop only when the output reaches `A` or the remaining cap requires external
   evidence not present in the current input corpus.

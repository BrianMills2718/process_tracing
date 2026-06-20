# Inference-Core Rebuild — Overnight Sprint Tracker

**Started:** 2026-06-19 (overnight autonomous run)
**Branch:** `feat/inference-core-rebuild`
**Mode:** NEVER STOP — execute all slices without pausing; stop only for an
irreversible+shared-state action or a genuine architectural fork not pre-decided
here. Circuit breaker: 3 failed attempts on one problem → log, move on.

Re-read this file after any compaction. Spec: `PROJECT_THEORY_AND_GOALS.md`.
Build order: `BUILDPLAN_pragmatic_process_tracing.md`. Methodology depth:
`WHITEPAPER_optimal_automated_process_tracing.md`.

## Mission

Complete the single-text inference-core rebuild: replace the pseudo-Bayesian
testing/update (per-hypothesis two-way LRs, uniform prior, point posteriors) with
the coherent design — per-evidence likelihood vectors → coherent posteriors →
bands/intervals → researcher priors + sensitivity — validated live on the French
Revolution text. Reuse extraction, hypothesize, cross-case CQ, report shell,
harness.

## Decisions locked (do NOT re-litigate overnight)

- Full **per-evidence likelihood-vector** design (not the post-process variant).
- **Live validation approved** (spends LLM budget on the user's keys; cost not a concern).
- Default model `gemini-2.5-flash`.
- One LLM "likelihood matrix" call per run: given all hypotheses + all evidence,
  return for each evidence item a vector of relative likelihoods across hypotheses
  + a Van Evera diagnostic label + a relevance score + justification.
- Posterior: `O_ij = (prior_i/prior_j) · Π_m (relL_{m,i}/relL_{m,j})`, normalized.
  Uninformative/circular evidence ⇒ flat vector (no update); relevance < 0.4 ⇒ flatten.
- Keep an LR-cap equivalent: clamp the per-item max/min relative-likelihood ratio.
- Van Evera labels retained as a presentation layer.
- Slices 5 (evidence-graph clustering) and 6 (critic pass) remain DEFERRED per the
  build plan unless everything else is done with time to spare.

## Acceptance criteria

- [ ] Each slice: unit tests pass; `make check` green except the known pre-existing
      markdown-links failure (missing `enforced_planning.worktree_paths`, fixed in
      unmerged PR #2 — not our concern here).
- [ ] Slice 3: likelihood matrix elicited; coherent posteriors; coherence unit test;
      **live run** on french_revolution.txt yields sane vectors + defensible ranking.
- [ ] Slice 4: likelihoods as bands; posterior **interval + rank-stability** reported;
      unit tests; report renders intervals.
- [ ] Slice 2: researcher-settable priors (default uniform, recorded) + prior-sensitivity;
      unit tests.
- [ ] Full pipeline runs end-to-end live; HTML report renders; result.json valid.
- [ ] Ledger in PROJECT_THEORY_AND_GOALS.md §6 updated; each slice committed + pushed.

## Slice status

- [x] Spec doc (PROJECT_THEORY_AND_GOALS.md) — committed
- [x] LLM boundary → call_llm_structured — committed + live-smoke validated
- [x] Slice 3 — coherent likelihood vector (MVP) — deterministic green + live-validated
- [x] Slice 4 — posterior interval + rank-stability + prior-stability surfaced in exec
      summary (shrunk per cutter: reuse existing sensitivity intervals; full band
      elicitation + joint Monte Carlo deferred as optimum-only). Truth-in-labeling
      applied here too (rebuilt report says "Support", not "Posterior probability").
- [x] Slice 2 — researcher priors + prior-sensitivity (CLI --priors, pipeline threading, PriorSensitivity)
- [x] Integration — Slice 2/4 report rendering validated on the live french_rev vectors
      (no new LLM calls): prior_sensitivity populated, interval/badges render, all checks pass
- [x] Ledger + docs update; PR opened (#6)

## Adversarial code review (round 1) — fixes applied

A fresh agent reviewed the rebuild and found the coherence claim was FALSE: the
update was still per-hypothesis binary-odds-then-normalize with per-step clamping
(order-dependent: 5 pro + 5 anti → 0.001; reversed → 0.999). Fixed + other defects:

- **DEFECT (headline): real joint update.** Rewrote to log-space softmax
  (log w_i = log prior_i + Σ log LR_i; softmax). Order-invariant (unit-tested +
  validated on real data), no per-step clamping. `_joint_posteriors` helper; trail
  records the joint normalized posterior after each item.
- **DEFECT: cap was per-LR (400:1 pairwise).** Now caps per-item PAIRWISE spread to
  LR_CAP (each centered log-LR clamped to ±0.5·log CAP).
- **DEFECT: silent vector incompleteness.** `pass_test` now fails loud on missing/
  duplicate/unknown hypothesis ids per item and missing/extra/duplicate evidence ids.
- **DEFECT: silent prior handling.** `run_bayesian_update` now raises on unknown/
  missing/non-positive priors.
- **DEFECT: partial truth-in-labeling.** Remaining report/pipeline "probability"
  labels → "support".
- GAP (labeled honestly): sensitivity is local (±50% top drivers); prior-sensitivity
  is one-at-a-time — tooltips say so.

**KEY EMPIRICAL FINDING (the fix surfaced it):** with the *correct* joint update, the
76-item French-Rev run gives h1 = **1.000** (the buggy clamping was masking this at
0.948). This is Naive-Bayes overconfidence from unmodeled evidence **dependence** —
exactly what the white paper warns about, and what dependence-clustering (deferred
Slice 5) fixes. **This run shows Slice 5 is now REQUIRED, not optional**, for usable
single-text magnitudes. Added an honest overconfidence banner (top>0.99 + fragile →
"read as ranking, not calibrated probability"). Order-invariance + rank are still
sound; only the magnitude is unreliable until clustering lands.

Verified: 96 passed / 1 skipped, mypy clean, 100% compliance.

Still open from the review: residual **H0** not implemented (estimand incomplete —
contract violation, was already marked Partial); white-paper/build-plan docs live on
PR #4's branch, not this one (broken cross-branch reference in PROJECT_THEORY).

## Slice 5 (dependence clustering) — done, with an honest limit

Implemented: matrix call also returns `dependence_clusters`; `bayesian` collapses
each cluster to one effective observation (log-average); `pass_test` validates
clusters fail-loud; evidence ids ASCII-sanitized at extraction (fixed a real
fail-loud catch: `evi_levée_…` mangled to a newline in the round-trip).

**Live finding (important):** on french_revolution the LLM clustered only 13/74
items (6 clusters), so the top support is still **0.997** — clustering removes
*duplicate* double-counting but NOT the structural Naive-Bayes overconfidence from
the ~67 non-duplicate-but-correlated items. The overconfidence **banner fires**
(honest: "read as ranking"). The real magnitude fix is the optimum's
partial-pooling / cross-cluster shared-error model (still deferred). Net state:
coherent + duplicate-safe + honestly-flagged; magnitude still rank-only on dense
texts. Order-invariance and ranking are sound. make check green; 96 passed.

## Residual H0 — done

Opt-in residual hypothesis (`RESIDUAL_ID = "H0_residual"`, `include_residual=True`
from the pipeline): exhaustive partition with a reserve prior + flat likelihood, so
the system isn't forced to crown a listed story. Default off (keeps unit math/tests
explicit); pipeline enables it; report renders it. Validated on the live french_rev
vectors (H0 present in posteriors/ranking/report). 100 passed; make check green.

## SPRINT COMPLETE (scoped slices)

All non-deferred slices done, verified, and live-validated. Branch
`feat/inference-core-rebuild` → PR #6. `make check`: 86 passed / 1 skipped, mypy
clean, 100% compliance (only the pre-existing markdown-links module fails).
Deferred (optimum-scope): band elicitation + joint MC, evidence-graph clustering,
critic/auditor + ablation, trace-production model, provenance guards.

## Slice 3 design (locked)

- Testing input becomes per-evidence vectors: `EvidenceLikelihood{evidence_id,
  hypothesis_likelihoods:[{hypothesis_id, relative_likelihood>0, diagnostic_type}],
  relevance, justification}`; `TestingResult{evidence_likelihoods, prediction_classifications}`.
  Drop `EvidenceEvaluation`/`HypothesisTestResult`.
- `bayesian`: per item m, per hyp i, derive `LR_{m,i} = relL_i / geomean_j(relL_j)`,
  clamp to [LR_FLOOR, LR_CAP], relevance-discount (`LR**relevance`; <0.4 ⇒ 1.0).
  Joint update `post_i ∝ prior_i · Π_m LR_{m,i}`, normalize. **Reuse** EvidenceUpdate/
  HypothesisPosterior/BayesianResult, robustness, top_drivers, sensitivity unchanged
  (they operate on the derived per-hyp LRs).
- `pass_test`: ONE matrix call (all hyps + all evidence) → TestingResult.
- `report`/`pass_absence`/`pipeline`: adapt to new structure (show vector + derived LR).
- Tests: port test_pt_bayesian/test_pt_schemas/test_pipeline_integration fixtures to vectors.
- Atomic commit (central structure change); make check green before commit; then live run.

## Running log

- 2026-06-19: boundary rebuilt + live smoke green (1 structured call, 2.8s). Keys
  present (Gemini/OpenAI). Mapped old-testing consumers. Starting Slice 3 schema.
- 2026-06-19: Slice 3 DONE. Rewrote schemas (EvidenceLikelihood vectors), bayesian
  (geomean-derived per-hyp LRs + joint normalization, reusing EvidenceUpdate/
  HypothesisPosterior/robustness/top_drivers/sensitivity), pass_test (one matrix
  call), pass3_test.yaml, pass_absence, pipeline, report (test matrix shows
  vector+derived LR). Ported all 3 test files to vectors. make check: 77 passed,
  mypy clean, 100% compliance (only pre-existing markdown-links module fails).
  LIVE RUN on french_revolution.txt: 76 evidence × 4 hyps in one call (99s),
  complete coherent vectors, low-relevance items flattened correctly, h1=0.948
  (fragile — honestly flags accumulation-from-weak), sensitivity h1 [0.75,0.99]
  rank-stable. result.json + report.html valid. NOTE: Slice 1 (truth-in-labeling)
  is a separate PR off master, not on this branch — caveat text will appear at
  merge. Next: Slice 4 (bands).

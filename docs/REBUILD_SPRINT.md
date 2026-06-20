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

## Adversarial code review (round 2) — fixes applied

- **DEFECT: fake-counterevidence quota.** Prompt Rule B ("AT LEAST 5 items a rival
  must lead") removed — it manufactured weaknesses and was impossible for <5 items.
  Replaced with honest "report one-sidedness as one-sided." Discrimination check
  de-quota'd too.
- **DEFECT: interpretive cap not enforced.** Prompt said interpretive max:min ≤5 but
  code applied 20. Now enforced: `INTERPRETIVE_LR_CAP=5`, per-item `caps` threaded
  through `lr_matrix`/`run_bayesian_update`; pipeline sets it from evidence_type.
- **DEFECT: diagnostic_type not an enum** → `DiagnosticType` Literal.
- **DEFECT: relative_likelihood accepted inf/nan** → `allow_inf_nan=False`.
- **DEFECT: duplicate upstream ids** → model validators on ExtractionResult
  (evidence) and HypothesisSpace (hypotheses).
- **GAP: synthesis truth-labeling** → pass4 prompt now says "comparative support",
  not "posterior probability", with a degenerate-support caveat.
- **GAP: stale sprint doc** → the Slice-3 design block updated to softmax + pairwise cap.
- Tests added for every code-enforced fix. 106 passed; make check green.

## Partial-pooling dependence model — implemented + mechanism-validated

Generalized cluster full-collapse to **partial pooling**: `EvidenceCluster.dependence_strength`
(ρ∈[0,1]); `_pool_clusters` computes effective count `k_eff = 1 + (k-1)(1-ρ)` and the cluster
contributes `exp(k_eff · mean_log_LR)` per hypothesis (ρ=1 collapse, ρ=0 independent). Prompt now
asks for broad clustering (source/event/mechanism/sub-narrative) + a dependence_strength.

Deterministic: collapse < partial < independent (unit-tested). 109 passed; make check green.

**Real-data demonstration (no LLM)** on the french-rev vectors — a broad cluster at varying ρ:
ρ=0 → top 0.994, ρ=0.3 → 0.964, ρ=0.6 → 0.796, ρ=0.8 → 0.500, ρ=1.0 → 0.182. So broad clustering
at moderate ρ turns the degenerate 0.997 into a non-degenerate posterior — the overconfidence
lever works on real magnitudes.

**Live LLM-behavior confirmation — DONE (2026-06-20, via OpenRouter** to bypass the Gemini
free-tier quota: `--model openrouter/google/gemini-2.5-flash`, french_revolution.txt, 78 evidence
× 5 hyps, 220s). Two findings:

1. **The LLM DOES cluster broadly, as designed.** 14 dependence clusters covering 59/78 items
   (76%), sizes 8/8/5/5/4×6/3×3/2×2, with ρ spread across 0.4–0.7 keyed to shared
   condition/event/mechanism sub-narratives. So the partial-pooling input the math relies on is
   actually produced in practice, not just in the deterministic demo.
2. **Yet the posterior is still degenerate** (top h4=0.9995, *fragile*; overconfidence banner
   fired; report reads "ranking"). Broad clustering at the *moderate* ρ the LLM assigns (0.4–0.7)
   is NOT enough to de-degenerate a dense 78-item text: 14 moderate-ρ clusters + 19 unclustered
   items still accumulate to ~1.0. The earlier deterministic demo de-degenerated only because it
   used a *single* cluster at *high* ρ.

Conclusion: mechanism validated end-to-end (priors, prior_sensitivity, top_drivers, robustness,
banner all work via OpenRouter), and the known overconfidence limit is now confirmed on real LLM
output. **Per-hypothesis ρ and/or stronger pooling (higher ρ, or a learned ρ) is the load-bearing
next refinement** — moderate uniform ρ is demonstrably insufficient on dense texts. Until then the
honest read remains: ranking, not calibrated probability (which the banner states).

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
  cap the per-item PAIRWISE spread to LR_CAP (centered log-LR clamped to ±0.5·log CAP;
  interpretive evidence capped tighter at INTERPRETIVE_LR_CAP), relevance-discount
  (`LR**relevance`; <0.4 ⇒ 1.0). Joint update is a log-space softmax
  `post_i = softmax(log prior_i + Σ_m log LR_{m,i})` — order-invariant, no per-step
  clamping. **Reuse** EvidenceUpdate/HypothesisPosterior/BayesianResult, robustness,
  top_drivers, sensitivity (they operate on the derived per-hyp LRs).
- `pass_test`: ONE matrix call (all hyps + all evidence) → TestingResult.
- `report`/`pass_absence`/`pipeline`: adapt to new structure (show vector + derived LR).
- Tests: port test_pt_bayesian/test_pt_schemas/test_pipeline_integration fixtures to vectors.
- Atomic commit (central structure change); make check green before commit; then live run.

## Adversarial code review (round 3) — fixes applied

Two earlier "round 3" pastes were STALE (reviewer on pre-rebuild commits) — disregarded.
The genuine round-3 review (against current head) split into stale-branch artifacts vs.
real issues on this branch. Real ones, all fixed:

- **HIGH: report LRs ≠ update LRs.** `_build_vis_data` and the test matrix called
  `lr_matrix` without the interpretive caps the update uses, so displayed LRs could
  exceed what actually drove the posterior. Added `_interpretive_caps(result)` helper;
  applied to both `lr_matrix(...)` calls in report.py.
- **HIGH: refinement fail-open.** `apply_refinement` silently ignored unknown
  reinterpretation/hypothesis targets, unknown spurious ids, and out-of-graph new-edge
  endpoints. Now fails loud (except a target removed-as-spurious in the same delta,
  which is a legitimate no-op) and reports ACTUAL removal counts. New
  `tests/test_apply_refinement.py` (8 cases).
- **MEDIUM: absence prompt/context mismatch.** Prompt described a per-prediction
  "which predictions had relevant evidence" linkage the code no longer sends (it sends
  `substantive_evidence_ids`, relevance ≥ 0.6). Prompt section rewritten to match.
- **MEDIUM: enum-like string fields unconstrained.** `evidence_type`, `severity`,
  `refinement_type`, verdict `status`, and `ReinterpretedEvidence` type fields are now
  Literals. Enum-rejection test added.
- **LOW: stale overconfidence banner.** Said "clustering is not yet applied" — now false
  (partial pooling IS applied). Reworded to "pooling applied; magnitude unreliable if
  items weren't grouped strongly enough."

Per user YAGNI guidance, report security hardening (click-handler escaping etc.) was
NOT done — only correctness/contract fixes and the cheap render-robustness already present.
118 passed / 1 skipped; mypy clean; make check green.

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

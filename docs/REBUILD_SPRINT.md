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
- [ ] Slice 4 — bands + posterior interval + rank-stability
- [ ] Slice 2 — researcher priors + prior-sensitivity (priors param already in run_bayesian_update)
- [ ] Integration live run + report adaptation
- [ ] Ledger + docs update; open PR

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

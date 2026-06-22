# Pragmatic Build Plan — Automated Process Tracing (the 80/20)

**Status:** Build plan / companion to `WHITEPAPER_optimal_automated_process_tracing.md`
**Relationship:** The white paper specifies the *quality-optimum* with cost unconstrained. This document does the opposite job on purpose: decide the **smallest set of changes that captures most of the quality**, at sane cost, on the *existing* `pt/` pipeline. Where the two disagree, the white paper is the north-star and this plan is the compromise — each compromise is named so it can be revisited.

It is deliberately short. The optimum is the long document; the build is supposed to fit in your head.

**Current status (2026-06-22):** Slices 1-5 have shipped in pragmatic form,
along with residual `H0`, source-hash validation, report audit, and temporal
network presentation. Slice 4 currently reports support sensitivity ranges and
rank/prior stability, not full elicited likelihood bands with Monte Carlo
propagation. Slice 5 currently uses scalar dependence-cluster pooling, not
per-hypothesis redundancy. Slice 6 (qualitative structural critic) and the
optimum-only machinery remain planned. Historical rebuild notes are archived at
`docs/archive/development/REBUILD_SPRINT.md`.

---

## 1. The 80/20 thesis

Most of the optimum's quality comes from a handful of cheap, local fixes to how likelihoods are formed and reported. The expensive machinery (multi-family triangulation auditor, cross-cluster shared-error sampling, trace-production model-averaging, post-selection correction) is high-cost and, for a *single text*, lower marginal quality than getting the basics right. So the first build:

- **does** fix the priors, the coherence of the likelihood, support sensitivity reporting, and the worst dependence double-counting;
- **approximates** the auditor as a single cheap critic pass on the evidence that moves the result;
- **defers** the genuinely expensive optimum-only components, with each deferral logged here so it is a decision, not an omission.

## 2. Component-by-component: keep / simplify / defer

| Optimum component (§ in white paper) | First build | Why |
|---|---|---|
| Estimand = posterior odds over MECE hypotheses + residual (§5) | **keep** | free; it's a labeling/contract change |
| Researcher-settable priors + prior-sensitivity (§5, §6.2) | **keep** | small; removes the indefensible hardcoded `1/n` |
| Coherent likelihood **vector**, derived pairwise (§6.2.1) | **keep** | this is the core "pseudo → real" fix |
| Likelihood **bands** + posterior **interval** + rank-stability (§6.2.1) | **keep** | kills false precision; cheap to report |
| Evidence-dependence: **lineage collapse** (§6.5) | **keep (subset)** | the common, high-impact case (copies/wire/transcript) |
| Evidence-dependence: per-hypothesis partial pooling (§6.5) | **simplify → defer** | start with collapse-or-independent; add partial pooling later |
| Qualitative critic on **top-driver** evidence, single model (§6.3, §6.7) | **simplify** | gated + single-family is the cheap auditor; all-cluster/multi-family is optimum-only |
| Missingness w/ observability bands (§6.6) | **simplify** | keep current qualitative-only absence pass; add observability grading later |
| Two-sided trace-production model (§4) | **simplify** | let the critic *flag* solicitation/false-positive confounds qualitatively; no formal measurement model yet |
| Multi-family / epistemic-diversity audit (§6.3) | **defer** | biggest cost; single-family critic first, measure if it helps |
| Cross-cluster shared error terms (§6.2.1) | **defer** | within-vector joint sampling first |
| Trace-production structure model-averaging (§4) | **defer** | single best diagnosis first |
| Post-selection correction + prior-provenance audit (§5.2) | **defer (guardrail only)** | keep existing anti-circularity rule; full provenance audit later |
| Cross-case CausalQueries path (§7) | **keep (already built)** | exists; just don't mix it into single-text |

## 3. Thin slices (ordered; each ships independently and leaves the pipeline green)

Each slice names the file(s), the white-paper claim it realizes, and how it is verified. Run `make check` after each.

**Slice 1 — Truth-in-labeling. DONE.** Stop calling the output "posterior probability"; call it "comparative posterior support / posterior odds over the hypothesis set." *Files:* `pt/report.py`, `pt/schemas.py` (field descriptions/labels), synthesis prompt wording. *Verify:* report/test assertion on the new label; no math change. *Cost:* ~1 hr. *Removes the overclaim immediately.*

**Slice 2 — Researcher-settable priors + prior-sensitivity. DONE.** Replace hardcoded uniform `1/n` with a priors input (CLI/file), defaulting to uniform but recorded; add a prior-perturbation pass alongside the existing LR sensitivity. *Files:* `pt/bayesian.py`, `pt/cli.py`, `pt/schemas.py`. *Verify:* unit tests — non-uniform priors change posteriors as expected; prior-sensitivity populated. *Realizes:* §5/§6.2.

**Slice 3 — Coherent multi-hypothesis likelihood (the core fix). DONE.** Change Pass 3 from per-(hyp, evidence) two-way `P(E|H)` vs `P(E|¬H)` to a likelihood **vector** across all hypotheses per evidence item; derive pairwise ratios; update posteriors from the vector. *Files:* `pt/pass_test.py`, `pt/prompts/pass3_test.yaml`, `pt/schemas.py`, `pt/bayesian.py`. *Verify:* unit test that derived ratios are coherent (reciprocity/transitivity hold by construction); regression on a fixed extraction. *Realizes:* §6.2.1. *Biggest single quality gain.*

**Slice 4 — Bands + interval reporting. PARTIAL / PRAGMATIC DONE.** The report now surfaces support sensitivity ranges, rank stability, and prior stability. Full LLM-elicited likelihood bands and Monte Carlo propagation remain deferred. *Files:* `pt/bayesian.py`, `pt/report.py`, `pt/schemas.py`. *Verify:* tests on support range, rank stability, and prior stability. *Realizes part of:* §6.2.1.

**Slice 5 — Lineage dependence collapse. PRAGMATIC DONE.** The testing pass emits dependence clusters and the Bayesian update partially pools them with scalar dependence strength. Per-hypothesis redundancy and trace-production structure remain deferred. *Files:* `pt/pass_test.py`, `pt/schemas.py`, `pt/bayesian.py`. *Verify:* planted-duplicate test — a copied report does not move the posterior like an independent one. *Realizes:* §6.5 (subset).

**Slice 6 — Cheap critic pass (gated). PLANNED.** A single-model critic that, for the **top-driver** evidence only, flags `confound`/`too-strong`/`void` + direction; the estimator re-elicits flagged items. *Files:* new `pt/pass_audit.py` + wiring in `pt/pipeline.py`. *Verify:* the pipeline runs end-to-end with audit on/off (ablation hook); audit decisions logged. *Realizes:* §6.3/§6.7 in pragmatic (gated, single-family) form — and gives the **ablation switch** the white paper's validation needs.

Slices 1–4 convert the single-text path from pseudo-Bayesian to defensible without touching extraction or the report's structure. 5–6 add the highest-value dependence and audit pieces in cheap form. Everything in §2's "defer" column waits until an ablation shows the cheap versions help.

## 4. What we will measure (so this isn't faith-based)

Even the pragmatic build should produce the white paper's headline signal cheaply:

- **Ablation hook (Slice 6):** run with the critic on vs off on 2–3 texts; eyeball whether flagged confounds are real and whether rankings change defensibly.
- **Coherence check (Slice 3):** automated test that pairwise ratios are always consistent.
- **Duplicate test (Slice 5):** planted copied evidence must not double-count.
- **Stability (Slice 4):** report rank-stability; if a conclusion is rank-unstable, say so.

These are unit/integration tests plus a couple of live runs — not the full §8 benchmark, which stays optimum-scope.

## 5. The "cutter" review prompt (for THIS document)

The white paper used an *optimum-seeking* reviewer that resists both accretion and simplification. This build plan needs the **opposite** bias: a reviewer that aggressively cuts toward the minimum viable quality. Use this to review the build plan (never the white paper):

```
You are reviewing a PRAGMATIC build plan whose explicit goal is the MINIMUM set of
changes that captures MOST of the quality of a separate north-star design. Cost,
simplicity, and time-to-ship ARE first-class concerns here. The north-star optimum is
already documented elsewhere; do NOT re-derive it or push toward it.

Your job is to make this plan SMALLER and SOONER without sacrificing the few changes
that actually carry the quality. For every item, ask:

  [CUT]   Can this be dropped or deferred entirely for the first build with acceptable
          quality loss? Name the quality lost and why it is acceptable now.
  [SHRINK] Can this be done in a cheaper/simpler form that keeps ~80% of its value?
          Give the simpler form.
  [KEEP]  Is this one of the few changes that carries most of the quality, such that
          cutting it would gut the build? Justify why it cannot be deferred.

Hard rules:
  1. Bias toward CUT and SHRINK. A plan that does less, sooner, is better unless a KEEP
     item is genuinely load-bearing.
  2. For every KEEP, state what breaks if it is deferred. If nothing concrete breaks in
     the FIRST build, reclassify as CUT/SHRINK.
  3. Do NOT recommend adding anything from the optimum. New scope is out of bounds;
     this is a subtraction exercise.
  4. Rank the slices by (quality carried ÷ effort). Recommend the first shippable slice.
  5. Flag any slice that secretly depends on a deferred optimum component.
  6. Identify the ONE slice that, alone, delivers the most quality — the true MVP.

Output: a CUT list, a SHRINK list, a (short) KEEP list, the ranked slice order, and the
single-slice MVP recommendation.
```

It is the mirror of the optimum prompt: that one forbids cost objections; this one *requires* them.

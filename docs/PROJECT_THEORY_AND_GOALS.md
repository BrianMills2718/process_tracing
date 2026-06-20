# Process Tracing — Theory, Goals, and the Inference-Core Rebuild

**Canonical agent-facing orientation for this repository.**

*Version 1.0 — 2026-06-19*

> **Purpose & audience.** This is the single statement of *what this system is, the
> theory it implements, the conceptual model it operates on, what is being rebuilt,
> and the honest state of what is built*. It is for the coding agents (and humans)
> who work on this repo. It **complements**, and does not replace:
> - `CLAUDE.md` — operational (commands, file map, model/config notes).
> - `docs/WHITEPAPER_optimal_automated_process_tracing.md` — the **methodology
>   optimum** (the *why* and the full probabilistic design; cost-unconstrained
>   north-star). When you need depth on *any* concept below, that paper is the source.
> - `docs/BUILDPLAN_pragmatic_process_tracing.md` — the **80/20 build** (what to
>   actually build first, and the slice order).
>
> When this doc and the code disagree about *current capability*, trust the code +
> the ledger in §6. When they disagree about *intent*, trust this doc and the white paper.

---

## 1. What this project is (one paragraph)

Automated Van-Evera-style **process tracing**: read a historical text, and report which of several competing causal explanations the evidence supports, how strongly, and how robustly. The LLM is the **engine of a methodology-aware pipeline** (extract → hypothesize → test → update → synthesize), with the human as reviewer/director. The output is **comparative explanatory support over an explicitly specified, mutually-exclusive hypothesis set** — *not* an absolute probability of truth, a causal-effect size, or a counterfactual. The bet: good single-text causal inference is neither pure-human (slow, undocumented) nor pure-LLM (fluent but incoherent); it is an LLM forming likelihoods under a disciplined Bayesian contract whose reasoning is externalized, audited, and honestly bounded.

## 2. Conceptual model (the contracts the rebuild must honor)

Each item points to the white-paper section with the full treatment.

- **Estimand (WP §5).** Single-text output = posterior odds over `{H₁ … H_k}` + an explicit **residual H₀**, each `Hᵢ` an intensional causal story. Reported as **comparative support + ranking + robustness + sensitivity**. Never an identified effect/necessity/sufficiency/counterfactual.
- **Two models, never conflated (WP §4).** The **substantive causal model** (events → events) is distinct from the **trace-production model** (events → the evidence we observe: solicited, recorded, survived, extracted — *and* a false-positive channel for claims asserted though false). `P(E|H)` decomposes across both.
- **Likelihoods are a coherent vector (WP §6.2.1).** Per evidence cluster, elicit one **relative-log-likelihood vector** across all hypotheses (anchored to a reference); **derive** pairwise ratios from it so reciprocity/transitivity hold by construction. Never elicit independent pairwise LRs. Express as **bands**, propagate as intervals (joint, not independent), report a **posterior interval + rank-stability**.
- **The auditor is a *qualitative* critic (WP §6.3).** A graph's topology cannot compute a likelihood. The single-text critic emits categories + direction (`confound` / `too-strong` / `void` / `confirm`); the **estimator re-elicits** the magnitude. True likelihood *derivation* from a parameterized graph belongs to the **cross-case** path only.
- **Dependence on an evidence graph (WP §6.5).** Conditional dependence (not shared ancestry alone) decides clustering; lineage collapses, same-event-independent-channels partially pool, per-hypothesis.
- **Absence = source-silence (WP §6.6).** The datum is `P(source omits E | H)` weighted by observability, never "prove world-absence."
- **Input provenance (WP §5.2).** A quantity may enter the calculation once: guard post-selection (hypotheses generated from the evidence) and prior/likelihood double-counting.
- **Pure math stays deterministic.** Bayesian updating, normalization, band propagation, sensitivity = pure Python in `pt/bayesian.py`, no LLM (per `CLAUDE.md` exception).

## 3. Rebuild scope: what is being rebuilt vs reused

We are doing a **clean-slate rebuild of the single-text inference core, in place**, and **reusing/extending** the rest. This is *not* a repo restart.

| Area | Decision | Notes |
|---|---|---|
| `pt/pass_test.py` + `pass3_test.yaml` (likelihood elicitation) | **REBUILD** | per-evidence likelihood **vector** across hypotheses, replacing per-hypothesis two-way `P(E\|H)` vs `P(E\|¬H)` |
| `pt/bayesian.py` (update/bands/sensitivity) | **REBUILD** | consume vectors; posterior **intervals** + rank-stability; researcher priors + prior-sensitivity |
| likelihood-related parts of `pt/schemas.py` | **REBUILD** | vector schema, bands, evidence-cluster, critic outputs |
| evidence graph / dependence | **NEW** | minimal lineage clustering first (BUILDPLAN Slice 5) |
| qualitative critic pass | **NEW** | gated, single-family first (BUILDPLAN Slice 6); ships the ablation switch |
| `pt/llm.py` | **REBUILD (small)** | delegate to `llm_client.call_llm_structured` (validated Pydantic, three-tier routing); stop hand-rolling schema-injection/parse-retry |
| `pt/pass_extract.py`, `pt/pass_hypothesize.py` | **REUSE + EXTEND** | sound; add provenance + MECE/residual fields |
| cross-case path (`pass_binarize`, `cq_bridge`, `multi_pipeline`) | **REUSE** | this *is* the white paper's formal path; already aligned |
| `pt/report.py` | **REUSE + ADAPT** | already security-fixed; adapt to render support-intervals + audit state |
| harness, Makefile, `tests/`, prompt loading | **REUSE** | infrastructure |

Rule of thumb: rebuild where the code embodies the *pseudo-Bayesian* design (two-way LRs, point posteriors, uniform-prior, independent multiplication); reuse where it is already correct and debugged.

## 4. Invariants (non-negotiable; violating one is a bug)

1. **LLM-first for semantics** (per `CLAUDE.md`): no keyword/rule-based evidence classification; all semantic judgments via LLM structured output.
2. **All LLM calls go through `llm_client`** (via `pt/llm.py`) with `task=/trace_id=/max_budget=`; no direct API calls.
3. **The schema is the contract.** Every LLM boundary returns a validated Pydantic model; required fields are required.
4. **Coherent likelihoods only.** Pairwise ratios are *derived from a vector*, never independently elicited.
5. **Comparative support, not probability of truth.** Outputs are normalized over the listed hypotheses; the report says so (truth-in-labeling).
6. **Fail loud.** Raise on LLM/parse/bridge failure; no silent None/0/[]; no `except: pass`.
7. **A quantity enters once.** Partition frozen pre-test; anti-circularity on hypothesis-generating evidence; prior-provenance checked.
8. **Math is deterministic and unit-tested.** No LLM in `bayesian.py`.
9. **Cross-case ≠ single-text.** No single analysis is scored by both engines; the cross-case (CausalQueries) path owns identified counterfactual/population estimands.

## 5. Claim discipline (what we may and may not say)

When describing outputs (reports, commits, user text):
- Say **"comparative support / posterior odds over the listed hypotheses,"** never "probability that H is true."
- Never claim an **identified causal effect, counterfactual, or necessity/sufficiency** from a single text.
- Distinguish **implemented / partial / planned** (see §6); do not let a named stage imply a demonstrated capability.
- Report **rank-stability and sensitivity**, not the third decimal of a point estimate.

## 6. Proven-vs-planned ledger (condensed; full table in WP §9)

**Implemented & sound (reuse):** extraction, hypothesis generation, Van Evera diagnostic classification, mechanical robustness, LR-sensitivity, cross-case CausalQueries bridge, report shell (security-fixed), harness/tests. Truth-in-labeling shipped (Slice 1, PR #5).

**Partial:** single-text Bayesian update exists but is the *pseudo* form (two-way LRs, uniform `1/n`, point posteriors); `top_drivers` computed but no auditor consumes them; absence pass is qualitative-only.

**Planned (this rebuild):** coherent likelihood vectors; researcher priors + prior-sensitivity; bands + posterior intervals + rank-stability; evidence-graph lineage clustering; qualitative critic pass + ablation switch; `llm_client.call_llm_structured` boundary; provenance guards.

**Not claimed:** no methodological validation has been run. Until the ablation (WP §8) executes, this is an *auditable inference architecture*, not a *validated* one.

## 7. Build sequence

Follow `BUILDPLAN_pragmatic_process_tracing.md`, adapted to the in-place rebuild. Each slice ships independently, leaves `make check` green, and carries a regression test:

1. **Slice 1 — truth-in-labeling** ✅ (PR #5).
2. **LLM boundary** — `pt/llm.py` → `call_llm_structured` (folds in the old "Slice 0"; do it before the vector schema so richer structured output is robust).
3. **Slice 3 (MVP) — coherent likelihood vector** — rebuild `pass_test` + prompt + schema + `bayesian` to elicit/derive vectors. **Needs one live validation run** (per `CLAUDE.md`: validate prompt changes by running the pipeline).
4. **Slice 4 — bands + posterior intervals + rank-stability.**
5. **Slice 2 — researcher priors + prior-sensitivity.**
6. **Deferred until a run shows they matter:** evidence-graph clustering (Slice 5), critic pass (Slice 6), and all optimum-only machinery (cross-cluster shared error, trace-production model-averaging, multi-family diversity).

## 8. How to verify (the loop that keeps us honest)

- **Deterministic:** unit tests for every `bayesian.py` change (coherence of derived ratios; band→interval propagation; prior-sensitivity; planted-duplicate non-double-counting). `make check` after each slice.
- **Behavioral (LLM contract):** any change to `pass_test`/prompts requires a **live run** on `input_text/revolutions/french_revolution.txt` (and a debate text, per prior lessons) before it is called done — eyeball whether vectors/bands are sane and whether the ranking is defensible.
- **Headline empirical question (later):** does the critic/auditor actually improve inference vs. narrative-only (WP §8 ablation)? That is what graduates this from "auditable" to "validated."

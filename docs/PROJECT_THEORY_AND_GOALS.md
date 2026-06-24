# Process Tracing — Theory, Goals, and the Inference-Core Rebuild

**Canonical agent-facing orientation for this repository.**

*Version 1.1 - 2026-06-22*

> **Purpose & audience.** This is the single statement of *what this system is, the
> theory it implements, the conceptual model it operates on, what has been rebuilt,
> and the honest state of what is built*. It is for the coding agents (and humans)
> who work on this repo. It **complements**, and does not replace:
> - `CLAUDE.md` — operational (commands, file map, model/config notes).
> - `docs/WHITEPAPER_optimal_automated_process_tracing.md` — the **methodology
>   optimum** (the *why* and the full probabilistic design; cost-unconstrained
>   north-star). When you need depth on *any* concept below, that paper is the source.
> - `docs/BUILDPLAN_pragmatic_process_tracing.md` — the **80/20 build** and
>   compromise record.
> - `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md` and
>   `docs/plans/002_sota_plus_recovery_plan.md` — the current SOTA+ reset:
>   external frontier map, failure postmortem, and thin-slice gates.
> - `docs/SOTA_PLUS_TARGET_ARCHITECTURE.md` — the end-goal design-plan
>   architecture for the intended SOTA+ system, including future boundaries and
>   contracts that are not fully implemented yet.
>
> When this doc and the code disagree about *current capability*, trust the code +
> the ledger in §6. When they disagree about *intent*, trust this doc and the white paper.

---

## 1. What this project is (one paragraph)

Automated Van-Evera-style **process tracing** is the first concrete slice of a broader mixed-methods research system: read historical/source material, construct and test rival causal explanations, quantify comparative support, then integrate the resulting traces with cross-case / quantitative causal designs. The LLM is the **engine of a methodology-aware pipeline** (extract → hypothesize → test → update → synthesize), with humans as research directors, validators, and accountability anchors rather than the assumed bottleneck for every within-case causal judgment. The output is **comparative explanatory support over an explicitly specified, mutually-exclusive hypothesis set** — *not* an absolute probability of truth, a causal-effect size, or a counterfactual. The bet: expert process-tracing work can be automated to PhD / think-tank / academic quality when within-case causal inference is decomposed into structured, inspectable operations and coupled to deterministic math, adversarial audits, provenance, sensitivity analysis, quantitative integration, active source design, and explicit trace-production modeling.

The project therefore rejects the default claim that humans are inherently better or less biased at process-tracing interpretation. Human judgment is useful, but not privileged as ground truth. Quality comes from architecture: explicit contracts, role separation, independent critique, reproducible traces, calibration tests, agentic assistant labor routed through governed coding harnesses, and iterative process tracing ↔ quantitative feedback.

## 2. Conceptual model (the contracts the rebuild must honor)

Each item points to the white-paper section with the full treatment.

- **Estimand (WP §5).** Single-text output = posterior odds over `{H₁ … H_k}` + an explicit **residual H₀**, each `Hᵢ` an intensional causal story. Reported as **comparative support + ranking + robustness + sensitivity**. Never an identified effect/necessity/sufficiency/counterfactual.
- **Two models, never conflated (WP §4).** The **substantive causal model** (events → events) is distinct from the **trace-production model** (events → the evidence we observe: solicited, recorded, survived, extracted — *and* a false-positive channel for claims asserted though false). `P(E|H)` decomposes across both.
- **Likelihoods are a coherent vector (WP §6.2.1).** Per evidence item or cluster, elicit one **relative-log-likelihood vector** across all hypotheses; **derive** pairwise ratios from it so reciprocity/transitivity hold by construction. Never elicit independent pairwise LRs. The optimum expresses these as bands with joint propagation; the current pipeline reports support sensitivity ranges and rank-stability while full band elicitation remains partial.
- **The auditor is a *qualitative* critic (WP §6.3).** A graph's topology cannot compute a likelihood. The single-text critic emits categories + direction (`confound` / `too-strong` / `void` / `confirm`); the **estimator re-elicits** the magnitude. True likelihood *derivation* from a parameterized graph belongs to the **cross-case** path only.
- **Dependence on an evidence graph (WP §6.5).** Conditional dependence (not shared ancestry alone) decides clustering; lineage collapses, same-event-independent-channels partially pool, per-hypothesis.
- **Absence = source-silence (WP §6.6).** The datum is `P(source omits E | H)` weighted by observability, never "prove world-absence."
- **Input provenance (WP §5.2).** A quantity may enter the calculation once: guard post-selection (hypotheses generated from the evidence) and prior/likelihood double-counting.
- **Pure math stays deterministic.** Bayesian updating, normalization, band propagation, sensitivity = pure Python in `pt/bayesian.py`, no LLM (per `CLAUDE.md` exception).

## 3. Rebuild scope: what was rebuilt vs reused

The project completed an in-place rebuild of the single-text inference core and
reused/extended the rest. This was not a repo restart.

| Area | Decision | Notes |
|---|---|---|
| `pt/pass_test.py` + `pass3_test.yaml` (likelihood elicitation) | **REBUILT** | per-evidence likelihood **vector** across hypotheses, replacing per-hypothesis two-way `P(E\|H)` vs `P(E\|¬H)` |
| `pt/bayesian.py` (update/sensitivity) | **REBUILT** | consumes vectors; log-space support update; residual `H0`; dependence pooling; researcher priors + prior-sensitivity |
| likelihood-related parts of `pt/schemas.py` | **REBUILT / PARTIAL** | vector schema and dependence-cluster schema implemented; full elicited bands and critic outputs deferred |
| evidence graph / dependence | **PARTIAL** | LLM-supplied dependence clusters with scalar partial pooling; per-hypothesis redundancy deferred |
| qualitative critic pass | **PLANNED** | gated, single-family first (BUILDPLAN Slice 6); should ship the ablation switch |
| `pt/llm.py` | **REBUILT (small)** | delegates to `llm_client.call_llm_structured` (validated Pydantic, routed model calls) |
| `pt/pass_extract.py`, `pt/pass_hypothesize.py` | **REUSE + EXTEND** | sound; Pass 2 now accepts source-packet context; remaining work is stronger MECE/residual and partition-provenance auditing |
| cross-case path (`pass_binarize`, `cq_bridge`, `multi_pipeline`) | **REUSE** | this *is* the white paper's formal path; already aligned |
| `pt/report.py` | **REUSE + ADAPT** | renders support, sensitivity, PhD audit, evidence triage, source-packet contract, temporal timeline, and temporal causal network |
| `pt/verdict_calibration.py` | **NEW GUARDRAIL** | deterministic post-synthesis guard downgrades status labels that overstate computed comparative support |
| agentic assistant harness | **PARTIAL** | source-packet draft task is implemented through `llm_client` `workspace_agent`; partition critique, benchmark repair, and report critique assistants are still planned |
| source-packet contract / source-design engine | **PARTIAL** | `--source-packet` loads typed packet artifacts, stores source-scope metadata, reports packet-source coverage from exact text markers, and supports acquisition planning; the full source-design engine loop (iterative source expansion, observability ranking, and acquisition disposition) is still incomplete |
| harness, Makefile, `tests/`, prompt loading | **REUSE** | infrastructure |

Rule of thumb for future work: extend where the remaining gaps are methodological
(source scope, hypothesis partition, dependence, trace production, validation);
do not reintroduce the old pseudo-Bayesian design (two-way LRs, hidden uniform
priors, independent multiplication, or "posterior probability" labels).

## 4. Invariants (non-negotiable; violating one is a bug)

1. **LLM-first for semantics** (per `CLAUDE.md`): no keyword/rule-based evidence classification; all semantic judgments via LLM structured output.
2. **All LLM and agent calls go through `llm_client`.** Structured pipeline calls go via `pt/llm.py`; workspace-agent assistant calls go through a narrow process-tracing wrapper over `llm_client` `execution_mode="workspace_agent"` with `task=/trace_id=/max_budget=`. No direct provider SDK calls, Codex CLI calls, Claude Code CLI calls, or assistant subprocess glue in this repo.
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

**Implemented (rebuild plus report-audit work):**
- LLM boundary on `llm_client.call_llm_structured` (live smoke ✓).
- **Coherent likelihood vectors** — per-evidence vector across hypotheses, geomean-derived
  per-hyp LRs, joint normalization (Slice 3; live-validated on french_revolution.txt ✓).
- **Researcher-settable priors + prior-sensitivity** — CLI `--priors`, `PriorSensitivity` (Slice 2).
- **Residual hypothesis** — `H0_residual` is included in the default pipeline update.
- **Dependence clustering** — Pass 3 returns dependence clusters and `bayesian.py`
  partially pools them before updating.
- **Support sensitivity range + rank-stability + prior-stability** surfaced in the report headline.
- **Truth-in-labeling** — report says "Support" (comparative), not "posterior probability."
- **Source provenance** — `source_text_sha256` prevents `--from-result` reuse with a different input text.
- **Report audit** — `make audit-result` and the HTML report expose caps,
  recommendations, evidence triage, and optimality status.
- **Temporal network** — report network uses fixed temporal coordinates and toggles
  top drivers, background drivers, additional links, temporal conflicts, and isolated nodes.
- Reused & intact: extraction, hypothesis generation, Van Evera classification, mechanical
  robustness, sensitivity, cross-case CausalQueries bridge, report shell, harness/tests.

**Partial:** absence pass is qualitative-only (observability grading deferred);
dependence pooling uses a scalar per cluster rather than per-hypothesis
redundancy; support ranges are sensitivity ranges, not full elicited likelihood
bands with Monte Carlo propagation; Van Evera labels are carried per cell but
`prediction_classifications` is not yet repopulated by the new pass.

**Planned / deferred (per build plan & cutter):** hypothesis partition audit;
agentic assistant tasks beyond source-packet drafting, including source-design
expansion, benchmark repair, and report critique through `llm_client`
Codex/Claude Code backends; full band *elicitation* + joint Monte-Carlo
propagation; qualitative critic/auditor pass + ablation switch; first-class
trace-production model; post-selection & prior-provenance guards;
per-hypothesis dependence; cross-cluster shared-error sampling; source-design
engine completion with high-priority missing-source resolution; formal
validation benchmark; and a tighter within-case -> cross-case causal-model
bridge.
These are optimum-scope or next-roadmap work; the current build approximates or defers them.

**Not claimed:** no methodological validation (the WP §8 auditor ablation) has been run. This is
an *auditable inference architecture*, not a *validated* one.

## 7. Implementation sequence

The historical slice order is preserved in
`docs/BUILDPLAN_pragmatic_process_tracing.md`; the current state is:

1. **Done:** truth-in-labeling, `llm_client` boundary, coherent likelihood vectors,
   researcher priors, residual `H0`, dependence pooling, sensitivity/prior
   stability, report audit, and temporal network presentation.
2. **Next:** hypothesis partition audit; completion of the source-design engine
   beyond static packets; stronger source-lineage/dependence modeling;
   first-class trace-production modeling; observability-weighted absence;
   benchmark/report assistant tasks; and the qualitative structural critic.
3. **Validation:** auditor/dependence ablations and a frozen benchmark are still
   required before claiming demonstrated PhD-level methodological validity.

## 8. How to verify (the loop that keeps us honest)

- **Deterministic:** unit tests for every `bayesian.py` change (coherence of
  derived ratios; residual behavior; prior-sensitivity; dependence pooling;
  planted-duplicate non-double-counting). `make check` after each slice.
- **Behavioral (LLM contract):** any change to `pass_test`/prompts requires a **live run** on `input_text/revolutions/french_revolution.txt` (and a debate text, per prior lessons) before it is called done — inspect whether likelihood vectors, dependence clusters, and rankings are defensible.
- **Headline empirical question (later):** does the critic/auditor actually improve inference vs. narrative-only (WP §8 ablation)? That is what graduates this from "auditable" to "validated."
- **SOTA+ gate:** every non-trivial slice must name the external SOTA frontier it
  advances, the capability-ladder row it moves, and the benchmark/failure mode it
  tests. Use Plan 002 as the operating model.
- **Agentic assistant gate:** any Codex/Claude Code assistant task must run
  through `llm_client`, emit a typed artifact, and preserve `task`, `trace_id`,
  `max_budget`, backend, working directory, and provenance in a testable place.

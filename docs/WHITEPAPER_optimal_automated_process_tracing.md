# A Bayesian Audit Architecture for Automated Process Tracing

### Externalized LLM Likelihoods, Trace-Production Models, and Local Causal-Graph Audits for Single-Text Historical Inference

**Status:** Draft methodology white paper — **v3.1**
**Scope:** Single-text and cross-case causal inference from historical narrative
**Audience:** Methodologists, computational social scientists, and engineers building automated causal-inference pipelines

---

## What changed in v3.1

Minor revision applying transferable lessons from a review of a sibling project. This remains a **planning/architecture document** (no empirical results claimed) and intentionally omits an ethics/threat section as out of scope.

1. **Implementation-status table (§9).** Per-capability honest status — Implemented / Partial / Planned — so no named stage is mistaken for a demonstrated capability.
2. **Construct-language pass (§6).** Distinguishes auditability of *process* vs. of *interpretation* vs. *validity* of interpretation, and states explicitly that the architecture buys the first two, not the third (which §8 must earn).

## What changed in v3

v3 responds to a second adversarial review (which graded v2 B+ and asked for "revise and resubmit"). It pins down the parts that were named but not operational:

1. **Formal likelihood-band semantics (§6.2.1).** Bands are now defined as LR intervals with a stated propagation model (log-uniform Monte Carlo → posterior-odds distribution), plus conservative interval-arithmetic and ordinal-only fallbacks. Open-ended bands are bounded by the implementation's LR floor/cap.
2. **Hypothesis-partition validity (§5.1).** Rules for mutual exclusivity, exhaustiveness via an explicit residual hypothesis, nesting, conjunctural causes, granularity, and split/merge — closing the "gerrymandered menu" hole.
3. **Formal evidence graph (§6.5).** Node/edge schema, an operational clustering rule, and a worked duplicate-evidence example.
4. **Validation metrics and falsification thresholds (§8).** Scoring rules, an operationally-identical baseline, benchmark-construction cautions, a cost metric, and provisional pass/fail thresholds.
5. **One complete worked run (§6.8).** Three hypotheses + residual, priors, bands, dependence graph, trace-production adjustment, a posterior interval, a reconciler log, and a sensitivity conclusion.
6. **Adversarial objections folded in:** procedural/model/**epistemic** independence distinction (§6.3); a faithful-abstraction "translation contract" the auditor must certify before comparing numbers (§6.3); cluster-/volume-aware audit triggers so correlated weak evidence is not missed (§6.7); **observability bands** for absence so negative evidence is not toothless (§6.6); a narrowed feasibility claim (§10); and "ritualized Bayesianism" countered by proper scoring + protocol preregistration (§8).

## What changed in v2

v2 revises v1 in response to adversarial review. The headline changes:

1. **Demoted overclaims.** v1 said the two traditions "are the same Bayesian inference differing in one place" and described the architecture as "optimal feasible." Both are unjustified. v2 reframes the relationship as *degree of externalization* and the architecture as a *principled approximation* with a stated (if informal) objective.
2. **Added a formal estimand (§5).** v1 slid between "posterior over explanations" and "infer causal structure." v2 fixes the single-text target precisely: posterior odds over mutually-exclusive explanatory hypotheses — **not** identified causal effects or a learned causal structure.
3. **Separated event-causation from evidence-generation (§4).** v1 treated every likelihood judgment as a *causal-mechanism* judgment. `P(E|H)` is about how the *evidence* was produced. v2 introduces a **trace-production model** distinct from the substantive causal model, and reworks the worked example accordingly.
4. **Separated priors from likelihoods (§6.2).** v1's worked example let a prior on a variable change a likelihood, then "resolved toward" a number — a category error. v2 adds a **likelihood-comparability requirement** and a reconciler that never fixes a prior by nudging a likelihood.
5. **Specified the reconciler as a protocol (§6.4)** with a constrained action table, replacing v1's discretionary "diagnose and adjust."
6. **Likelihood bands instead of false-precise points (§6.2); an explicit missingness model (§6.6); a validation section with the auditor ablation as the headline test (§8).**
7. **Scoped out** the upstream extraction layer and the full validation protocol as companion work (§9), and trimmed implementation-roadmap material to keep methodology and engineering registers separate.

The durable contribution is **architectural**, not a philosophical unification: use narrative likelihoods as the primary single-text engine, force their implicit models into explicit form, and audit only the posterior-moving evidence with local causal graphs.

---

## Abstract

Automated systems that read a historical text and output "what caused this" tend to fail in one of two ways: confident prose with no inferential discipline, or a rigid formal model that discards the textual richness that makes historical evidence informative. Both failures stem from misreading the relationship between the two traditions of Bayesian process tracing — the *narrative / likelihoodist* tradition (Fairfield & Charman) and the *formal / model-based* tradition (Humphreys & Jacobs). They can both be represented as Bayesian updating, but they differ in how they define hypotheses, encode mechanisms, represent dependence, and generate likelihoods. The operationally decisive difference for automation is the **degree to which the causal model behind the likelihood is externalized** — written down, made criticizable, and used to enforce coherence and detect dependence.

A holistic likelihood judgment typically rests on an *implicit* model. But that model is not only a model of historical causation; `P(evidence | hypothesis)` is fundamentally about **how the evidence was produced** — solicited, recorded, preserved, observed — and conflating the two is a recurring error. We therefore keep two models distinct: the **substantive causal model** (events causing events) and the **trace-production model** (events producing the evidence now available).

We do not claim optimality. We propose a **principled approximation** whose objective is to improve *auditability, dependence handling, and calibrated sensitivity reporting* relative to naive automated likelihood multiplication, at controlled cost. The architecture is **task-routed**: for a single text, a narrative likelihood engine whose estimates carry externalized reasoning and are stated as likelihood bands; for the posterior-moving subset of evidence, a structured second-pass audit by an independently-constructed local causal graph, with disagreements resolved by a **constrained reconciliation protocol** rather than averaging; and, for many comparable cases, the formal model (CausalQueries) as the source of truth for *identified* counterfactual and population estimands. Van Evera's diagnostic tests are retained as a human-legible labeling layer over the likelihoods. We state the estimand precisely, bound the architecture's claims by what the data can identify, and specify how one would test whether it works.

---

## 1. The problem

Given a historical text, we want to infer which of several competing explanations the evidence supports, how strongly, and how robustly. "Automated process tracing" attempts this end to end: extract evidence and hypotheses, test each hypothesis against the evidence, update beliefs, synthesize a verdict.

Two failure modes recur:

1. **Pseudo-quantification.** The system emits "posterior probabilities" from machinery that is not a coherent probability model — typically per-evidence likelihood ratios multiplied under an unstated independence assumption, then renormalized to look like a distribution.
2. **Over-formalization.** The system forces the problem into a rigid causal graph over binary variables, discarding the discriminating texture of the evidence and committing to a structure the text does not warrant.

This paper aims at a design that avoids both, *and is honest about what it cannot deliver.*

---

## 2. Background: two traditions of Bayesian process tracing

**Narrative / likelihoodist (Fairfield & Charman 2017, 2022).** The analyst specifies mutually-exclusive hypotheses, assigns explicit prior odds from background knowledge, and for each piece of evidence assesses a *likelihood ratio* — how expected the evidence is under one hypothesis versus another — then updates by Bayes' rule in odds space and conducts sensitivity analysis over priors and likelihoods. Hypotheses are holistic explanations; no formal graph is required. Van Evera's diagnostic tests describe the *shape* of these ratios.

**Formal / model-based (Humphreys & Jacobs 2015, 2023; the `CausalQueries` R package).** The analyst declares a causal DAG over (typically binary) nodes, places priors over *causal types* (how each node responds to its parents), and updates beliefs about those types from observed node values. Each hypothesis is a *query* against this one model. Within-case process tracing and cross-case inference are unified, and conditional dependence among evidence is handled structurally via d-separation.

The traditions differ in more than arithmetic: in **what counts as a hypothesis** (an intensional, mechanism-specific story versus an extensional query over outcomes), in what counts as evidence, in how measurement error is represented, and in whether causal claims are defined by historically-specific mechanisms or by behavior across possible cases. These are real differences. What unifies them is only that both are *representable* as Bayesian updating.

---

## 3. The operational lever: how much of the model is externalized

All process tracing here is Bayes' rule:

```
P(H | evidence) ∝ P(H) · P(evidence | H)
```

The traditions can be placed on a single axis by asking **how the likelihood `P(evidence | H)` is obtained**:

- **Narrative:** assessed **directly**, as a holistic judgment, with the underlying model left implicit.
- **Formal:** **derived** by propagating through an explicit causal graph with specified parameters.

A worked contrast. Take **H₁** = "fiscal crisis was the primary cause of the Revolution" and **E** = "the 1789 *cahiers de doléances* overwhelmingly complained about taxes, not philosophy."

- *Narrative.* The analyst reasons "if fiscal crisis were the driver, grievances would track what hurts people → taxes" and writes a likelihood. The number is an **input**.
- *Formal.* The analyst introduces nodes, a structure (`F → G`, `F → R`), and parameters, and the likelihood is an **output** of the model. "H₁" becomes a query about the `F → R` relationship.

### 3.1 The implicit model — and what it actually contains

A direct likelihood judgment usually rests on an implicit model. But it is a mistake — v1's mistake — to assume that model is always a model of *historical causation*. A likelihood judgment can rest on a causal mechanism, an empirical regularity, source criticism, a semantic cue, or survival/selection bias in the archive. Several of these are about **how the evidence came to exist and be observed**, not about how one historical event caused another.

This matters because `P(E|H)` is, by definition, a statement about the **evidence-generation process**. "The *cahiers* complained about taxes" may be likely under fiscal-crisis causation — *or* because tax grievances were administratively solicited, locally formulaic, and more documentable than ideological commitments. An audit that checks only the event-causation story will audit the wrong thing.

> **Claim (corrected from v1):** a holistic likelihood judgment rests on an implicit model, but that model spans both *event causation* and *evidence generation*. Externalizing only the former is insufficient.

### 3.2 Van Evera tests are a labeling layer

Hoop / smoking-gun / doubly-decisive / straw-in-the-wind are labels for regions of the likelihood ratio, independent of how the likelihood was obtained (hoop: `P(E|H)` high; smoking-gun: `P(E|¬H)` low; doubly-decisive: both; straw-in-the-wind: weak both ways). They are a presentation layer, not a third method.

---

## 4. Two models you must not conflate

The single most consequential design commitment in v2: keep two models explicitly separate.

1. **Substantive causal model** — how historical events produced other historical events (`fiscal crisis → state breakdown → revolution`).
2. **Trace-production model** — how those events (and institutions, archives, and analysts) produced the **evidence now observed** (`fiscal distress → grievances → administrative solicitation of grievances → document survival → extraction by the pipeline → observed proposition`).

`P(E|H)` decomposes across both. Schematically:

```
P(observed E | H) = P(latent fact | H)        ← substantive causal model
                  × P(fact recorded & survives & observed | latent fact, H)   ← trace-production model
```

Most automated systems (v1 included) collapse these, attributing all of `P(E|H)` to the causal story. The decomposition is what lets the architecture distinguish "this evidence is expected because the cause operated" from "this evidence is over-represented because of how the record was made."

---

## 5. The estimand

A methodology paper that criticizes pseudo-quantification must say exactly what quantity it infers.

**Single-text estimand.** A posterior distribution (equivalently, posterior odds) over a **finite set of mutually-exclusive explanatory hypotheses** `{H₁ … H_k}`, each an *intensional* causal claim (a specified mechanism). The reported outputs are: the posterior odds/ranking, a robustness signal, and sensitivity ranges. This is **comparative explanatory support**, not an identified causal quantity.

What the single-text estimand is **not**:

- not an identified average causal effect;
- not a data-identified probability of necessity or sufficiency;
- not a learned causal structure (the structure, where used, is an *audit instrument*, not an inferential target).

Concretely, "fiscal crisis caused the Revolution" is operationalized as "`H_fiscal` carries higher posterior odds than its rivals given the evidence and stated priors." It is **not** a counterfactual probability. The looser readings — necessity, sufficiency, INUS membership, probability-raising, main-cause-relative-to-X, one node in a conjunctural package — are distinct claims; the single-text path commits only to *comparative support among the specified hypotheses*.

**Cross-case estimand.** Identified population estimands (e.g., probability of causation, average effects) via the formal model, **conditional on comparability holding** (§7).

### 5.1 Hypothesis-partition validity

A posterior over a badly-constructed hypothesis set can be perfectly coherent and substantively meaningless. Because the estimand is *conditional on the partition*, the partition is itself an object that must be audited. Five rules govern it:

1. **Mutual exclusivity.** Hypotheses must be competing *complete* explanatory stories, not overlapping factors. The pipeline's existing self-check — "if H_A is decisive, does that make H_B unnecessary or false?" — is the operational test; a "no" means the two are not exclusive and must be merged.
2. **Exhaustiveness via an explicit residual.** The set always includes a residual hypothesis **H₀ = "none of the specified stories (other cause, or a genuinely conjunctural combination)."** Without it, the posterior is conditional on a closed menu and silently assumes the truth is on the list. H₀ carries real prior mass.
3. **Nesting and conjunction.** Conjunctural explanations ("fiscal crisis caused elite fragmentation, which let ideology mobilize") are *not* mutually exclusive with their components and must not sit alongside them as if they were. A conjunctural story is admitted only as **its own distinct hypothesis** with the component factors removed from the menu, or it is deferred to the formal model (§7), which represents composition natively. The single-text partition is a set of *mutually-exclusive whole stories*, never a mix of factors and combinations-of-factors.
4. **Granularity.** All hypotheses share a comparable mechanism grain; one richly specified story tested against vague rivals biases the comparison toward the rivals (more ways to be "consistent").
5. **Split/merge discipline.** Merge when exclusivity fails (rule 1); split when a single "hypothesis" bundles separable mechanisms that the evidence can discriminate.

These rules are enforced at the **partition-audit** step, which is preregistered and frozen before the testing pass (the pipeline's `--review` checkpoint is the natural home). A gerrymandered menu is the easiest way to game posterior odds; making the partition an audited, frozen artifact is the defense.

---

## 6. The architecture

The objective — not "optimality," but a stated target — is to **maximize auditability, dependence-handling, and calibrated sensitivity per unit of model-call cost**, relative to the naive baseline of multiplying LLM-elicited likelihoods. The components below each serve that objective.

**What "auditability" does and does not mean here.** Three things must not be conflated:

1. **Auditability of process** — logs, schemas, provenance, stage dependencies (can we see *what the system did*?);
2. **Auditability of interpretation** — can a third party trace, contest, and reproduce a *specific analytic claim*?
3. **Validity of interpretation** — are the resulting likelihoods and ranking *substantively defensible*?

This architecture directly buys (1) and most of (2): every likelihood carries externalized reasoning, dependence is on an explicit graph, the auditor logs translation losses, and the reconciler logs its actions. It does **not** by itself buy (3). Validity of interpretation is an *empirical* property that only the validation protocol (§8) can establish; the architecture makes invalid interpretations **visible and contestable**, not impossible. Where this paper uses words like "audit," "coherence," or "robustness," they denote process/interpretation affordances, not demonstrated substantive validity.

### 6.1 Route the source of likelihoods by task

- **Single rich text → narrative-primary.** Likelihoods that drive updating are assessed directly (texture preserved), but every estimate carries externalized reasoning covering *both* the causal and the trace-production rationale. Updating is genuine multi-hypothesis Bayes with researcher-settable priors.
- **Many comparable cases → formal (CausalQueries).** Binarization is appropriate (shared variables across cases), structural dependence-handling pays off, and counterfactual/population estimands are identified. Source of truth for cross-case numbers.

No single analysis is scored by two engines, so there is no contradiction surface between them.

### 6.2 Likelihood elicitation: bands, comparability, priors ≠ likelihoods

**Bands, not false-precise points.** LLM-elicited continuous values (`0.85` vs `0.80`) imply a precision the elicitation cannot support. Likelihoods are elicited as **bands**, and sensitivity is run over bands:

| Label | Likelihood ratio range |
|---|---|
| Strongly disconfirming | `< 0.2` |
| Moderately disconfirming | `0.2 – 0.5` |
| Weak / uninformative | `0.5 – 2` |
| Moderately confirming | `2 – 5` |
| Strongly confirming | `> 5` |

(Interval-valued continuous likelihoods are an acceptable alternative; the requirement is that uncertainty be represented and propagated, not that values be discrete.)

**Multi-hypothesis, not two-way.** Likelihoods are elicited comparatively across the full hypothesis set, not as pairwise `H` vs `¬H` (which becomes incoherent when "¬H" is a blob of rival hypotheses).

**Likelihood-comparability requirement (new in v2).** Whenever a narrative estimate and a formal-model estimate are to be compared (§6.3), they must:

- condition on the **same hypothesis event**,
- refer to the **same evidence event**, and
- separate uncertainty about *hypothesis truth* (which belongs in the prior over `H`) from uncertainty about *evidence production* (which belongs in the trace-production term).

Without this, disagreement between the two is *definitional, not diagnostic* — see the reworked example below.

**Priors are researcher inputs.** A uniform prior is a defensible *reference* choice only when justified; priors are an explicit input, and sensitivity ranges over them.

#### 6.2.1 Formal band semantics and propagation

A band is not a number; it is an interval on the likelihood ratio. The posterior odds between hypotheses `Hᵢ`, `Hⱼ` after `m` (clustered, §6.5) evidence items is

```
O_ij^post = O_ij^prior · ∏_m LR_m(i,j),     LR_m(i,j) = P(E_m | H_i) / P(E_m | H_j)
```

where each `LR_m(i,j)` is supplied only as a band → an interval `[lo, hi]`. Open-ended bands are bounded by the implementation's floor and cap (`LR_FLOOR = 0.05`, `LR_CAP = 20`): "strongly disconfirming" = `[0.05, 0.2]`, "strongly confirming" = `[5, 20]`. The output object and how the interval propagates must be stated; we specify three modes, in decreasing strength of assumption:

| Mode | Assumption | Output | Use |
|---|---|---|---|
| **Monte Carlo (default)** | each LR is **log-uniform** on its band | a posterior-odds *distribution* → median + central interval, and **rank-stability** = fraction of draws preserving each pairwise order | the standard report |
| **Interval arithmetic** | none beyond `[lo, hi]` | guaranteed posterior-odds **bounds** (worst/best case) | conservative robustness check |
| **Ordinal-only** | bands carry order, not magnitude | a **dominance** relation over hypotheses, no numbers | when even interval endpoints are indefensible |

Log-uniform is chosen because likelihood ratios live naturally on a log scale; the assumption is declared, not hidden, and the interval-arithmetic mode is always reported alongside as a no-assumption bound. The headline single-text output is therefore **a posterior-odds interval per hypothesis plus a rank-stability fraction**, never a single "posterior probability." For a dependent cluster, a **single joint band** is elicited (§6.5), not a product of member bands — this is where double-counting is prevented mechanically.

### 6.3 The DAG as auditor: structured second-pass criticism (not independent corroboration)

For a single text, the formal graph earns its keep not by computing the answer but by **cross-checking** the narrative estimate on the evidence that matters. The disagreement between a holistic likelihood and a structurally-derived one is the signal: it indicates either an incoherent/double-counting implicit model, or a mis-specified/over-lossy graph, **or** a trace-production assumption that one side encoded and the other did not.

Because the estimator, modeler, and reconciler may all be instances of the same base model reading the same text, their independence is limited; agreement is **support, not proof**. The audit is therefore framed as **structured second-pass criticism**, and independence is engineered, not assumed:

- different model families where available;
- **blinded roles** — the modeler does not see the estimator's number;
- randomized elicitation order; evidence-redaction probes;
- human spot-checks on top-driver evidence;
- retained disagreement logs for reproducibility.

**Three grades of independence — and we mostly buy the weakest.** Independence is not one thing:

| Grade | Means | What it rules out | Achieved here? |
|---|---|---|---|
| Procedural | different prompts, blinded roles, randomized order | prompt-leak, anchoring | **yes** |
| Model | different model families | family-specific quirks | sometimes (when a second family is available) |
| Epistemic | different information / differently-trained priors | shared training-data artifacts, shared historiographic stereotypes | **rarely** |

Only **epistemic** independence makes agreement strong evidence. If every model has absorbed the same "fiscal-crisis → French Revolution" trope, blinded agents can agree from a shared cultural prior, not from the evidence. The audit's evidential weight is therefore discounted accordingly: agreement is treated as *removal of procedural error*, not as corroboration of substance. Human spot-checks on top drivers, and (where possible) a genuinely different information source, are the only routes to epistemic independence, and are flagged as the highest-value — and currently weakest — link.

**Faithful-abstraction (translation) contract.** Before any number is compared, the auditor must certify that its local graph is a *faithful abstraction* of the narrative object — otherwise the audit silently changes the claim being audited. The auditor logs, for each translation:

| Narrative object | DAG representation | Logged loss check |
|---|---|---|
| mechanism | nodes + edges | which mechanism steps were dropped? |
| evidence proposition | observation node | what semantic detail was compressed? |
| trace process | measurement / survival nodes | what source-selection assumptions were *added*? |
| hypothesis | graph query | did the claim change in translation? |

If the translation loss is material (e.g., the query no longer expresses the same hypothesis), the comparison is **void** and reported as "not auditable at this granularity," not as agreement or disagreement.

> **Reworked worked example (v2).** Estimator (conditioning on `F=1`, i.e. fiscal crisis occurred and is the driver): "grievances would be fiscal → **strongly confirming**." Modeler, if it instead *marginalizes over* `P(F=1)`, returns a weaker band. **In v1 this gap was reported as a diagnostic disagreement and "resolved toward 0.8." That was an error:** the two were computing different conditional quantities, so the disagreement was *definitional*. Under the comparability requirement, both condition on `F=1`; the spurious gap disappears. The *genuine* diagnostic question then surfaces in the **trace-production** term: are fiscal grievances over-represented in the *cahiers* because they were administratively solicited? If so, the modeler lowers the trace-production factor, and that — not a prior on `F` — is the documented, auditable adjustment.

### 6.4 The reconciler is a protocol, not a discretion

The reconciler's power is constrained to a fixed action space, so it cannot launder arbitrary judgment into a number:

| Diagnosis | Permitted action |
|---|---|
| Missing mechanism/node in graph | Revise graph; recompute the formal likelihood |
| Unsupported narrative link | Lower the likelihood **band**; cite the missing evidence |
| Evidence dependence | Cluster the evidence; recompute a **joint** likelihood (§6.5) |
| Trace-production mismatch | Adjust the **trace-production** term explicitly; leave the causal term |
| Prior mismatch | Separate prior from likelihood; **adjust the prior, never the likelihood** |
| Irreducible disagreement | Preserve an interval / model-averaged uncertainty; do not collapse |

The reconciler emits an adjusted **band** plus the named diagnosis and action; "split the difference" is not in its action space.

### 6.5 Dependence handling via an evidence graph

The naive engine's characteristic error is multiplying likelihoods of dependent evidence (five reports of one fact counted as five). Dependence has distinct types requiring distinct treatment:

| Dependence type | Example | Treatment |
|---|---|---|
| Common source | Five articles cite one speech | Collapse / downweight |
| Common cause | Many grievances from one fiscal shock | Model as mediated evidence |
| Sequential | One event leaves a later trace | Temporal ordering |
| Selection / survival | Only certain documents survive | Trace-production / missingness model |
| Interpretive | One extractor produces overlapping claims | Audit the extraction stage |

Naming the types is not enough; the hard part is *deciding when two items are dependent and how much to downweight*. That requires an **evidence graph** distinct from the causal event graph.

**Schema.** Node types: `Event`, `Institution`, `ArchiveProcess`, `Source`, `Document`, `Extractor`, `Claim`, `ExtractedProposition`. Edge types:

```
Event        → Claim                  (the event gives rise to a claim)
Institution  → Source                 (who produced the source)
ArchiveProcess → Document             (survival / preservation)
Source       → Document → Claim       (a document carries claims)
Extractor    → ExtractedProposition   (the pipeline's reading)
Claim        → ExtractedProposition   (what the proposition is about)
```

**Operational clustering rule.** Two extracted propositions are **dependent** iff they share an ancestor in the evidence graph that explains their evidential similarity (a common `Source`, `Document`, `ArchiveProcess`, or originating `Event`). Dependent propositions are placed in one **cluster**, and the cluster receives **one joint likelihood band** (§6.2.1) — not the product of member bands. Independent clusters multiply; members within a cluster do not.

**Worked duplicate example.** A reported ministerial speech generates: (a) five newspaper articles, three of which copy the same wire report and two of which rely on the same official transcript; (b) a memoir recalling the speech 30 years later; (c) a police report on public reaction. The evidence graph yields:

- the three wire-copying articles + two transcript-based articles share the parent `Document = official transcript/wire` → **one cluster**, one joint band (≈ the band of a single independent report, not five);
- the memoir has a distinct path (`Event → Claim`, decades later, high distortion) → **separate** item, with a trace-production discount for recall decay;
- the police report observes *reaction*, not the speech content → a **different `Claim`** → independent item.

Result: five "sources" collapse to **three** evidentially-distinct contributions, and the speech is not counted five times. Clustering decisions are logged for audit.

### 6.6 Missingness model (absence of evidence)

Hoop tests turn on expected-but-absent evidence, which is dangerous to automate because "not in the supplied text" ≠ "did not occur." The architecture distinguishes four states:

| State | Meaning |
|---|---|
| Absent from world | the event/trace did not exist |
| Absent from source | the source omits it |
| Absent from extraction | the pipeline failed to detect it |
| Absent from query | the pipeline never looked |

An absence disconfirms a hypothesis **only** to the extent the evidence *would probably have been observed if it existed* — the **observability** `P(observe E | E true)`, supplied by the trace-production model. A blanket "always qualitative" rule is too toothless: genuinely probative silences (no tax record, parliamentary debate, or police file mentions a supposedly central mechanism) are then never used. We therefore graduate by observability:

| Observability if true | Treatment of the absence |
|---|---|
| low | qualitative note only (no update) |
| moderate | weak disconfirmation |
| high | quantitative hoop-test penalty |
| near-certain | strong disconfirmation |

The four-state distinction above (world / source / extraction / query) governs *eligibility*: only "absent from world (given the source would carry it)" can disconfirm; "absent from extraction" or "absent from query" are pipeline failures and must be excluded. (The reference implementation currently takes the most conservative stance — its absence pass is qualitative-only and does not feed the Bayesian update, §9 — which corresponds to treating all observability as "low"; graduating it per the table above is a defined next step.)

### 6.7 Top-driver gating

Triangulation (estimator + modeler + reconciler, with iteration) is costly and unnecessary for most evidence. Run the cheap narrative estimate everywhere; trigger the audit selectively. But a pure high-`|log(LR)|` trigger misses a real failure mode: **many individually-weak items that share a source or motif** can dominate the posterior collectively while no single one looks like a top driver. The audit is therefore triggered on the **union** of:

1. **high `|log(LR)|`** items (individually posterior-moving);
2. **high-volume / repeated-motif clusters** from the evidence graph (§6.5) — a cluster whose *combined* naive contribution is large, even if each member is weak;
3. items the reconciler or partition-audit flagged as contested.

This concentrates scrutiny where error would change conclusions *and* where correlated weak evidence would otherwise distort the posterior silently. Any cap on audited evidence is logged, not silently applied.

### 6.8 A complete worked run (illustrative)

This miniature run exercises every component. Numbers are illustrative, chosen to show the *mechanics and the honest conclusion*, not to settle the historiography.

**Step 1 — Partition (§5.1).** Mutually-exclusive whole stories plus a residual:

- `H₁` fiscal crisis was the primary driver
- `H₂` ideological radicalization was the primary driver
- `H₃` elite fragmentation was the primary driver
- `H₀` residual: none of these / a genuinely conjunctural combination

The exclusivity self-check passes (each is a distinct *primary* driver); the conjunctural story "fiscal → elite fragmentation → ideological mobilization" is **not** added alongside `H₁₋₃` — it is absorbed into `H₀` (or deferred to the formal model), per rule 3.

**Step 2 — Priors (researcher-set, justified).** Longstanding debate, no consensus → spread, with real mass on the residual: `P(H₁)=0.30, P(H₂)=0.25, P(H₃)=0.25, H₀=0.20`.

**Step 3 — Evidence and the evidence graph (§6.5).**

- `E₁` *cahiers de doléances* dominated by fiscal grievances
- `E₂` regional administrative grievance summaries, also fiscal-dominated — **but produced by the same 1788–89 royal solicitation** as `E₁`
- `E₃` surviving provincial elite correspondence showing coordination breakdown

Graph: `E₁, E₂` share parent `ArchiveProcess/Institution = royal solicitation of grievances` → **cluster C = {E₁, E₂}**, one joint band. `E₃` is independent.

**Step 4 — Bands, with trace-production adjustment (§4, §6.2.1).** Naively, cluster C reads *strongly confirming* for `H₁`. The trace-production model intervenes: grievances were *solicited through fiscal-administrative channels*, so fiscal content is over-represented by construction. The joint band for C is discounted to **moderately confirming (2–5)** for `H₁`, and — crucially — C is **one** band, not `(>5)×(>5)`.

| Item | vs `H₁` | vs `H₂` | vs `H₃` |
|---|---|---|---|
| Cluster C (fiscal grievances, solicited) | moderately confirming `2–5` | weak `0.5–2` | weak `0.5–2` |
| `E₃` (elite correspondence) | weak `0.5–2` | weak `0.5–2` | moderately confirming `2–5` |

**Step 5 — Reconciler log (§6.4).**

```
audit C:
  estimator band (vs H1): STRONGLY confirming (>5)
  modeler: local graph flags node `solicitation → grievance-content`
  diagnosis: trace-production mismatch  (not a prior issue)
  action:   adjust trace-production term; lower joint band to MODERATELY confirming (2–5)
  diagnosis: E1,E2 share parent `royal solicitation` (common process)
  action:   cluster {E1,E2}; elicit ONE joint band  (no product)
  translation check: query "primary driver = fiscal" preserved → audit valid
```

**Step 6 — Posterior (Monte Carlo, log-uniform within bands; §6.2.1).** Updating the priors by C and `E₃` and sampling yields, illustratively:

| Hypothesis | Posterior interval (central) | 
|---|---|
| `H₁` fiscal | ~0.30 – 0.48 |
| `H₃` elite | ~0.24 – 0.42 |
| `H₂` ideology | ~0.08 – 0.16 |
| `H₀` residual | ~0.10 – 0.20 |

**Rank-stability:** `H₁ > H₂` holds in ~96% of draws; `H₁ > H₃` holds in only ~58%.

**Step 7 — Sensitivity / conclusion.** Varying cluster C across its full plausible band (weak ↔ strongly-confirming) and the priors over their stated ranges: the conclusion **"`H₂` (ideology-primary) is disfavored"** is robust; the ordering **`H₁` vs `H₃`** is *not* — they are not distinguishable from this evidence. The honest report is therefore: *"the evidence favors a fiscal or elite-fragmentation primary driver over an ideology-primary one, but does not discriminate between fiscal and elite; residual mass remains for a conjunctural account."* Note what the trace-production step bought: without it, the un-discounted, un-clustered fiscal grievances would have produced a spuriously decisive `H₁`.

---

## 7. Identifiability and comparability boundaries

**Single-case identifiability.** A quantified counterfactual or population effect from a single case is, informationally, the slope of a line from one point: the data does not contain it. A formal DAG *can* run on a single case (Humphreys & Jacobs' original within-case use), yielding coherent updating and dependence-handling, but its single-case counterfactuals are **prior-dominated**. Identified counterfactual/population quantities require **many cases with variation**. This is an information limit, not a method preference, and it is why §5 forbids the single-text path from claiming them.

**Cross-case comparability.** The formal path is legitimate only when cases are genuinely comparable. Binarizing "fiscal crisis" across Bourbon France, late-Qing China, and Weimar Germany as a common `F=1` can manufacture false comparability and reproduce the very pseudo-quantification this paper criticizes. Cross-case eligibility therefore requires: common variable definitions; comparable measurement/trace-production regimes; sufficient variation in treatment and outcome; a defensible case-selection rule; an explicit missing-data model; and sensitivity to alternative codings. (The reference implementation already exhibits this failure mode when running uniformly "successful" cases against a fixed model, which collapses to all-`1`s and leaves nothing to estimate — a documented limitation, §9.)

---

## 8. Validation: how we would know it works

The architecture's central empirical claim is that **the local causal-graph audit improves coherence and dependence-handling over a narrative-only baseline**. That is testable, and the paper commits to testing it. A minimal protocol:

| Design | What it tests |
|---|---|
| **Auditor ablation** (headline) | Does the DAG audit improve results vs. narrative-only, at its cost? |
| Synthetic cases with known structure | Does the system recover known dependencies and rankings? |
| Calibration on band assignments | Do likelihood bands behave sensibly across many items? |
| Paraphrase robustness | Do conclusions survive rewording of the source? |
| Source-contamination test | Are repeated/copied reports correctly *not* double-counted? |
| Missing-evidence test | Is absence handled cautiously (no over-disconfirmation)? |
| Expert agreement (with caution) | Agreement with human process tracers, noting bias-reproduction risk |

The objective in §6 (auditability, dependence-handling, calibrated sensitivity per cost) is what these designs operationalize. Calibration against *truth* is hard for unique historical events; the protocol therefore leans on synthetic recovery, ablation, and internal-coherence tests, with expert agreement as a secondary, bias-aware check.

### 8.1 Scoring rules and a comparable baseline

A test menu is not a validation design without metrics, a baseline, and thresholds.

| Quantity | Scoring rule |
|---|---|
| Rank accuracy (which hypothesis the evidence favors) | Kendall's τ / top-1 accuracy vs. ground-truth ordering on synthetic cases |
| Probabilistic calibration | log score and Brier score on band-implied posteriors |
| Dependence detection | precision/recall of the evidence-graph clustering vs. injected duplicates |
| Duplicate overcounting | ratio of posterior shift from a planted copied report to that from an independent one (target ≈ 1) |
| Absence handling | false-disconfirmation rate on planted observability-low absences (target ≈ 0) |
| Cost | model calls, tokens, wall-time, and human-review minutes **per audited (top-driver) item** |

**Baseline.** "Narrative-only" must be the **operationally identical pipeline with the audit and evidence-graph clustering removed** — same extraction, same priors, same bands — so the ablation isolates the audit's contribution and nothing else.

**Benchmark construction caution.** Synthetic cases generated from the same assumptions the model uses are too easy and inflate results; generators must be **held-out / adversarial** (different causal structures, planted dependence and survival bias). Expert-coded historical cases supply external validity but **expert agreement is not ground truth** — inter-annotator variation is measured and reported, not assumed away.

### 8.2 Falsification thresholds (provisional)

The architecture must be able to fail. Provisional pass criteria — exact values to be preregistered:

> The auditor **passes** only if, relative to the narrative-only baseline, it (i) improves dependence-detection F1 by ≥ X, (ii) brings duplicate-overcounting ratio within ±Y of 1.0, (iii) holds posterior-rank accuracy on synthetic cases within Z of baseline-or-better, and (iv) achieves this at ≤ C additional cost per audited item. Failing (i)–(iii) **falsifies** the central claim that the audit improves inference.

### 8.3 Against "ritualized Bayesianism"

Bands, graphs, reconciliation tables, and robustness ranges can become an elaborate ritual that still launders opaque judgment. The antidote is not more architecture; it is **proper scoring** (calibration + discrimination measured against held-out outcomes) and **protocol preregistration**: the hypothesis partition (§5.1), priors, band scheme, audit triggers, and scoring rules are frozen *before* the testing pass (the `--review` checkpoint is the freeze point). Architecture without scoring is decoration; the validation protocol is what makes the architecture answerable.

---

## 9. Scope boundaries and companion work

To keep this a methodology paper rather than a research program, two areas are **scoped out** of the core and named as companion work:

- **Upstream extraction validity.** This paper begins after evidence and hypotheses are extracted, yet automated process tracing often fails earlier — segmentation, event extraction, coreference, chronology, hypothesis exhaustiveness and mutual-exclusivity. Bayes' rule cannot rescue a bad evidence representation. A companion *extraction-validity* note should cover source characterization, event/proposition extraction, hypothesis generation, and mutual-exclusivity/chronology audits as preconditions for the inference layer described here.
- **Full validation protocol and benchmark construction.** §8 specifies the tests; building the synthetic generators, the expert-coded benchmark, and the calibration harness is a separate empirical effort.

**Relationship to the reference implementation (implementation status).** This is a planning/architecture document, not a results paper: it describes a *target* architecture, and the companion pipeline realizes some of it today. To avoid letting a named stage stand in for a demonstrated capability, the table below states honest status — **Implemented** (in the codebase, exercised by tests), **Partial** (present but not in the form the methodology specifies), or **Planned** (designed here, not built). It claims *engineering/architectural* status only; substantive analytic validity is the separate, not-yet-executed question of §8.

| Capability (§) | Status | Where / form today | Gap to methodology target |
|---|---|---|---|
| Bayesian likelihood updating (§3) | **Partial** | `pt/bayesian.py`, odds-space update | two-way `H` vs `¬H` + uniform prior; not multi-hypothesis comparative |
| Van Evera diagnostic classification (§3.2) | **Implemented** | `pass3_test` prompt + schema | — |
| Mechanical robustness signal (§6) | **Implemented** | `pt/bayesian._compute_robustness` | — |
| Sensitivity over likelihoods (§6.5) | **Implemented** | `pt/bayesian._run_sensitivity` | — |
| Sensitivity over priors (§6.2) | **Planned** | — | priors not yet an input |
| Researcher-settable priors (§5, §6.2) | **Planned** | hardcoded uniform `1/n` | elicited/justified priors |
| Likelihood bands + propagation (§6.2.1) | **Planned** | continuous point LRs, capped/relevance-discounted | bands, log-uniform MC, rank-stability |
| Multi-hypothesis comparative likelihoods (§6.2) | **Planned** | pairwise vs-`¬H` | full-set comparative elicitation |
| Estimand discipline & residual `H₀` (§5, §5.1) | **Partial** | exclusivity self-check in prompt; `--review` freeze | explicit residual hypothesis; estimand stated in output |
| Externalized causal-**and-trace** reasoning (§4) | **Partial** | per-evaluation `justification` text | trace-production reasoning not separated from causal |
| Trace-production model (§4) | **Planned** | — | not represented |
| Evidence graph + dependence clustering (§6.5) | **Planned** | independent multiplication; LR cap + relevance as crude dampers | explicit evidence graph, joint cluster bands |
| Triangulation auditor + translation contract (§6.3) | **Planned** | — | estimator/modeler/reconciler not built |
| Reconciler protocol (§6.4) | **Planned** | — | constrained action table |
| Top-driver gating (§6.7) | **Partial** | `top_drivers` computed | no auditor yet to gate; cluster/volume trigger absent |
| Missingness model w/ observability bands (§6.6) | **Partial** | qualitative-only absence pass (`pass_absence`), excluded from updating | observability grading |
| Cross-case formal estimation (§7) | **Implemented** | `pt.multi` + `cq_bridge` → CausalQueries | — |
| Cross-case eligibility criteria (§7) | **Planned** | documented all-`1`s failure mode | enforced comparability checks |
| Validation protocol / auditor ablation (§8) | **Planned** | `make check` (software tests only) | methodological evaluation not executed |

The Implemented rows establish **software integrity and the scaffolding**; they do **not** establish analytic validity, which §8 exists to earn. The Planned/Partial rows are intended for **staged adoption**; engineering sequencing is deliberately left to a separate implementation note.

---

## 10. Why an LLM changes what is feasible

Classical process tracing could not externalize reasoning, run blinded parallel estimators, and reconcile them *at scale*; the labor was prohibitive, so analysts left the model implicit (narrative) or paid full formalization once (formal). Language models change the **cost structure**, not (by themselves) the reliability. The precise claim:

> LLMs reduce the marginal cost of producing structured, inspectable likelihood and trace-production rationales, and of running blinded parallel estimation and protocol-constrained reconciliation. **Whether those rationales improve inferential reliability is an empirical question, answered only by the auditor-ablation protocol (§8), not by the affordances themselves.**

"Feasible" here means *technically and economically* feasible. Epistemic reliability (§6.3) and institutional auditability are claims the validation protocol must earn, not consequences of using an LLM. §7 bounds the ambition regardless of cost.

---

## 11. Limitations

- **Correlated model errors** can make blinded agents agree while both wrong; agreement is support, not proof (§6.3).
- **Reconciler scope.** Even constrained, the action table encodes judgment; its choices are logged for audit, not presumed correct.
- **Single-case counterfactuals remain prior-dominated** and are excluded from the single-text estimand (§5, §7).
- **Trace-production models are themselves assumptions, and may be near-speculative.** In many cases the record-production process (how much was lost, censored, standardized, copied, selectively preserved) is poorly known. Visibility does not make a bad assumption good. The discipline is to mark each trace-production term as *estimated* vs. *speculative*; when speculative, widen the band (§6.2.1) or fall back to qualitative treatment rather than assert a discount.
- **Epistemic independence is rarely achieved** (§6.3). The audit mostly removes procedural error; it is weak against shared training-data priors and historiographic stereotypes. Treat audit agreement accordingly.
- **Cost.** Triangulation is expensive; top-driver gating is the control, and any cap is logged (§6.7).
- **No optimality claim.** This is a principled approximation against a stated objective, not a proven optimum; the true Bayesian optimum (a coherent joint posterior averaged over causal structures) is intractable for real cases.

---

## 12. Summary

**Thesis.** Single-text historical inference should report **comparative posterior support over an explicitly partitioned set of mutually-exclusive explanatory hypotheses** — not identified causal effects. Because a likelihood judgment confounds *event causation* with *evidence production*, automated systems should externalize **both** the substantive causal model and the trace-production model, propagate likelihood uncertainty as **bands/intervals** with declared semantics, and audit the posterior-sensitive and dependence-sensitive evidence through a **local causal graph whose translation losses are logged** and whose disagreements are resolved by a **constrained protocol**, never by averaging.

The choice was never narrative *versus* formal. Both are Bayes' rule; they differ in how hypotheses, mechanisms, and dependence are represented, and — operationally — in how much of the model behind the likelihood is externalized. A "direct" likelihood is an undocumented model spanning both historical causation and evidence generation, and conflating those two is a core error. No single text can identify counterfactual or population quantities, so those are reserved for the multi-case path where the data support them. Language models lower the *cost* of externalization and audit; they do not, by themselves, establish reliability — that is what the validation protocol (§8) exists to earn. Identifiability bounds the claims; preregistered scoring keeps the architecture from becoming ritual; and the headline empirical question — does the audit beat narrative-only — is left testable, and falsifiable, by design.

---

## References

- Fairfield, T., & Charman, A. (2017). *Explicit Bayesian Analysis for Process Tracing: Guidelines, Opportunities, and Caveats.* Political Analysis 25(3). https://eprints.lse.ac.uk/69203/2/Fairfield_Explicit%20bayesian%20analysis_author_2017%20LSERO.pdf
- Fairfield, T., & Charman, A. (2022). *Social Inquiry and Bayesian Inference: Rethinking Qualitative Research.* Cambridge University Press.
- Humphreys, M., & Jacobs, A. (2015). *Mixing Methods: A Bayesian Approach.* American Political Science Review 109(4).
- Humphreys, M., & Jacobs, A. (2023). *Integrated Inferences: Causal Models for Qualitative and Mixed-Method Research.* Cambridge University Press. https://macartan.github.io/integrated_inferences/
- *CausalQueries: Make, Update, and Query Binary Causal Models* (R package). https://cran.r-project.org/package=CausalQueries — Guide: https://macartan.github.io/causalmodels/process-tracing.html
- Van Evera, S. (1997). *Guide to Methods for Students of Political Science.* Cornell University Press.
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science.* Cambridge University Press.
- Mahoney, J. (2012). *The Logic of Process Tracing Tests in the Social Sciences.* Sociological Methods & Research 41(4). (INUS conditions and necessity/sufficiency in process tracing.)
- Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* Journal of the American Statistical Association 102(477). (Log and Brier scores; calibration vs. discrimination.)
- *Two fully specified Bayes factors for hypothesis testing and sensitivity analysis in process tracing* (2026). arXiv. https://arxiv.org/html/2606.16683

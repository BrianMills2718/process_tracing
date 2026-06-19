# A Bayesian Audit Architecture for Automated Process Tracing

### Externalized LLM Likelihoods, Trace-Production Models, and Local Causal-Graph Audits for Single-Text Historical Inference

**Status:** Draft methodology white paper — **v2**
**Scope:** Single-text and cross-case causal inference from historical narrative
**Audience:** Methodologists, computational social scientists, and engineers building automated causal-inference pipelines

---

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

---

## 6. The architecture

The objective — not "optimality," but a stated target — is to **maximize auditability, dependence-handling, and calibrated sensitivity per unit of model-call cost**, relative to the naive baseline of multiplying LLM-elicited likelihoods. The components below each serve that objective.

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

### 6.3 The DAG as auditor: structured second-pass criticism (not independent corroboration)

For a single text, the formal graph earns its keep not by computing the answer but by **cross-checking** the narrative estimate on the evidence that matters. The disagreement between a holistic likelihood and a structurally-derived one is the signal: it indicates either an incoherent/double-counting implicit model, or a mis-specified/over-lossy graph, **or** a trace-production assumption that one side encoded and the other did not.

Because the estimator, modeler, and reconciler may all be instances of the same base model reading the same text, their independence is limited; agreement is **support, not proof**. The audit is therefore framed as **structured second-pass criticism**, and independence is engineered, not assumed:

- different model families where available;
- **blinded roles** — the modeler does not see the estimator's number;
- randomized elicitation order; evidence-redaction probes;
- human spot-checks on top-driver evidence;
- retained disagreement logs for reproducibility.

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

This requires an **evidence graph** distinct from the causal event graph, with nodes for *source*, *claim*, *observation*, and *extracted proposition*. Dependence is flagged on the evidence graph **before** likelihoods are combined. "Cluster the evidence" (§6.4) is defined operationally as: detect a shared parent in the evidence graph, then elicit one joint likelihood for the cluster.

### 6.6 Missingness model (absence of evidence)

Hoop tests turn on expected-but-absent evidence, which is dangerous to automate because "not in the supplied text" ≠ "did not occur." The architecture distinguishes four states:

| State | Meaning |
|---|---|
| Absent from world | the event/trace did not exist |
| Absent from source | the source omits it |
| Absent from extraction | the pipeline failed to detect it |
| Absent from query | the pipeline never looked |

An absence disconfirms a hypothesis **only** to the extent the evidence *would probably have been observed if it existed* — a quantity supplied by the trace-production model. Absent that, absence findings are reported **qualitatively** and excluded from quantitative updating. (The reference implementation already takes this conservative stance: its absence pass is qualitative-only and does not feed the Bayesian update — see §9.)

### 6.7 Top-driver gating

Triangulation (estimator + modeler + reconciler, with iteration) is costly and unnecessary for most evidence. Run the cheap narrative estimate everywhere; trigger the audit only on the **evidence that moves the posterior** — the top drivers by `|log(LR)|`, already computed for sensitivity. This concentrates scrutiny where error would change conclusions, and is the cost term in the §6 objective. Any cap on audited evidence is logged, not silently applied.

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

---

## 9. Scope boundaries and companion work

To keep this a methodology paper rather than a research program, two areas are **scoped out** of the core and named as companion work:

- **Upstream extraction validity.** This paper begins after evidence and hypotheses are extracted, yet automated process tracing often fails earlier — segmentation, event extraction, coreference, chronology, hypothesis exhaustiveness and mutual-exclusivity. Bayes' rule cannot rescue a bad evidence representation. A companion *extraction-validity* note should cover source characterization, event/proposition extraction, hypothesis generation, and mutual-exclusivity/chronology audits as preconditions for the inference layer described here.
- **Full validation protocol and benchmark construction.** §8 specifies the tests; building the synthetic generators, the expert-coded benchmark, and the calibration harness is a separate empirical effort.

**Relationship to the reference implementation (methodology register).** The companion pipeline already realizes parts of this architecture and points at the rest: a narrative-style single-text path (likelihood updating, Van Evera classification, mechanical robustness, sensitivity), a separate cross-case path bridging to `CausalQueries`, a qualitative-only absence pass consistent with §6.6, per-hypothesis top-driver identification supplying the §6.7 gate, and a documented cross-case false-comparability failure mode motivating §7. The gaps this methodology defines as next steps — a formal estimand, researcher-settable priors with prior-sensitivity, comparative multi-hypothesis bands replacing the two-way formulation, externalized causal-and-trace reasoning per likelihood, the blinded triangulation auditor with the §6.4 protocol, and the evidence-graph dependence layer — are intended for **staged adoption**. (Engineering sequencing is deliberately left to a separate implementation note.)

---

## 10. Why an LLM changes what is feasible

Classical process tracing could not externalize reasoning, run blinded parallel estimators, and reconcile them *at scale*; the labor was prohibitive, so analysts left the model implicit (narrative) or paid full formalization once (formal). Language models make three things affordable: (1) externalized causal-and-trace reasoning emitted as structured output per likelihood; (2) blinded, separately-framed estimation so disagreement is informative (subject to the correlated-error caveat of §6.3); and (3) protocol-constrained reconciliation that localizes disagreement to a specific assumption and drives iterative refinement. This makes "externalize and check coherence" operational — while §7 bounds the ambition.

---

## 11. Limitations

- **Correlated model errors** can make blinded agents agree while both wrong; agreement is support, not proof (§6.3).
- **Reconciler scope.** Even constrained, the action table encodes judgment; its choices are logged for audit, not presumed correct.
- **Single-case counterfactuals remain prior-dominated** and are excluded from the single-text estimand (§5, §7).
- **Trace-production models are themselves assumptions** — better externalized and criticizable than implicit, but not ground truth.
- **Cost.** Triangulation is expensive; top-driver gating is the control, and any cap is logged (§6.7).
- **No optimality claim.** This is a principled approximation against a stated objective, not a proven optimum; the true Bayesian optimum (a coherent joint posterior averaged over causal structures) is intractable for real cases.

---

## 12. Summary

The choice was never narrative *versus* formal. Both are Bayes' rule; they differ in how hypotheses, mechanisms, and dependence are represented, and — operationally — in how much of the model behind the likelihood is externalized. A "direct" likelihood is an undocumented model spanning *both* historical causation and evidence generation, and conflating those two is a core error. No single text can identify counterfactual or population quantities, so the single-text estimand is fixed as comparative posterior odds over specified explanatory hypotheses. The proposed architecture keeps narrative texture as the inferential substrate, forces the implicit causal-and-trace models into the open, audits only posterior-moving likelihoods with a blinded local causal graph under a constrained reconciliation protocol, handles dependence on an explicit evidence graph, treats absence through a missingness model, and reserves identified counterfactual/population estimands for the multi-case path where the data support them. Language models make this affordable at scale; identifiability bounds its claims; and the headline empirical question — does the audit beat narrative-only — is left testable by design.

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
- *Two fully specified Bayes factors for hypothesis testing and sensitivity analysis in process tracing* (2026). arXiv. https://arxiv.org/html/2606.16683

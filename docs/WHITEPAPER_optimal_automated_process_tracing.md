# Toward Optimal Automated Process Tracing

### An LLM-Native Bayesian Methodology for Causal Analysis of Historical Events

**Status:** Draft methodology white paper
**Scope:** Single-text and cross-case causal inference from historical narrative
**Audience:** Methodologists, computational social scientists, and engineers building automated causal-inference pipelines

---

## Abstract

Automated systems that read a historical text and output "what caused this" tend to do one of two things badly: produce confident prose with no inferential discipline, or impose a rigid formal model that discards the very textual richness that makes historical evidence informative. We argue that both failure modes stem from a misunderstanding of what separates the two established traditions of Bayesian process tracing — the *narrative / likelihoodist* tradition (Fairfield & Charman) and the *formal / model-based* tradition (Humphreys & Jacobs). The two are not rival theories of inference; they are the same Bayesian inference differing in **one** place: where the likelihood `P(evidence | hypothesis)` comes from. The narrative tradition assesses it by holistic judgment; the formal tradition derives it from an explicit causal graph. Crucially, a holistic likelihood judgment **always encodes an implicit causal model** — the difference is only whether that model is written down.

From this observation we derive a methodology for *optimal feasible* automated process tracing. The theoretical optimum — a single coherent joint posterior with uncertainty over causal structures — is intractable. Among feasible approximations, the right design depends on what the data can *identify*: a single rich text cannot identify quantified counterfactuals or population effects no matter how it is modeled, while many comparable cases can. We therefore propose a **task-routed, LLM-native triangulation architecture**: for a single text, a narrative likelihood engine whose every estimate carries externalized reasoning, audited for coherence and hidden dependence by an independently-constructed formal causal graph, with disagreements *diagnosed* (not averaged) by a reconciler, and the expensive audit gated to the evidence that actually moves the posterior; for many cases, the formal model (CausalQueries) as the source of truth. Van Evera's diagnostic tests are retained as a human-legible labeling layer over the likelihoods. The result keeps narrative texture, gains structural coherence and dependence-handling where it matters, and is honest about the one thing no method can recover from a single case.

---

## 1. The problem

Given a historical text, we want to infer the causal structure behind the events it describes: which of several competing explanations the evidence supports, how strongly, and how robust that conclusion is. "Automated process tracing" attempts to do this end to end — extract evidence and hypotheses, test each hypothesis against the evidence, update beliefs, and synthesize a verdict.

Two failure modes recur:

1. **Pseudo-quantification.** The system emits "posterior probabilities" produced by machinery that is not a coherent probability model — typically per-evidence likelihood ratios multiplied together under an unstated independence assumption, then renormalized to look like a distribution. The numbers invite trust they have not earned.
2. **Over-formalization.** The system forces the problem into a rigid causal graph over binary variables, discarding the discriminating texture of the evidence and committing to a structure the text does not warrant.

Both are avoidable once the relationship between the two academic traditions is made precise.

---

## 2. Background: two traditions of Bayesian process tracing

**Narrative / likelihoodist (Fairfield & Charman 2017, 2022).** The analyst specifies mutually exclusive hypotheses, assigns explicit prior odds from background knowledge, and for each piece of evidence assesses a *likelihood ratio* — how expected the evidence is under one hypothesis versus another — then updates by Bayes' rule in odds space and conducts sensitivity analysis over priors and likelihoods. The hypotheses are holistic explanations; no formal graph is required. Van Evera's diagnostic tests (hoop, smoking-gun, doubly-decisive, straw-in-the-wind) describe the *shape* of these likelihood ratios.

**Formal / model-based (Humphreys & Jacobs 2015, 2023; the `CausalQueries` R package).** The analyst declares a causal directed acyclic graph (DAG) over (typically binary) nodes, places priors over *causal types* (how each node responds to its parents), and updates beliefs about those types from observed node values. Every hypothesis is a *query* against this one model. Process tracing (within-case) and cross-case correlational inference are unified in a single framework, and conditional dependence among evidence is handled structurally via d-separation.

These are routinely presented as alternatives. They are better understood as two settings of one dial.

---

## 3. The core insight: both are Bayesian; they differ in *one* place

All process tracing here is Bayes' rule:

```
P(H | evidence) ∝ P(H) · P(evidence | H)
```

The only substantive disagreement between the traditions is **where the likelihood `P(evidence | H)` comes from**:

- **Narrative:** `P(evidence | H)` is assessed **directly**, as a holistic judgment.
- **Formal:** `P(evidence | H)` is **derived** by propagating through a causal DAG with specified node-response parameters.

A worked contrast makes this concrete. Take hypothesis **H1** = "fiscal crisis caused the Revolution" and evidence **E** = "the 1789 *cahiers de doléances* overwhelmingly complained about taxes, not philosophy."

- *Narrative.* The analyst reasons "if fiscal crisis were the driver, grievances would track what hurts people → taxes" and writes `P(E | H1) = 0.85`. The number is an **input**: one judgment per (evidence, hypothesis) cell.
- *Formal.* The analyst introduces nodes `F` (fiscal crisis), `G` (grievances are fiscal in content), `R` (revolution); a structure `F → G`, `F → R`; and parameters such as `P(G=1 | F=1) = 0.9`, `P(G=1 | F=0) = 0.3`, plus a prior on `F`. Then `P(E | H1)` is an **output**, computed from the structure and parameters. "H1" is no longer a free-standing story but a query about the `F → R` link.

### 3.1 The decisive realization: the implicit model

When the analyst writes `P(E | H1) = 0.85` "directly," the reasoning that produced it — *fiscal driver → grievances track harm → taxes dominate* — **is a causal model, with a mechanism**. The narrative method does not make it explicit. Therefore:

> **The narrative method is formal inference with the causal model left implicit, undocumented, and thus neither checkable nor enforced for coherence.**

The difference between the traditions was never "model versus no model." There is *always* a causal model in the reasoning. The difference is **how much of that ever-present model is externalized** — written down, made criticizable, and used to enforce consistency and detect dependence.

### 3.2 Van Evera tests sit *above* the likelihood

Hoop / smoking-gun / doubly-decisive / straw-in-the-wind are labels for regions of the likelihood ratio, independent of how the likelihood was obtained:

- **Hoop:** `P(E | H)` high → absence of `E` strongly disconfirms `H` (necessary-ish).
- **Smoking gun:** `P(E | ¬H)` low → presence of `E` strongly confirms `H` (sufficient-ish).
- **Doubly decisive:** both. **Straw-in-the-wind:** weak both ways.

They are a human-legible presentation layer over the likelihoods, compatible with either source. They do not constitute a third method.

---

## 4. What is theoretically optimal

In the Jaynes/Cox sense of "probability as extended logic," the optimal inference is the single coherent posterior obtained from a complete joint model that encodes *all* relevant background knowledge and conditions on *all* the evidence. Pushed to its limit, the analyst is also uncertain about the **causal structure itself**, so the ideal is **Bayesian model averaging over causal structures**:

```
P(causal claim | all evidence) = Σ over structures  P(claim | structure, evidence) · P(structure | evidence)
```

This is the true optimum, and it is **intractable** for real historical cases — in elicitation (no one can specify priors over all structures and all parameters) and in computation. Every usable method is therefore a tractable approximation that *chooses what to externalize*:

| Approach | Externalized | Left implicit |
|---|---|---|
| Theoretical optimum | full joint distribution **+ uncertainty over structures** | nothing |
| Formal (Humphreys–Jacobs) | one (or few) causal model(s) + type priors | the choice of graph |
| Narrative (Fairfield–Charman) | likelihoods + reasoning + sensitivity | the causal structure |
| Naive automated tool | a few likelihood numbers | the structure **and** the reasoning behind the numbers |

The naive automated baseline is the *least* externalized of all: it does not even record the reasoning behind each number. Closing that gap — not choosing a "side" — is the primary opportunity.

---

## 5. The identifiability boundary (what no single text can give)

A quantified counterfactual ("probability the Revolution would not have occurred absent the fiscal crisis = 0.7") or a population effect ("fiscal crisis causes revolution in 60% of comparable cases") is, from a single case, like asking for the slope of a line from one point: **the information is not in the data.** A richer model does not manufacture it; it only makes the *assumptions* that fill the gap explicit and reports them back, dressed as a result.

Two consequences follow, and they are not matters of taste:

1. A formal DAG *can* be run on a single case (this is Humphreys & Jacobs' original within-case use), and it yields coherent updating, structural dependence-handling, and even case-level counterfactual queries — but those single-case counterfactuals are **prior-dominated**, because one case does not identify them from data.
2. **Identified** quantified counterfactuals and population effects with heterogeneity genuinely require **many cases** with variation. This is an information limit shared by every method, not a narrative-versus-formal distinction.

The practical reading: reserve the formal model's distinctive deliverables for the multi-case setting where they are identified, and do not present prior-dominated single-case counterfactuals as if the evidence produced them.

---

## 6. The methodology

We propose a **task-routed, LLM-native triangulation** architecture. Its design principles follow directly from Sections 3–5.

### 6.1 One Bayesian likelihood engine; route the *source* of likelihoods by task

- **Single rich text → narrative-primary.** The likelihoods that drive updating are assessed directly (texture preserved), but every estimate carries **externalized reasoning** (the implicit model, written down). Updating is genuine multi-hypothesis Bayes with researcher-settable priors.
- **Many comparable cases → formal (CausalQueries).** Binarization is natural (you genuinely want shared variables across cases), structural dependence-handling pays off, and counterfactual/population estimands are identified. This is the source of truth for cross-case numbers.

No single analysis is scored by two engines, so there is no contradiction surface between them.

### 6.2 The DAG as an *auditor*, not the calculator (triangulation)

For a single text, the formal graph earns its keep not by computing the answer but by **independently cross-checking** the narrative estimate. The disagreement between a holistic likelihood and a structurally-derived likelihood is itself the signal: it means either the narrative's implicit model is incoherent or double-counting, or the explicit graph is mis-specified or too lossy.

The mechanism is a three-role, parallel, *independent* process per audited likelihood:

1. **Estimator (narrative).** Produces `P(E | H)` with an explicit mechanism/reasoning chain.
2. **Modeler (formal).** Independently constructs a local causal graph and derives a comparable `P(E | H)`.
3. **Reconciler.** Compares the two **and their logic**, and — critically — **diagnoses the source of any disagreement rather than averaging it**:
   - graph missing structure the narrative relies on → *add the node / mechanism and re-derive* (the "back-and-forth");
   - narrative double-counting or assuming an unsupported link → *correct the estimate*;
   - genuinely irreducible → *record as modeling uncertainty* for sensitivity analysis.

The reconciler's output is an adjusted estimate **plus the named structural reason**, optionally triggering another iteration. The loop terminates on agreement or on a certified-irreducible disagreement. This is, in effect, a procedure for **constructing the externalized causal model by reconciling it against holistic judgment** — an operational approximation to the "externalize and enforce coherence" ideal of Section 4, without committing to formal-all-the-way-down.

> **Worked example.** Estimator: `0.85` ("fiscal driver → grievances track harm → taxes"). Modeler: `≈0.60`, because its prior on `P(F=1)` is moderate. Reconciler: "Disagreement traces to the modeler's prior on `F`, which the estimator implicitly set near 1; the mechanism itself is sound and represented. Resolve toward `0.8`; flag the prior on `F` as a sensitivity lever." The triangulation did not merely yield a number — it **excavated a hidden assumption** that would otherwise never have surfaced.

### 6.3 Gate the expensive audit to high-influence evidence

Three or more model calls per likelihood, with iteration, is costly. It is also unnecessary for most evidence. Run the cheap narrative estimate everywhere; trigger the triangulation audit only on the **evidence that actually moves the posterior** — the top drivers by `|log(LR)|`, which the pipeline already identifies for sensitivity analysis. Audit those; trust the cheap estimate for the rest. This is frugal *and* principled: scrutiny is concentrated exactly where error would change conclusions.

### 6.4 Dependence handling

The narrative engine's characteristic error is treating correlated evidence as independent and multiplying likelihoods (five reports of one fact counted as five), producing overconfidence. Two complementary defenses:

- **Within the narrative engine:** cluster corroborating evidence and assign the *cluster* a single joint likelihood rather than multiplying members. This preserves texture and catches the dependence the analyst can see.
- **Via the auditor:** the formal graph reveals when two pieces descend from a common cause, catching dependence the analyst missed. This is one of the audit's highest-value functions.

### 6.5 Priors and sensitivity

Priors are a **researcher input**, not a hardcoded uniform default; a uniform prior is a defensible *reference* choice only when justified. Sensitivity analysis ranges over **both** priors and likelihoods (and, where the auditor flags them, over the structural assumptions), and is reported alongside the posterior. Robustness signals (whether a conclusion rests on a few decisive items or an accumulation of weak ones) are mechanical and reported, not asserted.

### 6.6 What each component is *for*

| Component | Role | Why |
|---|---|---|
| Narrative likelihood engine | drives updating for single text | preserves discriminating texture |
| Externalized reasoning | documents the implicit model per estimate | auditability, coherence checking |
| Formal DAG (auditor) | cross-checks high-influence likelihoods | catches incoherence and hidden dependence |
| Reconciler (diagnostic) | resolves disagreement by naming its source | turns disagreement into a learning signal, not mush |
| Top-driver gating | scopes the expensive audit | concentrates scrutiny where it changes conclusions |
| Van Evera labels | presentation layer over likelihoods | human legibility for process-tracing scholars |
| Sensitivity (priors + likelihoods) | reports robustness | honesty about dependence on assumptions |
| Formal model (CausalQueries) | source of truth for multi-case | identified counterfactuals and population effects |

---

## 7. Why an LLM changes what is feasible

Classical process tracing could not externalize reasoning, run independent parallel estimators, and reconcile them *at scale* — the labor was prohibitive, so analysts left the model implicit (narrative) or paid the full formalization cost once (formal). Language models make three things cheap that were not before:

1. **Externalized reasoning per estimate** — the implicit model can be emitted as structured output alongside every likelihood.
2. **Independent parallel estimation** — narrative and formal estimates can be produced by separate agents with genuinely different framings, so agreement is corroboration and disagreement is diagnostic (subject to the correlated-error caveat in Section 8).
3. **Diagnostic reconciliation and iterative model construction** — a third agent can localize disagreements to specific structural assumptions and drive the back-and-forth that refines both estimate and graph.

This is what makes the "externalize and check coherence" ideal of Section 4 operational rather than aspirational, while staying honest about the identifiability limits of Section 5.

---

## 8. Limitations and known boundaries

- **Correlated model errors.** Independent agents drawn from the same model family can share blind spots and agree while both wrong. Agreement is *support*, not proof. Mitigations: distinct prompts/roles, and where possible distinct model families; treat the auditor as a check, not an oracle.
- **The reconciler must diagnose, not average.** Splitting the difference between two models is not Bayesian; it is noise reduction at the cost of meaning. The reconciler is constrained (by schema) to name the structural source of disagreement and a corrective action.
- **Single-case counterfactuals remain prior-dominated.** The methodology refuses to present them as evidence-driven. Identified counterfactuals require the multi-case path.
- **Texture vs. structure at the audit boundary.** Comparing a narrative likelihood to a formal one requires representing the evidence as a node for the comparison — a local, temporary binarization. This is scoped to the audit; it does not bind the primary narrative estimate.
- **Cost.** Triangulation is expensive; gating to top-driver evidence is the control. The trade-off (depth on consequential evidence, cheap judgment elsewhere) should be made explicit and logged, never silently truncated.
- **The optimum is unreachable.** Full Bayesian model averaging over structures is intractable; this methodology is the best *feasible* approximation, not the optimum itself.

---

## 9. Relationship to the reference implementation

The companion pipeline already instantiates parts of this methodology and points at the rest:

- A single-text path (narrative-style likelihood updating, Van Evera diagnostic classification, mechanical robustness, sensitivity over likelihoods) and a separate cross-case path bridging to `CausalQueries` for formal estimation — i.e., the task-routing of Section 6.1 is structurally present.
- Per-hypothesis "top driver" identification already exists, supplying the gate of Section 6.3 at no additional cost.

The gaps this methodology defines as next steps: researcher-settable priors with prior-sensitivity; comparative multi-hypothesis likelihoods replacing the two-way "H vs. ¬H" formulation; externalized reasoning attached to each likelihood; the triangulation auditor with diagnostic reconciliation, gated to top drivers; and explicit evidence-dependence clustering. Each is independently shippable and leaves the pipeline working — the methodology is intended to be adopted in thin slices, not as a flag-day rewrite.

---

## 10. Summary

The choice was never narrative *versus* formal. Both are Bayes' rule; they differ only in where the likelihood comes from, and a "direct" likelihood is simply an undocumented causal model. The theoretical optimum — one coherent posterior averaged over causal structures — is intractable, and a single text cannot identify counterfactual or population quantities regardless of method. The best feasible automated methodology therefore (i) keeps narrative texture as the inferential substrate for single texts, (ii) forces the implicit model into the open via externalized reasoning, (iii) uses an independently-constructed formal graph to *audit* the high-influence likelihoods for coherence and hidden dependence, resolving disagreements by diagnosis rather than averaging, (iv) reserves the formal model's identified counterfactual and population estimands for the multi-case setting where the data support them, and (v) reports sensitivity to priors, likelihoods, and structure throughout. Language models are what make this affordable at scale; identifiability is what bounds its ambitions.

---

## References

- Fairfield, T., & Charman, A. (2017). *Explicit Bayesian Analysis for Process Tracing: Guidelines, Opportunities, and Caveats.* Political Analysis 25(3). https://eprints.lse.ac.uk/69203/2/Fairfield_Explicit%20bayesian%20analysis_author_2017%20LSERO.pdf
- Fairfield, T., & Charman, A. (2022). *Social Inquiry and Bayesian Inference: Rethinking Qualitative Research.* Cambridge University Press.
- Humphreys, M., & Jacobs, A. (2015). *Mixing Methods: A Bayesian Approach.* American Political Science Review 109(4).
- Humphreys, M., & Jacobs, A. (2023). *Integrated Inferences: Causal Models for Qualitative and Mixed-Method Research.* Cambridge University Press. https://macartan.github.io/integrated_inferences/
- *CausalQueries: Make, Update, and Query Binary Causal Models* (R package). https://cran.r-project.org/package=CausalQueries — Guide: https://macartan.github.io/causalmodels/process-tracing.html
- Van Evera, S. (1997). *Guide to Methods for Students of Political Science.* Cornell University Press. (Hoop / smoking-gun / doubly-decisive / straw-in-the-wind diagnostic tests.)
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science.* Cambridge University Press. (Probability as extended logic.)
- *Two fully specified Bayes factors for hypothesis testing and sensitivity analysis in process tracing* (2026). arXiv. https://arxiv.org/html/2606.16683

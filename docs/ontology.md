# Process Tracing Ontology

This document defines the current analytic objects used by the `pt` pipeline and
the report network. It is a reading contract for analysts and agents, not a
historical implementation plan. Superseded phase notes live in
`~/archive/process_tracing/raw/repo-archive-2026-06-24/`.

## Core Claim

The ontology separates four things that are easy to conflate:

- **Substantive process:** actors, events, mechanisms, and causal edges in the
  historical case.
- **Trace/evidence record:** source-grounded evidence items and missing predicted
  traces.
- **Hypothesis space:** rival causal explanations and observable predictions.
- **Inference layer:** likelihood vectors, dependence clusters, priors,
  comparative support, robustness, and synthesis verdicts.

The report is valid only as comparative support over the listed hypothesis set
plus the residual hypothesis. It is not an absolute probability of truth, an
identified causal effect, or a counterfactual estimate.

## Objects

| Object | Code contract | Meaning |
|---|---|---|
| Actor | `Actor` | A person, organization, institution, or group mentioned in the source. Actors explain agency and constraints but do not by themselves count as evidence. |
| Event | `Event` | A dated or approximately dated occurrence in the process sequence. Events can be causes, intermediates, outcomes, or contextual conditions. |
| Mechanism | `Mechanism` | A proposed causal process connecting causes to outcomes. Mechanisms should explain how or why a cause produces an effect. |
| Causal edge | `CausalEdge` | An extracted directional relationship between two extracted objects. It is a substantive-process claim, not an evidence weight. |
| Evidence | `Evidence` | A quoted or closely paraphrased trace from the input text. Evidence can be empirical or interpretive and must preserve source text. |
| Hypothesis | `Hypothesis` | A rival causal explanation with a causal mechanism and observable predictions. Hypotheses should be as mutually exclusive and discriminating as the corpus allows. |
| Prediction | `Prediction` | A trace we would expect to observe if a hypothesis were true. Predictions feed diagnostic testing and absence checks. |
| Evidence likelihood | `EvidenceLikelihood` | One evidence item's relative likelihood vector across all hypotheses. This is the central testing object. |
| Dependence cluster | `EvidenceCluster` | Evidence items that are not conditionally independent because they share source lineage, event origin, mechanism, or sub-narrative. |
| Absence evaluation | `AbsenceEvaluation` | A qualitative finding that a predicted trace is missing from the source. This is source silence, not proof that the event never occurred. |
| Bayesian posterior | `HypothesisPosterior` | Comparative support after deterministic updating, including update trail, top drivers, and robustness. |
| Synthesis verdict | `HypothesisVerdict` | Narrative conclusion for each hypothesis, calibrated against comparative support and caveats. |

## Inference Semantics

**Likelihood vectors.** Pass 3 asks for one vector per evidence item across all
hypotheses. Pairwise likelihood ratios are derived from that vector, so
reciprocity and transitivity hold by construction.

**Relevance gating.** Evidence below the relevance gate is forced uninformative.
Evidence above the gate is discounted on the log scale according to relevance.
Interpretive evidence is capped more tightly than empirical evidence.

**Dependence pooling.** Dependence clusters are partially pooled before Bayesian
updating. A fully redundant cluster acts like one observation; a partially
redundant cluster contributes more than one but less than the raw count.

**Residual hypothesis.** The update includes `H0_residual`, an explicit "none of
the listed explanations" competitor. It prevents the system from being forced to
crown a listed story when the hypothesis menu is incomplete.

**Support.** Reported support is normalized comparative support across the
hypothesis set used in the update. Read it with robustness, sensitivity range,
prior stability, and source-scope caveats.

## Report Network Semantics

The interactive network is an analyst-facing projection of the ontology, not the
complete result. The full evidentiary record remains in `result.json`, the
diagnostic matrix, and the Evidence Inventory.

| Network element | Meaning |
|---|---|
| Event nodes | Extracted dated process events. They are placed in left-to-right temporal order when dates are available. |
| Evidence nodes | Source-grounded traces. They are placed by `approximate_date` when available. |
| Hypothesis nodes | Rival explanations. They sit downstream of the dated trace sequence because they are claims being evaluated, not historical events. |
| Actor and mechanism nodes | Extracted agency/context objects. They are hidden by default to reduce clutter but can be toggled on. |
| Causal edges | Extracted substantive process relationships, shown left-to-right when temporally consistent. |
| Top-driver links | Evidence-hypothesis links for the evidence items with the largest absolute log-likelihood impact on a hypothesis after caps and gating. These are shown by default because they explain the largest updates. |
| Background-driver links | Top-driver links where the evidence is far upstream of the outcome window. They may have evidentiary value, but the report warns that they are background context unless paired with proximate mechanism traces. |
| Additional evidence links | Other displayed evidence-hypothesis links that clear the visual LR threshold but are not top drivers. They are hidden by default and available by toggle. |
| Temporal conflicts | Extracted causal edges that point backward in the dated layout. They are hidden by default and shown by toggle because they require analyst review. |
| Isolated nodes | Nodes with no currently visible edge. They are not discarded; they are hidden only by the initial view. The Evidence Inventory preserves the full list. |

## Concrete Example

Suppose the source says a military-backed coup annulled elections in 1797, and
the hypotheses are:

- `h1`: institutional weakness caused collapse.
- `h2`: ideological polarization caused collapse.
- `h3`: military coercion caused collapse.

The evidence item is represented once as `Evidence`. Pass 3 gives it a likelihood
vector, for example `{h1: 4, h2: 2, h3: 9}`. The Bayesian layer derives each
hypothesis's LR from that shared vector, applies caps/relevance, pools it with
dependent evidence if needed, and updates comparative support. If the item has
one of the largest absolute log-LR impacts for `h3`, the report draws a top-driver
link from the evidence node to `h3`. If the evidence is decades before the focal
outcome, it becomes a background-driver link: still evidentiary, but not enough
by itself to establish the proximate mechanism.

## Current Limitations

- The network is a projection, not the inference engine. The Bayesian update is
  in `pt/bayesian.py`; the graph makes the update inspectable.
- Dependence clustering uses a scalar dependence strength per cluster. Per-
  hypothesis redundancy and full trace-production model averaging remain future
  methodology work.
- Absence findings are qualitative. Observability-weighted missingness does not
  yet enter the Bayesian update.
- A limited source corpus can produce an optimal exploratory report but still be
  capped below PhD-review-ready causal identification.

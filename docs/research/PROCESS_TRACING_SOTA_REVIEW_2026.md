# Process Tracing SOTA Review - 2026-06-22

This review defines the external state of the art the project must exceed. It is
not a literature review for publication; it is an engineering input for the
SOTA+ recovery plan.

## Bottom Line

Current SOTA is fragmented:

- **Process-tracing methodology** has strong theory for rival hypotheses,
  diagnostic evidence, Bayesian updating, and analytic transparency.
- **Formal mixed-methods causal inference** has tools for causal models,
  Bayesian updating over causal types, and cross-case queries.
- **LLM/text-analysis research** has scalable extraction and coding machinery,
  but it is not yet reliable enough as a standalone causal-inference engine.
- **Process-tracing software** exists or is emerging for formal Bayesian
  bookkeeping, but it still expects humans to specify evidence, priors,
  likelihoods, hypotheses, or causal models.

The SOTA+ opportunity is therefore not to imitate one existing tool. It is to
integrate the best pieces into an agentic pipeline that removes the usual
constraints: qualitative labor bottlenecks, low technical capacity in mixed
methods, and manual translation between source evidence and quantitative causal
models.

## SOTA Dimensions

### 1. Bayesian Process Tracing

Fairfield and Charman (2017) remain central for explicit Bayesian process
tracing: they frame likelihood assessment as the key inferential step, emphasize
well-specified rival hypotheses over an underspecified negation, and treat
Bayesian probability as a bridge between qualitative and quantitative
methodology.

SOTA strengths:

- explicit priors and likelihood ratios;
- direct comparison against concrete rivals;
- transparency about where scholars disagree;
- iterative dialogue with evidence.

SOTA limits:

- probability assignment for qualitative evidence is ambiguous;
- applying the method well requires substantial training and effort;
- manual analysts struggle to scale beyond a few pieces of evidence;
- dependence, source-production, and hypothesis partition validity are hard to
  enforce consistently.

Key sources:

- Tasha Fairfield and Andrew Charman, "Explicit Bayesian Analysis for Process
  Tracing" (Political Analysis, 2017): https://cpd.berkeley.edu/wp-content/uploads/2018/02/CPC_Fairfield_article.pdf
- Tasha Fairfield and Andrew Charman, "Understanding Bayesianism: Fundamentals
  for Process Tracers" (Political Analysis, 2021/2022): https://www.cambridge.org/core/journals/political-analysis/article/understanding-bayesianism-fundamentals-for-process-tracers/E543498F74EA03536A0693CE5BA5490C

### 2. Formal Mixed-Methods Causal Models

Humphreys and Jacobs' Integrated Inferences / CausalQueries line is the strongest
formal route for integrating within-case and cross-case evidence. The approach
forms causal models, updates them with data, and queries the models for causal
answers. The CausalQueries package lets users declare binary-node causal models,
update beliefs about causal types, and calculate arbitrary queries.

SOTA strengths:

- explicit causal models rather than narrative-only inference;
- queryable estimands;
- Bayesian updating over model types;
- a real software package (`CausalQueries`);
- natural bridge from qualitative process evidence to quantitative/cross-case
  designs.

SOTA limits:

- no automated text-to-evidence pipeline;
- analyst must specify model structure, variables, data, and measurements;
- binary-variable formalization can discard textual detail;
- single-case counterfactual/population claims remain prior-dominated unless
  cross-case variation exists.

Key sources:

- Macartan Humphreys and Alan Jacobs, *Integrated Inferences: Causal Models for
  Qualitative and Mixed-Method Research*: https://macartan.github.io/integrated_inferences/HJC1.html
- `CausalQueries` package page: https://integrated-inferences.r-universe.dev/CausalQueries

### 3. Emerging Bayesian Process-Tracing Software

Esarey's "Robust and Replicable Bayesian Process Tracing" is directly relevant
because it recognizes the same adoption bottleneck: Bayesian process tracing has
high training requirements, lacks standardized translation from qualitative
evidence to probability distributions, and struggles with non-exclusive theories.
The proposed software lowers the barrier through reproducible inputs, visual
interpretability, robustness checks, and support for interdependent/non-exclusive
theories.

SOTA strengths:

- reproducible Bayesian process-tracing reports;
- robustness checks for evidence interpretation;
- visual mapping from assumptions to conclusions;
- explicit attention to non-exclusive theories.

SOTA limits:

- still not an automated source-reading analyst;
- no LLM extraction/hypothesis-generation/source-packet workflow;
- does not solve source acquisition, provenance, trace-production modeling, or
  cross-case integration by itself.

Key source:

- Justin Esarey, "Robust and Replicable Bayesian Process Tracing" (preliminary,
  2024): https://www.justinesarey.com/2024-11-12-bayesian-process-tracing-short.pdf

### 4. LLM Causal Reasoning From Text

Recent LLM work shows why architecture is necessary. The ReCAST benchmark paper
argues that existing causal-reasoning tests often rely on synthetic or explicit
texts, while real-world causal claims are long, implicit, and structurally
complex. Their benchmark reports that the best model reached only 0.477 average
F1 on real-world causal-relationship extraction.

SOTA strengths:

- LLMs can process scale and text complexity impossible for manual teams;
- benchmark work is moving from synthetic to real-world causal text;
- LLMs can provide prior knowledge, direct inference, and post-hoc refinement
  for causal discovery workflows.

SOTA limits:

- LLMs alone are not reliable causal graph extractors;
- real-world implicit causal relationships remain difficult;
- extraction accuracy is not the same as process-tracing validity;
- benchmark-driven validation is mandatory.

Key sources:

- Saklad et al., "Can Large Language Models Infer Causal Relationships from
  Real-World Text?" (2025): https://arxiv.org/html/2505.18931v2
- "Large Language Models for Causal Discovery: Current Landscape and Future
  Directions" (IJCAI 2025 survey): https://www.ijcai.org/proceedings/2025/1186.pdf

### 5. LLM Qualitative Analysis and Coding

LLM qualitative-analysis research matters as adjacent evidence about scaling
interpretive labor, but it is not process tracing. The SOTA is early, rapidly
moving, and focused on coding/content analysis rather than causal inference.
Recent work argues that LLMs can scale coding, but also warns that LLM coding can
introduce systematic bias and misleading downstream inferences.

SOTA strengths:

- scalable text classification and coding;
- prompt/codebook design lessons;
- hybrid workflows for validation and iteration;
- concrete warnings about bias measurement.

SOTA limits:

- coding is not causal inference;
- human-gold-standard framing can become a ceiling rather than a benchmark;
- measurement error and bias can propagate into quantitative analysis;
- reflexive coding workflows do not solve hypothesis partition, diagnostic
  evidence, likelihoods, dependence, or trace-production.

Key sources:

- "Large Language Model for Qualitative Research: A Systematic Mapping Study"
  (2024/2025): https://arxiv.org/html/2411.14473v4
- Dunivin, "Scaling hermeneutics" (EPJ Data Science, 2025): https://link.springer.com/article/10.1140/epjds/s13688-025-00548-8
- Ashwin, Chhabra, and Rao, "Using Large Language Models for Qualitative
  Analysis can Introduce Serious Bias" (Sociological Methods & Research, 2025):
  https://journals.sagepub.com/doi/10.1177/00491241251338246

## SOTA+ Implications For This Repo

The project should not accept the field's existing bottlenecks as constraints.
Manual qualitative labor and weak technical mixed-methods implementation are
limitations of current practice, not limits of the method.

SOTA+ means:

1. **Agentic source work:** automatically build source packets with primary
   documents, rival secondary accounts, provenance, dates, and source-genre
   metadata.
2. **Hypothesis-partition governance:** freeze the research question, focal
   window, rival hypotheses, residual, and pairwise discriminators before
   testing.
3. **Structured extraction with provenance:** every event, evidence item,
   actor, mechanism, and causal edge must trace to source text and source
   metadata.
4. **Coherent Bayesian testing:** keep vector likelihoods, priors, residual
   hypotheses, sensitivity, and dependence pooling.
5. **Trace-production modeling:** explicitly model solicitation, recording,
   survival, omission, translation, false-positive claims, and source
   incentives.
6. **Adversarial causal audit:** use causal graphs as qualitative critics of
   likelihood claims, not as unparameterized likelihood calculators.
7. **Cross-case bridge:** use process-tracing outputs to build and stress-test
   CausalQueries/QCA/statistical models when cases and variables support it.
8. **Benchmark gates:** validate every capability against frozen cases and
   known failure modes before claiming PhD/think-tank quality.

## What Went Wrong In The Current Repo

- The SOTA+ vision was documented as aspiration, but not enforced as a landing
  gate for each slice.
- Presentation improvements were allowed to stand in for inference-quality
  improvements.
- The project lacked a frozen benchmark and therefore could not discipline
  claims like "A-level" or "optimal."
- Future optimum, current implementation, and historical sprint notes drifted
  together until documentation cleanup separated them.
- Thin slices were not consistently tied to one named SOTA frontier and one
  measurable failure mode.

## Research Gaps To Close Next

1. Build a source-packet benchmark from public historical cases.
2. Define PhD-grade hypothesis-partition checks before testing.
3. Add a source-lineage/dependence benchmark with planted duplicate and shared
   source examples.
4. Add a real-world causal-text extraction benchmark slice, using ReCAST-style
   failure modes as inspiration.
5. Define cross-case eligibility gates before invoking CausalQueries.
6. Compare the pipeline to manual Bayesian process-tracing software and to
   formal CausalQueries workflows on the same case material.

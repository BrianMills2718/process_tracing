# Process Tracing Methodology

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Goals And Constraints

Goal: make process tracing agent-drivable and auditable without turning it into
generic summarization or false-precise causal scoring.

The system should help an analyst:

- define a causal question and rival hypotheses
- extract source-grounded observations
- test evidence against all hypotheses in one coherent likelihood-vector frame
- represent absence of expected evidence separately from positive evidence
- update comparative support through deterministic math
- inspect caveats, source-scope gaps, dependence risks, and report quality

Constraints:

- Single-text output is comparative support over the listed hypotheses, not an
  identified causal effect or probability of truth.
- The LLM handles semantic judgments; deterministic code handles math,
  calibration, report generation, and checks.
- The project must stay generalist. Dataset-specific historical rules are a
  bug.
- Validation claims must stay below the current evidence. Architecture and
  auditability are demonstrated; PhD-level methodological validity is not yet.

## Borrow-Vs-Build

| Area | Borrowed | Built Locally | Rationale |
|---|---|---|---|
| Process-tracing theory | Van Evera-style diagnostic tests; Fairfield and Charman-style Bayesian process tracing | Pipeline stages that make hypotheses, diagnostic evidence, absence checks, and support updates explicit | Existing theory gives the method; local code makes it executable and auditable. |
| Mixed-methods causal inference | Humphreys/Jacobs and `CausalQueries` as the cross-case/formal-model path | Bridge from source-text process evidence into structured causal objects | Existing tools do not automate source reading or source-grounded evidence construction. |
| LLM engineering | Shared `llm_client`, structured output, prompt templates, cost/trace discipline | Process-tracing-specific schemas, prompts, and quality gates | Shared infra supplies governance; local code owns the analytic domain. |
| Report validation | Academic process-tracing critique and evidence caps | `make audit-result`, HTML report audit sections, output-quality rubric | No off-the-shelf tool grades this specific artifact shape. |
| Agentic research labor | Codex/Claude Code style workspace agents through `llm_client` | Typed source-packet and future critique tasks | Direct agent subprocesses would bypass observability and budget control. |

## ADR Map

| ADR | Decision | Why It Matters |
|---|---|---|
| `docs/adr/0001_process_tracing_methodology_spine.md` | Present and govern this project as an auditable process-tracing methodology system, not generic qualitative coding. | Keeps portfolio narrative, implementation priorities, and validation claims aligned. |

## Modality Assessment

| Surface | Mode | Treatment |
|---|---|---|
| Pipeline schemas and boundary contracts | Deductive / plan-first | Pydantic models define stage outputs; invalid structured output fails loudly. |
| Bayesian support update and sensitivity | Deductive / plan-first | Deterministic, unit-tested Python; no LLM judgment in math. |
| Truth-in-labeling and report provenance | Deductive / plan-first | Reports must say comparative support, show source hashes, expose source packets and audit caveats. |
| Source scope and source acquisition | Exploratory / ladder | Use source packets and coverage readouts to reveal source gaps before pretending corpus adequacy. |
| Hypothesis partition quality | Hybrid | Structural blockers are specifiable; thresholds and case-specific repair rules require benchmark readouts. |
| Dependence and trace-production modeling | Hybrid | Basic dependence clusters are implemented; per-hypothesis redundancy and source-production structures remain exploratory. |
| Methodological validation | Exploratory / ladder | Needs frozen cases, adversarial review, ablations, and human/benchmark comparison before promotion to gates. |

## Failure Modes

| Failure | Detection | Response |
|---|---|---|
| False probability language | Report or docs call support "truth probability" | Reword to comparative support over the listed hypothesis set. |
| Broad absorptive winner | One hypothesis explains everything and rivals are not discriminated | Split/merge hypotheses and add pairwise discriminators. |
| Source-scope overclaim | Report treats overview text as publication-strength evidence | Add or repair source packet; cap claims until source coverage is adequate. |
| LLM semantic shortcut | Code uses keywords/rules for evidence classification | Replace with structured LLM boundary or deterministic non-semantic check. |
| Dependent evidence overconfidence | Top support comes from correlated traces | Cluster, pool, or downgrade; expose fragility and collect independent evidence. |
| Polished report hides weak evidence | HTML looks good while audit caps remain | Treat audit caps as binding; collect/design evidence rather than polish prose. |

## Promotion Rule

Exploratory instruments become governed contracts only after they produce stable
readouts across more than one case. Promotion requires:

1. a typed artifact or schema,
2. a deterministic or live-gated check,
3. a report/audit surface that makes failures visible,
4. concern-register triage, and
5. removal or quarantine of temporary scaffolding.

Until then, the project should say "instrumented" or "partial," not
"validated."

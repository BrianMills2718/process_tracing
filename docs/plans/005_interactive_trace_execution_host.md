---
status: complete
owner: process-tracing
updated: 2026-06-25
---

# Plan 005 - Interactive Trace Execution Host

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None for the current host slice; later visual-audit slices remain open
**Blocks:** stage-by-stage process-tracing review, richer audit UI, source-acquisition promotion

---

## Gap

**Current:** The local workbench can load a final report and build/enrich
source-acquisition targets. It does not let a reviewer run the pipeline one
stage at a time, inspect each stage's typed output, compare intermediate
artifacts, or learn how the process-tracing method works while operating it.

**Target:** A local interactive host where a human or agent can start a real
process-tracing run, trigger each pipeline stage, inspect stage outputs with
concise explanations and tooltips, and view visual summaries wherever the data
supports them.

**Why:** The user needs to understand and audit the inferential workflow, not
only the final report. A PhD-quality process-tracing system should expose how
evidence, hypotheses, diagnostic tests, absence findings, Bayesian support, and
source gaps transform across stages.

---

## Frame

Goal: make the process-tracing pipeline inspectable as a sequence of typed,
auditable transformations without weakening the existing CLI/report path.

Constraints:

- The host must use real pipeline code and real artifacts; no mocked E2E
  success path.
- Every UI action must have a JSON API endpoint.
- Stage outputs are views over typed artifacts, not free-form UI state.
- Explanations and tooltips must be concise and interpretive, not marketing
  copy.
- Existing final-report and source-acquisition workbench capabilities must
  remain available until the new host replaces them.

Out of scope for Slice 005a:

- Multi-user auth, remote deployment, or production hosting.
- Editing and approving new source material into the corpus.
- Replacing the static report renderer.
- Solving benchmark calibration or PhD-quality grading thresholds.

Borrow-vs-build:

| Capability | Decision | Rationale |
|---|---|---|
| Local host | Build locally on current stdlib server first | The current repo already has a small agent-drivable workbench. The first slice should preserve that simplicity. |
| Stage execution | Build locally around existing pass functions | Stage boundaries are repo-specific and already typed in `pt/schemas.py`. |
| Visualizations | Build small HTML/CSS/JS views first | The first readout is whether the visual grammar helps review; avoid a frontend stack migration before the contract is proven. |
| Future richer UI | Defer Vite/React decision | If stage replay, streaming, or complex graph interaction becomes heavy, refresh this plan and choose the frontend stack explicitly. |

Clean-docs note: The current `pt.workbench` is not archived now. It remains a
tested narrow source-acquisition workbench until this host preserves or replaces
its capabilities. Archive it only after a replacement passes live E2E and an
archive rationale is written.

Implementation note: the current slice now ships `pt.trace_host` plus host API
endpoints for run creation, stage replay, and run inspection, with live non-mocked
verification recorded for run `output/workbench_runs/run_20260624_200616_d4f6`.

---

## Modality Split

| Surface | Mode | Reason |
|---|---|---|
| Stage order and typed artifacts | Deductive | Existing pipeline stages and schemas define the sequence. |
| JSON API contracts | Deductive | Request/response shapes can be specified before implementation. |
| Persistence of stage outputs | Deductive | The host can write/read a run directory with stage JSON artifacts. |
| Visual usefulness of each panel | Exploratory | We need reviewer readouts to learn which views actually improve understanding. |
| Tooltip wording and explanation density | Exploratory | Human review should tune whether text is helpful or distracting. |

Exploratory readout for Slice 005a: Brian can open the mockup/live host and
explain what each stage does, what it consumed, what it produced, and what the
main audit risk is without reading source code.

---

## Design Artifacts

Static mockup: `docs/plans/005_interactive_trace_execution_host_mockup.html`

Planning notebook / contract examples:
`docs/plans/005_interactive_trace_execution_host_contracts.ipynb`

Implementation may not start until the mockup is reviewed. If the notebook is
changed before implementation, this plan must be updated with the new contract
examples.

---

## Boundary Diagram

```mermaid
flowchart LR
  subgraph External["External dependencies"]
    Browser["Human browser"]
    Agent["Codex / Claude Code"]
    LLM["llm_client providers"]
    Web["open_web_retrieval"]
  end

  subgraph Host["process_tracing interactive host"]
    API["Host API\nJSON endpoints"]
    Runner["Stage runner\nexisting pass functions"]
    Store["Run store\nstage artifacts"]
    Viz["Visualization layer\nHTML views"]
    Explain["Explanation registry\ntooltips + stage guides"]
  end

  subgraph Core["Existing process_tracing core"]
    Pipeline["pt.pipeline / pass_*"]
    Schemas["pt.schemas + source_packet"]
    Report["pt.report"]
    Acquisition["pt.source_acquisition"]
  end

  Browser -->|"click stage action"| API
  Agent -->|"POST /api/stage/run"| API
  API -->|"validated request"| Runner
  Runner -->|"structured prompts"| LLM
  Runner -->|"typed stage result"| Store
  Store -->|"stage artifact JSON"| Viz
  Explain -->|"tooltip copy"| Viz
  Viz -->|"HTML + JSON state"| Browser
  Runner -->|"imports schemas"| Schemas
  Runner -->|"calls pass functions"| Pipeline
  Store -->|"result.json"| Report
  Store -->|"result + source packet"| Acquisition
  Acquisition -->|"optional retrieval"| Web
```

---

## Domain Model Diagram

```mermaid
classDiagram
  class TraceRun {
    run_id
    case_name
    input_path
    output_dir
    current_stage
    status
  }
  class StageExecution {
    stage_id
    status
    started_at
    completed_at
    input_refs
    output_refs
    error
  }
  class StageArtifact {
    artifact_id
    stage_id
    schema_name
    path
    summary
    provenance
  }
  class StageGuide {
    stage_id
    purpose
    consumes
    produces
    audit_questions
  }
  class VisualizationSpec {
    stage_id
    view_type
    data_ref
    empty_state
  }
  class Tooltip {
    term
    explanation
    stage_id
  }
  class ProcessTracingResult {
    extraction
    hypothesis_space
    testing
    absence
    bayesian
    synthesis
    source_coverage
  }

  TraceRun "1" --> "1..*" StageExecution
  StageExecution "1" --> "0..*" StageArtifact
  StageExecution "1" --> "1" StageGuide
  StageExecution "1" --> "0..*" VisualizationSpec
  StageGuide "1" --> "0..*" Tooltip
  TraceRun "0..1" --> "1" ProcessTracingResult
```

---

## Data-Flow And Contract Diagram

```mermaid
sequenceDiagram
  autonumber
  participant U as User / Agent
  participant API as Host API
  participant Store as Run Store
  participant Runner as Stage Runner
  participant LLM as llm_client
  participant View as Stage View

  U->>API: POST /api/runs {input_path, source_packet_path, options}
  API->>Store: create TraceRun
  Store-->>API: RunState
  API-->>U: JSON {ok, run}

  U->>API: POST /api/runs/{run_id}/stages/extract/run
  API->>Store: load required inputs
  alt missing prior input or invalid path
    Store-->>API: validation error
    API-->>U: 400 JSON {ok:false,error_type}
  else inputs valid
    API->>Runner: run_extract(...)
    Runner->>LLM: structured call with ExtractionResult schema
    alt malformed LLM output
      LLM-->>Runner: validation failure
      Runner-->>API: error with trace_id
      API-->>U: 500 JSON {ok:false,error_type,trace_id}
    else valid output
      Runner-->>Store: StageArtifact ExtractionResult
      Store-->>API: StageExecution complete
      API-->>U: JSON {ok, run, stage, artifacts}
    end
  end

  U->>API: GET /api/runs/{run_id}
  API->>Store: load RunState + artifacts
  Store-->>API: typed summaries
  API->>View: render timeline, visualizations, tooltips
  View-->>U: stage dashboard
```

---

## API Contract Sketch

`POST /api/runs`

```json
{
  "input_path": "input_text/source_packets/18_brumaire_source_packet.txt",
  "source_packet_path": "docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json",
  "research_question": "Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup?",
  "model": null,
  "refine": true,
  "max_budget": 1.0
}
```

returns:

```json
{
  "ok": true,
  "run": {
    "run_id": "run_20260624_001",
    "status": "ready",
    "current_stage": "extract",
    "output_dir": "output/workbench_runs/run_20260624_001"
  }
}
```

`POST /api/runs/{run_id}/stages/{stage_id}/run`

```json
{
  "force": false
}
```

returns:

```json
{
  "ok": true,
  "stage": {
    "stage_id": "extract",
    "status": "complete",
    "output_refs": ["extraction.json"]
  },
  "artifacts": [
    {
      "schema_name": "ExtractionResult",
      "path": "output/workbench_runs/run_20260624_001/extraction.json",
      "summary": "12 events, 50 evidence items, 4 mechanisms"
    }
  ]
}
```

Failure responses must use the same JSON shape across endpoints:

```json
{
  "ok": false,
  "error_type": "ValidationError",
  "error": "stage 'test' requires completed hypothesis stage",
  "trace_id": "optional llm_client trace id"
}
```

---

## Stage Views

| Stage | Consumes | Produces | Visual view | Tooltip focus |
|---|---|---|---|---|
| Setup | source text, optional packet, theories, priors | `TraceRun` | input/source packet checklist | source packet is scope control, not evidence |
| Extract | text + packet context | `ExtractionResult` | temporal event lane, evidence/source table, causal edge list | evidence vs event vs mechanism |
| Hypothesize | extraction + theories | `HypothesisSpace` | hypothesis cards with mechanism chains and predictions | rival hypotheses and distinguishability |
| Test | extraction + hypotheses | `TestingResult` | evidence-by-hypothesis diagnostic matrix | likelihood vector, hoop/smoking gun labels |
| Absence | extraction + hypotheses + testing | `AbsenceResult` | missing-evidence warnings by hypothesis | when absence is informative |
| Update | testing + priors | `BayesianResult` | support bars, top-driver deltas, sensitivity band | support is comparative, not truth probability |
| Synthesize | all prior artifacts | `SynthesisResult` | claim/caveat split and source-scope caps | what can and cannot be concluded |
| Refine | full first pass | `RefinementResult` | before/after delta board | second reading changes inference only if rerun |
| Acquisition | result + source packet | `AcquisitionPlan` | ranked evidence needs and web-query agenda | retrieved hit is not evidence yet |

---

## Acceptance Criteria

- Static mockup is reviewed before implementation.
- Host exposes JSON endpoints for run creation, stage execution, run state,
  artifact reads, and report/source-acquisition views.
- Deterministic tests cover stage-order validation, error JSON, artifact
  persistence, and one stage visualization payload.
- Live non-mocked E2E runs Brumaire through at least setup, extract,
  hypothesize, test, update, synthesize, and report generation through the host.
- The UI shows concise explanations and tooltips for stage purpose, consumed
  inputs, produced outputs, and main audit risk.
- Existing source-acquisition workbench behavior remains available or is
  explicitly replaced by equivalent host functionality.
- Adversarial review checks whether the host clarifies method rather than
  merely adding a nicer dashboard.
- Cleanup updates `docs/ARCHITECTURE.md`, Plan 003 progress, wiki, and archive
  rationale if any old surface is retired.

---

## Failure Modes

| Failure | Diagnostic | Response |
|---|---|---|
| UI looks complete but uses stale artifacts | Compare stage timestamps and run ids | Show artifact provenance in every stage panel and reject mixed-run state unless explicitly loaded as comparison |
| User treats retrieved web hit as evidence | Check acquisition panel labels | Label hits as candidates only; require promotion/rerun in a later slice |
| Stage can run out of order | Stage-order test fails | Block with 400 JSON and explain missing prerequisites |
| Visualizations hide uncertainty | Audit reads only charts, not caveats | Pair every chart with source/gap/sensitivity caveat chips |
| Host duplicates pipeline logic | Diff shows reimplemented pass behavior | Host runner must call existing pass functions or `run_pipeline` helpers |
| Current workbench becomes confusing legacy | Both hosts offer overlapping buttons | Mark old route as narrow source-acquisition surface; archive after replacement passes live E2E |

---

## Slice Roadmap

### Slice 005a - Mockup And Contract Approval

**Goal:** approve the interaction model and contracts before implementation.

**Done when:** Brian reviews the mockup, requested changes are dispositioned,
contract notebook examples are current, and this plan is updated.

### Slice 005b - Stage Host Walking Skeleton

**Goal:** create a local host that can create a run, execute setup/extract, and
show typed extraction artifacts with explanations.

**Done when:** deterministic tests pass, one live non-mocked extract stage runs
through the host, and review confirms artifact provenance is visible.

### Slice 005c - Full Stage Execution

**Goal:** run extract through synthesize stage-by-stage and persist every stage
artifact.

**Done when:** live Brumaire host E2E completes through report generation and
`make check` passes.

### Slice 005d - Visual Audit Panels

**Goal:** add matrix, support, sensitivity, source-coverage, and acquisition
views driven from stage artifacts.

**Done when:** adversarial review finds the views improve method
interpretability without creating unsupported causal claims.

### Slice 005d - Visual Audit Panels (Full Spec)

**Goal:** add evidence × hypothesis matrix, support bars with sensitivity bands,
evidence provenance panel, and dependence cluster overlay driven from stage
artifacts. Make the causal structure visually auditable without creating
unsupported causal claims.

**Concern register:** `docs/plans/CONCERNS_005d.md` — triage at every slice
boundary.

**Modality split:**

| Surface | Mode | Reason |
|---|---|---|
| Stage view API contracts | Deductive | Input/output specifiable from artifact schemas |
| Evidence × hypothesis matrix layout | Deductive | Structure follows from TestingResult schema |
| Support bars + sensitivity bands | Deductive | Structure follows from BayesianResult |
| HTML template extraction boundary | Deductive | Component responsibility split is specifiable |
| 59-item matrix legibility | **Exploratory** | Can't predict until rendered with real data |
| Fragile/robust visual treatment | **Exploratory** | Reviewer reaction is emergent |

**ADR-005-HTML — HTML architecture for visual audit panels:**

- **Decision:** Extract `_html()` to `pt/templates/workbench.html` loaded at
  server startup via `Path(__file__).parent / 'templates' / 'workbench.html'`.
  `_html()` becomes a one-liner calling `template.format(...)` with named keys
  (no `{{`/`}}` escaping).
- **Context:** `_html()` was 495 lines before Slice 005d. Adding matrix,
  support bars, provenance, and cluster panels would push it to 1000+ lines
  of inline Python-string HTML with brittle `{{`/`}}` double-escaping.
- **Alternatives considered:** Keep inline (rejected — unmaintainable, no static
  analysis, no browser DevTools editing); Vite/React (deferred per the
  borrow-vs-build table in the Frame section — revisit when the contract is
  proven by a running slice).
- **Consequences:** Template file can be edited with normal HTML tooling.
  Tested in isolation. Loaded once at import time (no per-request I/O).
- **Superseded by:** Vite/React migration, if and when the plan gate is reached.

**Done when:**
- Deterministic tests cover each ViewRenderer projection, view API error paths,
  and template load.
- Live non-mocked Brumaire run produces all panels with correct data.
- Adversarial audit passes the 5-case battery (incomplete run, malformed
  artifact, refine=False, all-below-threshold, single hypothesis).
- CONCERNS_005d.md fully triaged.
- `make check` passes.
- Plan doc and `docs/progress/plan003_sota_plus_progress.md` updated.

---

## Slice 005d Design Artifacts

### Boundary Diagram (updated for Slice 005d)

```mermaid
flowchart LR
  subgraph External["External"]
    Browser["Human browser"]
    Agent["Agent / curl"]
  end

  subgraph Host["process_tracing workbench host"]
    Server["WorkbenchHandler\nHTTP routing"]
    Template["WorkbenchTemplate\npt/templates/workbench.html"]
    HostStore["TraceHostStore\nrun creation + stage execution"]
    ViewRenderer["ViewRenderer\nstage artifact → ViewPayload"]
    RunStore["RunStore\noutput/workbench_runs/{run_id}/"]
    PreRefineStore["PreRefineStore\noutput/workbench_runs/{run_id}/pre_refine/"]
  end

  subgraph Core["process_tracing core"]
    Pipeline["pt.pipeline / pass_*"]
    Schemas["pt.schemas"]
    SchemasView["pt.schemas_view\nViewPayload types"]
    Report["pt.report"]
  end

  Browser -->|"GET /"| Server
  Browser -->|"GET /api/runs/{id}/stages/{id}/view"| Server
  Agent -->|"POST /api/runs, /api/runs/{id}/stages/{id}/run"| Server
  Server -->|"load template"| Template
  Server -->|"create_run / run_stage"| HostStore
  Server -->|"render_*_view(artifact)"| ViewRenderer
  ViewRenderer -->|"read stage JSON"| RunStore
  ViewRenderer -->|"read pre_refine JSON"| PreRefineStore
  ViewRenderer -->|"ViewPayload"| Server
  ViewRenderer -->|"imports"| SchemasView
  HostStore -->|"write stage artifacts"| RunStore
  HostStore -->|"write pre_refine/ before overwrite"| PreRefineStore
  HostStore -->|"calls pass functions"| Pipeline
  HostStore -->|"imports"| Schemas
  RunStore -->|"result.json"| Report
```

### Domain Model Diagram (updated for Slice 005d)

```mermaid
classDiagram
  class TraceRun {
    run_id
    case_name
    input_path
    status : RunStatus
    current_stage : StageId
    result_path
    report_path
  }
  class StageExecution {
    stage_id : StageId
    status : StageStatus
    artifacts : list[StageArtifact]
  }
  class StageArtifact {
    artifact_id
    stage_id
    schema_name
    path
    summary
    provenance : dict
  }
  class MatrixViewPayload {
    stage_id = "test"
    hypotheses : list[str]
    rows : list[MatrixRow]
    cluster_overlays : list[ClusterOverlay]
    below_threshold_count : int
  }
  class MatrixRow {
    evidence_id
    lr_vector : dict[str, float]
    relevance : float
    below_threshold : bool
    diagnostic_type : str
    justification_snippet : str
    cluster_id : str | None
  }
  class ClusterOverlay {
    cluster_id
    evidence_ids : list[str]
    strength : float
    reason_snippet : str
  }
  class SupportViewPayload {
    stage_id = "update"
    bars : list[SupportBar]
    prior_sensitivity_stable : bool
    perturbation_factor : float
  }
  class SupportBar {
    hypothesis_id
    label : str
    posterior : float
    posterior_low : float
    posterior_high : float
    robustness : str
    rank_stable : bool
  }
  class ProvenanceViewPayload {
    stage_id = "synthesize"
    rows : list[ProvenanceRow]
    items_with_marker : int
    items_total : int
  }
  class ProvenanceRow {
    evidence_id
    source_quote_snippet : str
    source_marker : str | None
    favored_hypothesis_id : str
    peak_lr : float
  }
  class DeltaViewPayload {
    stage_id = "refine"
    new_evidence_count : int
    reinterpreted_count : int
    spurious_count : int
    hypothesis_refined_count : int
    posterior_shifts : list[PosteriorShift]
  }
  class PosteriorShift {
    hypothesis_id
    before : float
    after : float
    delta : float
  }

  TraceRun "1" --> "1..*" StageExecution
  StageExecution "1" --> "0..*" StageArtifact
  MatrixViewPayload "1" --> "1..*" MatrixRow
  MatrixViewPayload "1" --> "0..*" ClusterOverlay
  SupportViewPayload "1" --> "1..*" SupportBar
  ProvenanceViewPayload "1" --> "0..*" ProvenanceRow
  DeltaViewPayload "1" --> "0..*" PosteriorShift
```

### Data Flow Diagram (updated for Slice 005d — view rendering path)

```mermaid
sequenceDiagram
  autonumber
  participant B as Browser
  participant H as WorkbenchHandler
  participant V as ViewRenderer
  participant S as RunStore

  B->>H: GET /api/runs/{run_id}/stages/{stage_id}/view
  H->>S: get_run(run_id)
  alt run not found
    S-->>H: TraceHostError
    H-->>B: 404 {ok:false, error_type:"TraceHostError"}
  else run found
    S-->>H: TraceRun
    H->>H: check stage status == "complete"
    alt stage not complete
      H-->>B: 400 {ok:false, error_type:"StageNotComplete"}
    else stage complete
      H->>V: render_view(stage_id, run_dir)
      V->>S: read stage artifact JSON (e.g. testing.json)
      S-->>V: raw artifact bytes
      alt artifact read/parse error
        V-->>H: ProjectionError with trace_id
        H-->>B: 500 {ok:false, error_type:"ProjectionError", trace_id}
      else artifact valid
        V->>V: project artifact → ViewPayload
        V-->>H: ViewPayload (Pydantic model)
        H-->>B: 200 {ok:true, view: ViewPayload.model_dump()}
      end
    end
  end
```

### Backward Runtime Pass — per panel

The backward pass works from the final rendered payload backward to what must
exist at runtime. Each panel's chain:

**Matrix panel (stage: test)**

1. **Final payload:** `MatrixViewPayload` — `hypotheses: list[str]`, `rows:
   list[MatrixRow]` sorted by `max(abs(log(lr)))` descending, `cluster_overlays`
   grouping correlated rows, `below_threshold_count`.
2. **Selector:** stage_id == "test" and stage status == "complete".
3. **Evidence needed at request time:** `testing.json` → `evidence_likelihoods`
   (for `lr_vector`, `relevance`, `diagnostic_type`) + `dependence_clusters`
   (for `ClusterOverlay`). Also `extraction.json` → `evidence[*].source_text`
   (for `justification_snippet`).
4. **Offline artifacts:** `testing.json` and `extraction.json` written by
   `TraceHostStore.run_stage("test")` and `run_stage("extract")`.
5. **Contract reconciliation:** `ViewRenderer.render_matrix_view(testing,
   extraction) -> MatrixViewPayload`. Input: `TestingResult + ExtractionResult`.
   Output: `MatrixViewPayload`. Failure: raises `ProjectionError` if evidence
   IDs in `evidence_likelihoods` don't match `extraction.evidence`.

**Support panel (stage: update)**

1. **Final payload:** `SupportViewPayload` — `bars: list[SupportBar]` ordered by
   `posterior` descending, `prior_sensitivity_stable`, `perturbation_factor`.
2. **Selector:** stage_id == "update" and stage status == "complete".
3. **Evidence needed:** `bayesian.json` → `posteriors` (posterior, robustness,
   top_drivers), `sensitivity` (posterior_low, posterior_high, rank_stable),
   `prior_sensitivity`. `hypothesis_space.json` → `hypotheses[*].label` for
   human-readable bar labels.
4. **Offline artifacts:** `bayesian.json` and `hypothesis_space.json`.
5. **Contract:** `ViewRenderer.render_support_view(bayesian, hypothesis_space)
   -> SupportViewPayload`.

**Provenance panel (stage: synthesize)**

1. **Final payload:** `ProvenanceViewPayload` — `rows: list[ProvenanceRow]`
   sorted by `abs(peak_lr)` descending, `items_with_marker`, `items_total`.
2. **Selector:** stage_id == "synthesize" and stage status == "complete".
3. **Evidence needed:** `testing.json` → `evidence_likelihoods` (for LRs).
   `extraction.json` → `evidence[*].source_text` (for quote snippet) and
   `evidence[*].source_ids` (for source marker).
4. **Offline artifacts:** `testing.json` and `extraction.json`.
5. **Contract:** `ViewRenderer.render_provenance_view(testing, extraction)
   -> ProvenanceViewPayload`. `peak_lr` = max `relative_likelihood` across all
   hypotheses for that evidence item (signed by direction of favored hypothesis).

**Delta panel (stage: refine)**

1. **Final payload:** `DeltaViewPayload` — counts of delta categories +
   `posterior_shifts` comparing `pre_refine/bayesian.json` to `bayesian.json`.
2. **Selector:** stage_id == "refine" and stage status == "complete" and
   `pre_refine/bayesian.json` exists.
3. **Evidence needed:** `pre_refine/bayesian.json` (before), `bayesian.json`
   (after), `refinement.json` (counts).
4. **Offline artifacts:** `pre_refine/` directory written by Task 2's fix.
5. **Contract:** `ViewRenderer.render_delta_view(refinement, pre_bayesian,
   post_bayesian) -> DeltaViewPayload`.

---

### Slice 005e - Retire Or Fold Old Workbench

**Goal:** decide whether `pt.workbench` remains as a source-acquisition alias or
is archived/superseded.

**Done when:** equivalent acquisition behavior exists in the new host, archive
rationale is written if code/docs move, and old search surfaces no longer
compete with active docs.

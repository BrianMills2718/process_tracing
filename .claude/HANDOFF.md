# Session Handoff — Process Tracing Pipeline
**Date**: 2026-06-25 | **Status**: Active | **Branch**: master | **HEAD**: 5f6853f (pushed)

---

## What this project is

An LLM-first automated process tracing pipeline implementing Stephen Van Evera's causal inference methodology. Given a historical text, it runs: Extract → Hypothesize → Partition Audit → Test (likelihood matrix) → Diagnostic Matrix → Absence Evaluation → Bayesian Update → Synthesize → (optionally Refine). Output is `result.json` + `report.html`. Goal: PhD-quality causal inference on historical texts without manual coding.

**Architecture rule:** LLM for all semantic judgment; pure math for Bayesian updates; deterministic code for everything else. No rule-based logic for semantic tasks. All LLM calls use `llm_client` routing via `pt/llm.py`.

---

## What was done this session

### Audit of Slices 3–6 (completed, all clear)

Previous session delivered Slices 3–6. This session ran an independent 5-dimension audit:

| Dimension | Finding |
|-----------|---------|
| Schema coherence | Sound. Three minor tensions documented below. |
| Prompt quality | All field names match schema. ba9f07b fix verified. |
| Test coverage | 135 tests pass. Two coverage gaps documented below. |
| Audit cap exercisability | Caps 80/82/88 exercised. Cap-84 lineage inflation requires multi-source E2E. |
| Report rendering | All new sections render correctly (diagnostic matrix, genre badges, "Acquire from" column). |

### U1 resolved (expected_source_genre prompt fix)
Pre-fix: 5/11 absence findings returned `expected_source_genre="overview"` (wrong — the current text's genre, not the acquisition target). Post-fix run (`output/slice6_postfix`): 0/15 findings returned "overview". All now return proper acquisition genres (`primary_document`, `parliamentary_record`, `news_dispatch`, `secondary_analysis`, `legal_constitutional`). Fix confirmed effective.

### U2 accepted (lineage_type null)
All dependence clusters return `lineage_type=null` on gpt-5-mini. Accepted as model-tier limitation. Gemini-2.5-flash expected to populate. No code change needed.

### 18 commits pushed to origin/master
All prior session commits + session handoff commit now on remote.

---

## Audit findings to carry forward

### Open coverage gaps (log before or during Slice 7)

**Gap A:** `test_extraction_quality.py::TestAbsenceAcquisitionFields::test_prompt_contract_includes_expected_source_genre` only checks the field name appears in the prompt — does not assert the acquisition-target disambiguation ("NOT the current text's genre") is present. One extra assertion would protect the invariant against future prompt edits.

**Gap B:** Cap-84 source-lineage inflation (`audit_result_quality.py` lines 421–452) is structurally unreachable with the French Revolution Wikipedia corpus because all evidence is labeled `source_group="Main text"` and the check explicitly excludes "Main text". The cap requires a multi-source source-packet run with distinct named groups. Log in concern register, defer to Slice 10 (frozen benchmark suite).

### Minor schema tensions (no action needed, just awareness)

- `RivalPairDiagnostic.discriminator_count` and `grade_capped` are derived fields computed correctly in `pass_diagnostic.py` but have no `model_validator` enforcing consistency with `len(discriminators)`.
- `AbsenceEvaluation.expected_source_genre` is `Optional` (schema allows None) but the prompt behavioral contract says populate always. No enforcement gap exposed at current model tiers.
- `EvidenceCluster.lineage_type` description says "Set this so the audit can verify..." but no audit logic checks it.

---

## Next: Slice 7 — Structural Critic Ablation

**Goal:** Use causal graphs as qualitative critics of likelihood claims, not as unparameterized likelihood calculators. Add a critic pass that flags confounds, missing pathways, void links, too-strong claims, and confirmed links — and triggers re-elicitation of Pass 3 when needed.

---

### Hard constraints (DO NOT VIOLATE)

1. **No direct LR mutation.** The critic produces findings only. Any numeric change must happen by re-running Pass 3 with the critic summary injected as context. The critic never writes to `TestingResult` or `BayesianResult` directly.
2. **Off by default.** `--critic` CLI flag, default `False`. Existing pipeline behavior is unchanged without the flag.
3. **Ablation artifacts.** When critic is active: write `result_base.json` (before re-elicitation) and `result_critic.json` (after). Compute `critic_delta.json` = posterior diff between base and critic runs.
4. **Critic is advisory.** `CriticResult` fields findings + re_elicitation_needed. The pipeline acts on `re_elicitation_needed`, but the critic findings themselves are informational.

---

### Schema changes (pt/schemas.py)

```python
CriticFindingType = Literal[
    "confound",          # a third variable explains both cause and effect
    "missing_pathway",   # a causal mechanism is missing from the extraction
    "void_link",         # a causal edge in the graph lacks evidentiary support
    "too_strong_claim",  # a likelihood ratio is not justified by the evidence
    "confirmed_link",    # a causal link is well-supported and should be noted
]

class CriticFinding(BaseModel):
    finding_type: CriticFindingType
    target: str = Field(
        description="ID of the evidence item, hypothesis, or causal edge (format: 'src_id->tgt_id') "
        "this finding refers to."
    )
    severity: Literal["high", "medium", "low"]
    reasoning: str = Field(description="Why this is a problem and what evidence supports the concern.")
    recommendation: str = Field(description="Concrete action: re-elicit, cluster, merge, or discard.")

class CriticResult(BaseModel):
    findings: list[CriticFinding] = []
    summary: str = Field(description="2-3 sentence overall structural assessment.")
    re_elicitation_needed: bool = Field(
        description="True if high-severity findings suggest the likelihood matrix should be re-run "
        "with the critic's structural context injected."
    )

# Add to ProcessTracingResult:
critic: Optional[CriticResult] = None
```

---

### New files

**`pt/pass_critic.py`** — single LLM call. Receives:
- The research question
- All hypotheses + causal mechanisms
- All causal edges from extraction
- The likelihood matrix (evidence → vector of LRs)
- The diagnostic matrix (which pairs lack discriminators)

Returns `CriticResult`. The prompt should instruct the model to look for:
- Confounds: variables correlated with both cause and effect that aren't controlled for
- Missing pathways: mechanisms mentioned in text but absent from the causal graph
- Void links: causal edges with no supporting evidence in the likelihood matrix
- Too-strong claims: evidence items assigned smoking-gun/doubly-decisive labels without justification
- Confirmed links: well-evidenced pathways worth explicitly affirming

**`pt/prompts/pass_critic.yaml`** — Jinja2 prompt. Fields: `{{ research_question }}`, `{{ hypotheses_json }}`, `{{ causal_edges_json }}`, `{{ likelihood_matrix_json }}`, `{{ diagnostic_matrix_json }}`.

---

### Pipeline changes (pt/pipeline.py)

Insertion point: after `pass_diagnostic` (Pass 3.6), before Bayesian update.

```
if args.critic:
    # 1. Save result_base.json with testing result before critic intervention
    # 2. Run pass_critic → CriticResult
    # 3. If critic.re_elicitation_needed:
    #    - Re-run pass_test with critic_context injected into prompt
    #    - Re-run pass_diagnostic on new testing result
    # 4. Save result_critic.json after critic re-elicitation
    # 5. Compute critic_delta.json
```

`critic_delta.json` shape:
```json
{
  "hypothesis_id": "h1",
  "posterior_base": 0.45,
  "posterior_critic": 0.38,
  "top_driver_change": ["evi_x removed", "evi_y added"],
  "critic_findings_count": 3
}
```

---

### CLI change (pt/cli.py)

```python
parser.add_argument("--critic", action="store_true", default=False,
    help="Run structural critic pass after testing. Triggers re-elicitation if high-severity findings.")
```

---

### Report change (pt/report.py)

Add a collapsed section after the Diagnostic Test Matrix section:

```
Structural Critic (collapsed by default)
├── Summary paragraph
├── Table: finding_type | target | severity | reasoning | recommendation
└── Note if re_elicitation_needed=True: "Pass 3 was re-run; result_critic.json contains updated posteriors"
```

Each `target` in the table should link to the relevant evidence/hypothesis anchor in the report if one exists.

---

### Required tests

Add to `tests/test_pass_critic.py` (new file):

| Test | What it verifies |
|------|-----------------|
| `test_critic_result_schema_roundtrip` | CriticResult serializes/deserializes cleanly |
| `test_critic_finding_type_accepts_all_valid_literals` | All 5 finding types are valid |
| `test_critic_finding_type_rejects_invalid` | Invalid type raises |
| `test_re_elicitation_needed_false_when_no_high_severity` | No high findings → re_elicitation_needed=False |
| `test_critic_result_in_process_tracing_result` | ProcessTracingResult.critic is Optional, defaults None |
| `test_critic_does_not_modify_testing_result` | CriticResult has no LR fields |

Add integration test to `tests/test_pipeline_integration.py`:
- `test_critic_off_produces_no_critic_field` — pipeline with `critic=False` leaves `result.critic = None`
- `test_critic_on_requires_base_and_critic_artifacts` — pipeline with `critic=True` writes both JSON files

---

### E2E commands

```bash
# Base run (critic OFF)
PYTHONPATH=. python -m pt input_text/revolutions/french_revolution.txt \
    --output-dir output/slice7_base --model openrouter/openai/gpt-5-mini --json-only

# Critic run (critic ON)
PYTHONPATH=. python -m pt input_text/revolutions/french_revolution.txt \
    --output-dir output/slice7_critic --model openrouter/openai/gpt-5-mini --json-only --critic

# Verify delta exists and is non-trivial
python3 -c "
import json
delta = json.load(open('output/slice7_critic/critic_delta.json'))
print('Delta entries:', len(delta))
for d in delta:
    print(f\"  {d['hypothesis_id']}: base={d['posterior_base']:.3f} critic={d['posterior_critic']:.3f}\")
"

# Audit both
PYTHONPATH=. python scripts/audit_result_quality.py output/slice7_base/result.json \
    --report output/slice7_base/report.html
PYTHONPATH=. python scripts/audit_result_quality.py output/slice7_critic/result.json \
    --report output/slice7_critic/report.html

# Deterministic tests
PYTHONPATH=. pytest tests/ -q --tb=short
```

---

### Success criteria

1. `result.critic` is populated in the critic run, `None` in the base run
2. `critic_delta.json` shows at least one hypothesis with a posterior change > 0.01 when `re_elicitation_needed=True`
3. No evidence of direct LR mutation — all numeric changes traceable to re-elicitation (new `trace_id` visible in logs)
4. At least one `CriticFinding` with `finding_type="confound"` or `"missing_pathway"` on the French Revolution or Brumaire case
5. Deterministic test count >= 135 (no regressions)
6. Audit grade not below B on either output

---

## Current state of the codebase

### Active pass files

| File | Pass | LLM? |
|------|------|------|
| `pt/pass_extract.py` | 1: Extract | Yes |
| `pt/pass_hypothesize.py` | 2: Hypothesize | Yes |
| `pt/pass_test.py` | 3: Test (likelihood matrix) | Yes |
| `pt/pass_absence.py` | 3b: Absence | Yes |
| `pt/pass_diagnostic.py` | 3.6: Diagnostic matrix (deterministic) | No |
| `pt/bayesian.py` | 3.5: Bayesian update (pure math) | No |
| `pt/pass_synthesize.py` | 4: Synthesize | Yes |
| `pt/pass_refine.py` | 5: Refine | Yes |
| `pt/pipeline.py` | Orchestrator | No |
| `pt/cli.py` | CLI entry point | No |
| `pt/report.py` | HTML report | No |
| `pt/schemas.py` | All Pydantic models | No |
| `pt/llm.py` | LLM boundary | Yes |
| `scripts/audit_result_quality.py` | Audit grader | No |

### Key schema types (pt/schemas.py)

`ProcessTracingResult` is the top-level output. Fields: `extraction`, `hypothesis_space`, `testing`, `absence`, `bayesian`, `synthesis`, `source_packet?`, `source_coverage?`, `partition_audit?`, `diagnostic_matrix?`, `refinement?`, `is_refined`, `critic?` (to add).

New in Slices 3–6:
- `Evidence.{source_genre, source_group, date_confidence, trace_production_relevance}` — all Optional
- `EvidenceCluster.lineage_type` — Optional[LineageType]
- `DiagnosticMatrix` + `RivalPairDiagnostic` + `RivalDiscriminator` — deterministic
- `AbsenceEvaluation.{expected_source_genre, expected_source_location}` — both Optional

### Default model / provider

Gemini-2.5-flash (quota currently exhausted). Use `--model openrouter/openai/gpt-5-mini` for E2E runs.

### Test suite

```bash
# Fast deterministic suite (135 tests, ~2s)
PYTHONPATH=. pytest tests/test_pt_bayesian.py tests/test_pt_schemas.py \
    tests/test_extraction_quality.py tests/test_pass_diagnostic.py -q

# Full suite (some failures expected from Gemini quota)
PYTHONPATH=. pytest tests/ -q --tb=short
```

---

## Files that must NOT be edited directly

- `docs/plans/CLAUDE.md` — generated by plan-status sync script
- `AGENTS.md` — generated mirror of CLAUDE.md; update canonical CLAUDE.md instead

## Untracked files (cleanup deferred)

```
docs/plans/005d_visual_audit_mockup.html   — superseded mockup, safe to delete
docs/plans/005d_workbench_mockup_v2.html   — superseded mockup, safe to delete
docs/plans/005d_workbench_mockup_v3.html   — superseded mockup, safe to delete
pt/schemas_view.py                          — verify not imported: grep -r "schemas_view" pt/ tests/
```

## Remote state

All commits are pushed. `git push origin master` is a no-op.

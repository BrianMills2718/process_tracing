# Plan 003 Slice 3 Evidence: Extraction Provenance, Source Metadata, And Trace-Production Hooks

## Implementation Summary

**Date:** 2026-06-25
**Plan:** 003 SOTA+ Execution Master Plan
**Slice:** 3 — Extraction Provenance, Source Metadata, And Trace-Production Hooks

### What was built

1. **`pt/schemas.py`** — Added three new Literal type aliases:
   - `SourceGenre`: 9-value controlled vocabulary (overview, primary_document, speech,
     legal_constitutional, memoir, parliamentary_record, secondary_analysis, news_dispatch, other)
   - `DateConfidence`: "high" | "medium" | "low"
   - `TraceProductionRelevance`: "direct" | "indirect" | "background"

   Added 4 new Optional fields to `Evidence` (all default None for backward compat):
   - `date_confidence: Optional[DateConfidence]`
   - `source_group: Optional[str]` — free-text section label
   - `source_genre: Optional[SourceGenre]` — controlled vocabulary
   - `trace_production_relevance: Optional[TraceProductionRelevance]`

2. **`pt/prompts/pass1_extract.yaml`** — Added "Source provenance metadata" section
   instructing the LLM to fill all 4 new fields for every evidence item, with explicit
   value definitions for each controlled vocabulary.

3. **`pt/report.py`** — Extended Evidence Inventory table (Section 7) with 2 new columns:
   - "Genre / Group": source_genre badge (color-coded by genre) + source_group label
   - "Trace role": trace_production_relevance badge (green=direct, yellow=indirect, grey=background)

4. **`tests/test_extraction_quality.py`** — 16 new deterministic tests:
   - `TestEvidenceProvenanceFields` (13 tests): defaults, valid literals, invalid rejection,
     free-text group, roundtrip JSON, backward compat with old result.json, prompt contracts
   - `TestSourceGenreType` (1 test): all 9 genre values
   - `TestDateConfidenceType` (1 test): all 3 confidence values
   - `TestTraceProductionRelevanceType` (1 test): all 3 trace values

### Test results

```
214 deterministic tests passed (all Slice 3 tests: 20/20 in test_extraction_quality.py)
```

Live-LLM tests in test_pipeline_integration.py fail due to Gemini free-tier daily quota
exhaustion — pre-existing condition, not caused by Slice 3 changes.

### E2E result

**Command:**
```bash
python -m pt input_text/revolutions/french_revolution.txt \
  --output-dir output/slice3_provenance_e2e_openrouter_20260625 \
  --model openrouter/openai/gpt-5-mini \
  --json-only
```

**Run directory:** `output/slice3_provenance_e2e_openrouter_20260625`
**Model:** `openrouter/openai/gpt-5-mini`
**Audit grade:** B (80/100)
**Pipeline:** Extract→Hypothesize→Partition→Test→Absence→Bayesian→Synthesize (658.4s)

**Provenance field coverage (37 evidence items):**
- `source_genre`: 37/37 (34 overview, 3 secondary_analysis)
- `source_group`: 37/37 (all "Main text" — undivided Wikipedia article)
- `date_confidence`: 37/37 (28 high, 6 low, 3 medium)
- `trace_production_relevance`: 37/37 (24 direct, 11 background, 2 indirect)

**Report surface:**
- "Genre / Group" and "Trace role" columns appear in Evidence Inventory table ✅
- Genre badges color-coded by type ✅
- Trace role badges (green=direct, yellow=indirect, grey=background) ✅

### Success criteria status

| Criterion | Status |
|---|---|
| 4 new fields in Evidence schema | ✅ |
| All new fields Optional, default None | ✅ |
| Old result.json backward compat | ✅ (test_old_evidence_without_new_fields_loads_cleanly) |
| Prompt instructs LLM to fill all 4 fields | ✅ |
| E2E: all fields populated by LLM | ✅ 37/37 |
| Report: Genre / Group column | ✅ |
| Report: Trace role column | ✅ |
| Deterministic tests | ✅ 20/20 |
| make audit-result grade | ✅ B (80/100) |

### Strongest concern remaining

The Wikipedia article is a single undivided secondary overview, so `source_group = "Main text"`
for all 37 items and `source_genre = "overview"` for 34/37 — no multi-source discrimination.
The fields will be more useful when a source packet with distinct source blocks is provided
(e.g., Brumaire with primary documents + secondary analysis + parliamentary records).

Trace-production relevance shows meaningful variation (24 direct, 11 background, 2 indirect)
and is the most immediately useful new field for auditing which evidence items are mechanistic
vs. contextual.

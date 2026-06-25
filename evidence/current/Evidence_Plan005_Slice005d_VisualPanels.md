# Slice 005d Evidence: Visual Audit Panels

## Task 17 — Live E2E Verification

**Date:** 2026-06-25  
**Run directory:** `output/workbench_runs/run_20260625_114420_6710`  
**Model:** gemini/gemini-2.5-flash (real LLM run)  
**Input:** 18 Brumaire input text (source packet: 3 sources, 1 known gap)

### Commands run

```bash
# View renderer endpoints (all four)
python -c "from pt.view_renderer import build_view_payload; ..."  # see session

# Audit
make audit-result \
  RESULT=output/workbench_runs/run_20260625_114420_6710/result.json \
  REPORT=output/workbench_runs/run_20260625_114420_6710/report.html
```

### Panel verification results

| Stage | Endpoint | Result |
|-------|----------|--------|
| test | `build_matrix_payload` | 14 rows, 0 below threshold, 2 hypotheses — OK |
| update | `build_support_payload` | 2 bars, fragile_warning=True, rank_instability=False — OK |
| synthesize | `build_provenance_payload` | 14 rows, 0 with marker — OK |
| refine | N/A — run used refine=False | N/A |

Matrix top row: `evi_background_00` — diag=hoop  
Support: h1 post=0.9991 fragile lo=0.995 hi=1.000; h2 post=0.0009 fragile lo=0.000 hi=0.005  
Provenance: peak_lr=2.00 favored=h1 (background item)

### Audit grade: C (76/100)

**Root cause of C grade (pre-existing, not caused by Slice 005d):**
- Source packet has 0 evidence assigned — markers not present in input text
- Background conditions dominate; no proximate top drivers
- h2 labeled "supported" with posterior 0.001 (verdict calibration issue)

**Slice 005d verdict:** View endpoints produce correct typed payloads and render
without error. Grade C reflects source quality limitations in this run, not bugs
in the view rendering layer.

### HTTP fixture endpoint (demo mode)

```
GET /fixtures/brumaire_demo.json → 200 OK
hypothesis_space.hypotheses: 5 (h1–h5)
testing.top_evidence: 10 items
bayesian.posteriors: h2=0.9942, h1=0.0047, h4=0.0008, h3=0.0003, h5=0.0
```

Demo mode renders matrix + support bars + provenance from fixture on page load. ✓

### make check

226 tests pass (including 27 new view_renderer tests). ✓

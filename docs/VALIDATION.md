# Process Tracing Validation

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Current Validation Posture

This project has meaningful engineering and artifact validation. It does not
yet have enough empirical validation to claim automated PhD-level historical
inference.

Validated today:

- structured pipeline stages exist and are tested
- LLM calls are routed through `llm_client` boundaries
- deterministic Bayesian math and calibration live outside the LLM
- reports expose comparative support, sensitivity, source-scope, audit, and
  caveat surfaces
- a public French Directory case bundle demonstrates the workflow shape
- `make audit-result` grades report outputs against academic evidence caps

Not yet validated:

- stable PhD-quality performance across a frozen benchmark
- human/adversarial comparison against expert process tracing
- general source acquisition adequacy
- hypothesis partition thresholds across multiple cases
- full trace-production and per-hypothesis dependence modeling

## Verification Commands

Core local gates:

```bash
make check
pytest tests/ -q
python scripts/check_markdown_links.py
```

Run a public case:

```bash
python -m pt input_text/revolutions/french_revolution.txt \
  --output-dir output/french_rev
```

Audit a generated result/report pair:

```bash
make audit-result RESULT=output/french_rev/result.json REPORT=output/french_rev/report.html
```

Run with a source packet:

```bash
make source-packet-run \
  INPUT=input_text/revolutions/french_revolution.txt \
  SOURCE_PACKET=docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  RUN_OUTPUT_DIR=output/source_packet_run
```

## Evidence Types

| Evidence | Current Status | Interpretation |
|---|---|---|
| Unit and integration tests | Present | Necessary for code correctness and regression control. |
| Live E2E run requirement | Documented in `CLAUDE.md` | Required for pipeline/prompt/report changes; avoids mock-only confidence. |
| Output quality rubric | Present | Gives report-grade criteria and evidence caps. |
| French Directory case bundle | Present | Good reviewer walkthrough; not a benchmark result. |
| Source packet/source coverage | Partial and active | Improves source-scope honesty; does not prove source adequacy by itself. |
| SOTA review | Present | Defines external comparison surface. |
| Frozen benchmark | Missing | Required before strong methodological-validity claims. |
| Human/adversarial audit study | Missing | Required before claiming PhD-level automated analyst performance. |

## Current Limitations

The most important limits are methodological, not cosmetic:

- Overview texts can produce clean reports while still being too thin for
  publication-strength causal inference.
- Source packets can define the desired corpus without proving the corpus has
  been acquired or used adequately.
- Dependence pooling reduces raw double-counting but does not yet model all
  source-lineage and trace-production uncertainty.
- A single case can demonstrate workflow shape but cannot calibrate quality
  thresholds.
- Generated support values should be read as comparative rankings with
  sensitivity/caveats, not as settled historical probabilities.

## Portfolio Readiness Rule

Use this project as portfolio evidence for:

- method-aware AI engineering,
- structured LLM systems,
- computational social-science workflow design, and
- analyst-facing causal reasoning support.

Do not use it yet as evidence that the system is a validated autonomous
historian or a production intelligence platform.

# Process Tracing Reviewer Walkthrough

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Portfolio Claim

This project is the strongest analyst-facing portfolio evidence because it operationalizes a familiar analytic method: competing hypotheses, diagnostic evidence, causal mechanisms, and explicit updating. The project should be framed as an auditable analysis workflow, not as generic qualitative coding.

## Walkthrough Shape

Use one question, one evidence set, and one final report path. The current
reviewer-safe case bundle is
`docs/portfolio/FRENCH_DIRECTORY_COLLAPSE_CASE_BUNDLE.md`.

1. Start with a case question that has rival explanations.
2. Extract actors, events, mechanisms, causal edges, and evidence records.
3. Generate competing hypotheses.
4. Test evidence with a coherent likelihood vector across all hypotheses.
5. Track absence of evidence separately from evidence of absence.
6. Read the temporal causal network, support table, PhD audit, and final
   narrative together; no single view is the whole analysis.

## What A Reviewer Should Inspect

| Artifact | Why it matters |
|----------|----------------|
| `README.md` | Shows the end-to-end pipeline and validated input texts |
| `docs/PROJECT_THEORY_AND_GOALS.md` | States the project goal, current capability ledger, and claim discipline |
| `docs/WHITEPAPER_optimal_automated_process_tracing.md` | Explains the methodology and why the workflow is analytically meaningful |
| `docs/ontology.md` | Shows the typed analytic objects behind the workflow |
| `docs/OUTPUT_QUALITY_RUBRIC.md` | Shows how generated reports are graded and capped |
| `evidence/current/Evidence_Plan003*.md` | Shows how active SOTA+ claims are grounded in inspectable evidence artifacts |
| `docs/portfolio/FRENCH_DIRECTORY_COLLAPSE_CASE_BUNDLE.md` | Shows one public-source question through evidence, hypothesis updates, absence checks, and report paths |

## Analyst Differentiator

The useful portfolio signal is the method architecture: process tracing becomes a structured workflow with intermediate objects and update logic. That is more relevant to intelligence analysis than a surface-level "summarize these documents" demo.

## Caveat

The methodology evidence is stronger than product polish. The French Directory
case bundle is safe for reviewer orientation because it uses public historical
source material, but it is still a compact excerpt. For a full demo, rerun the
pipeline and inspect the generated `result.json` and `report.html`.

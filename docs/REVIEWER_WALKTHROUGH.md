# Process Tracing Reviewer Walkthrough

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Portfolio Claim

This project is the strongest analyst-facing portfolio evidence because it operationalizes a familiar analytic method: competing hypotheses, diagnostic evidence, causal mechanisms, and explicit updating. The project should be framed as an auditable analysis workflow, not as generic qualitative coding.

## Walkthrough Shape

Use one question, one evidence set, and one final report path:

1. Start with a case question that has rival explanations.
2. Extract actors, events, mechanisms, causal edges, and evidence records.
3. Generate competing hypotheses.
4. Test each hypothesis against evidence using diagnostic tests and likelihood-ratio style updates.
5. Track absence of evidence separately from evidence of absence.
6. Synthesize the final narrative with visible support and caveats.

## What A Reviewer Should Inspect

| Artifact | Why it matters |
|----------|----------------|
| `README.md` | Shows the end-to-end pipeline and validated input texts |
| `docs/WHITEPAPER_optimal_automated_process_tracing.md` | Explains the methodology and why the workflow is analytically meaningful |
| `docs/ontology.md` | Shows the typed analytic objects behind the workflow |
| `evidence/*.md` | Shows how claims are grounded in inspectable evidence artifacts |

## Analyst Differentiator

The useful portfolio signal is the method architecture: process tracing becomes a structured workflow with intermediate objects and update logic. That is more relevant to intelligence analysis than a surface-level "summarize these documents" demo.

## Caveat

The methodology evidence is stronger than product polish. Before external sharing, use a small sanitized case and show the path from question to evidence diagnostics to hypothesis updates without relying on hidden session context.

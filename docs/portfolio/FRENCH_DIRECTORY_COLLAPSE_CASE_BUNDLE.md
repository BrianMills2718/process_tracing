# French Directory Collapse Case Bundle

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Purpose

This is a reviewer-facing, public-source case bundle for the Process Tracing
portfolio story. It connects one question to rival hypotheses, diagnostic
evidence, absence checks, Bayesian support updates, and generated report paths.

The full local run artifacts are generated files and are intentionally not
tracked in git:

| Artifact | Local path |
|----------|------------|
| Input text | `input_text/revolutions/french_revolution.txt` |
| Structured result | `output/rebuild_smoke/result.json` |
| HTML report | `output/rebuild_smoke/report.html` |
| Alternative HTML report | `output/rebuild_smoke/report_v2.html` |

To regenerate the same artifact shape:

```bash
python -m pt input_text/revolutions/french_revolution.txt --output-dir output/rebuild_smoke
```

Model outputs may differ across providers and time. Treat this page as a
compact, inspectable excerpt from the checked local run, not as a frozen
benchmark claim.

## Case Question

Why did the French Directory collapse into Napoleon's coup rather than evolving
into a stable republic?

## Run Shape

| Object | Count in `output/rebuild_smoke/result.json` |
|--------|---------------------------------------------|
| Actors | 10 |
| Events | 21 |
| Mechanisms | 9 |
| Evidence records | 83 |
| Rival hypotheses | 4 |
| Absence evaluations | 13 |

## Rival Hypotheses

| ID | Hypothesis |
|----|------------|
| `h1` | The Directory collapsed because persistent institutional instability and ineffective elite governance created a preference for authoritarian rule. |
| `h2` | The collapse was primarily caused by external military threats and internal counter-revolutionary movements, which empowered the military and favored a strong executive. |
| `h3` | The Thermidorian Reaction demobilized popular republican participation and suppressed radical republicanism, creating a political vacuum. |
| `h4` | Napoleon's personal ambition, strategic opportunism, and military charisma were decisive because he actively orchestrated the coup for himself. |

## Evidence Excerpts

| Evidence ID | Source-backed observation | Why it is diagnostic |
|-------------|---------------------------|----------------------|
| `evi_coup_1797` | Republicans used troops from Napoleon's Army of Italy to force approval of Royalist arrests, annul elections, deport 63 Royalists, and pass laws against regime enemies. | Directly tests institutional instability and reliance on military force. The model scored it `h1=4.0`, `h2=3.0`, `h3=0.5`, `h4=2.0`, relevance `1.0`. |
| `evi_directory_collapse_authoritarian_preference` | By 1799, many reportedly preferred authoritarian uncertainty to parliamentary ambiguity. | Tests whether the endpoint was a broader legitimacy failure rather than only Napoleon's ambition. The model scored it `h1=3.0`, `h2=2.0`, `h3=0.5`, `h4=2.5`, relevance `0.9`. |
| `evi_calonne_reforms_rejected` | Calonne's proposed universal land tax and other reforms were rejected by the Assembly of Notables and the parlements. | Supports the longer institutional-failure pathway that predates the Directory. The model scored it `h1=3.0`, `h2=1.0`, `h3=1.5`, `h4=1.0`, relevance `0.7`. |
| `evi_declaration_pillnitz` | Leopold II and Frederick William II declared support for Louis XVI and hinted at invasion. | Supports the external-threat hypothesis. The model scored it `h1=2.0`, `h2=4.0`, `h3=1.0`, `h4=1.0`, relevance `0.9`. |
| `evi_war_first_coalition` | Britain, the Dutch Republic, Austria, Prussia, Spain, Portugal, Naples, and Tuscany joined conflict against revolutionary France. | Also supports the external-threat pathway. The model scored it `h1=2.0`, `h2=4.0`, `h3=1.0`, `h4=1.0`, relevance `0.9`. |
| `evi_law_22_prairial` | The Law of 22 Prairial denied enemies of the people the right to defend themselves and executions increased sharply. | Tests legal-institutional breakdown and weighs against a clean popular-republican-demobilization account. The model scored it `h1=4.0`, `h2=1.0`, `h3=0.5`, `h4=1.0`, relevance `0.9`. |
| `evi_economy_stabilized_1799` | By 1799 the economy had stabilized and reforms enabled industrial expansion. | Counter-evidence against a simple "the regime failed in every domain" account. The model scored it `h1=0.5`, `h2=1.0`, `h3=1.0`, `h4=1.0`, relevance `0.7`. |

## Support Update Result

The run began with equal priors over four hypotheses and produced this final
support ranking:

| Rank | Hypothesis | Final support | Sensitivity range | Rank stable? |
|------|------------|---------------|-------------------|--------------|
| 1 | `h1` institutional instability | `0.947563` | `0.753143` to `0.989272` | yes |
| 2 | `h2` external/internal security threats | `0.025577` | `0.005488` to `0.091808` | no |
| 3 | `h3` demobilized republican participation | `0.025486` | `0.002377` to `0.182329` | no |
| 4 | `h4` Napoleon's personal ambition | `0.001374` | `0.001092` to `0.001435` | yes |

The important portfolio signal is the traceable update process, not the exact
numeric magnitude. The repo's current methodology caveat still applies:
condition-dependent evidence can make aggregate support overconfident.

## Absence Checks

The absence pass found 13 missing-prediction checks. Three reviewer-relevant
examples:

| Hypothesis | Missing predicted evidence | Severity | Interpretation |
|------------|----------------------------|----------|----------------|
| `h1` | Specific, significant Directory attempts to implement lasting constitutional reforms were not present in the source text. | notable | Weakens an overly broad institutional-failure account because reform attempts, if central, should be visible. |
| `h2` | The text did not show public and elite sentiment explicitly prioritizing national security and military strength over constitutional stability before the coup. | notable | Limits the security-threat explanation even though war evidence is present. |
| `h4` | The text did not provide direct evidence of Napoleon's personal planning and decisive execution of 18 Brumaire. | damaging | Strongly limits the "Napoleon ambition alone" explanation in this source. |

## What This Demonstrates

For a CIA analyst or AI-engineering reviewer, this case demonstrates:

1. The system forms rival explanations rather than a single summary.
2. Evidence records keep source-backed observations separate from hypothesis
   support.
3. Diagnostic testing is explicit: the same evidence can support one hypothesis,
   weakly support another, or count against a third.
4. Absence-of-evidence is recorded separately from positive evidence.
5. The final report is auditable through `result.json` and `report.html`.

## Known Caveats

- The source text is a public historical overview, not a full archival record.
- The run is evidence for workflow shape and method implementation, not a
  historical finding about the French Revolution.
- The joint update can overstate confidence when evidence items are dependent.
- The generated outputs under `output/` are local build artifacts and should be
  regenerated before any external demonstration.

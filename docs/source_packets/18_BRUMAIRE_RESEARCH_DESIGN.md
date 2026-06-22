# 18 Brumaire Source-Packet Research Design

## Purpose

The prior audit reached `C` because the report was clear but the evidence base
was too broad, background-heavy, and weakly diagnostic. This design defines the
next iteration: rerun the pipeline on a multi-source packet with primary legal
texts, official proclamations, a narrative secondary account, and a critical
historiographical account.

## Corpus

- Input packet: `input_text/source_packets/18_brumaire_source_packet.txt`
- Source-packet contract:
  `docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json`
- Theory lenses: `input_text/theories/18_brumaire_rival_frameworks.txt`
- Source groups:
  - Bonaparte justification, 10 November 1799: https://revolution.chnm.org/d/461
  - Brumaire Decree and Proclamation of the Consuls: https://www.napoleon-series.org/research/government/legislation/c_brumaire.html
  - Constitution of the Year VIII: https://www.napoleon-series.org/research/government/legislation/c_constitution8.html
  - Eymeric Job, Fondation Napoleon narrative account: https://www.napoleon.org/en/history-of-the-two-empires/articles/18-brumaire-the-context-and-course-of-a-coup-detat/
  - Malcolm Crook, critical H-France account: https://h-france.net/the-myth-of-the-18th-brumaire/

## Research Question

Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup
and the Consulate rather than a stable parliamentary republic, a revived
Jacobin-dominated republic, or a royalist restoration?

## Rival Mechanisms

1. Elite constitutional revision: Sieyes and allied civilians initiated the coup
   to replace the Directory with a stronger executive, using Bonaparte as a
   risky military instrument.
2. Bonapartist personalist-military conversion: Bonaparte, Lucien, Murat, and
   loyal troops converted a faltering civilian plot into personal rule.
3. Structural regime collapse: war, prior coups, factionalism, and security
   threats made the Directory unsustainable regardless of the individual coup
   coalition.
4. Popular demand for order: public exhaustion and desire for stability made
   Bonaparte's solution broadly accepted before the new regime was designed.
5. Fear-of-extremes avoidance: elites chose Brumaire primarily to avoid Jacobin
   revival or royalist restoration.

## Pre-Specified Tests

| Test | Evidence that would pass | Most helped mechanism | Most hurt mechanism |
|---|---|---|---|
| Civilian initiation hoop | Sieyes planned revision and sought a military backer before Bonaparte dominated the outcome | Elite constitutional revision | Pure Bonapartist authorship |
| Military coercion smoking gun | Deputies resisted and were dispersed or rounded up by troops before ratification | Bonapartist personalist-military conversion | Voluntary parliamentary revision |
| Constitutional concentration smoking gun | Final constitution made First Consul decision sufficient and gave him major appointments | Bonapartist personalist-military conversion | Mere anti-Jacobin emergency |
| Directory viability discriminator | Evidence of ongoing electoral, press, or republican capacity in 1799 | Contingent path closure | Inevitable structural collapse |
| Popular legitimacy hoop | Independent evidence of mass demand before Brumaire, not only post-coup proclamations | Popular demand for order | Elite imposition/apathy |
| Fear-of-extremes discriminator | Direct elite statements that Jacobin or royalist alternatives drove the choice of Bonaparte | Fear-of-extremes avoidance | Personalist ambition as sufficient cause |

## Run Command

```bash
python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir /tmp/pt_smoke/brumaire_packet \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --max-budget 1.0

make audit-result RESULT=/tmp/pt_smoke/brumaire_packet/result.json \
  REPORT=/tmp/pt_smoke/brumaire_packet/report.html \
  FOCAL_YEAR=1799
```

## A-Gate Expectations

An improved result should clear these gates:

- Multiple source groups appear in extracted evidence; the synthesis must not
  treat the corpus as a single historical source.
- `result.json` stores source-packet metadata, and the report includes the
  source-packet contract table plus the rule that packet metadata is not itself
  evidence.
- At least 20 percent of evidence is proximate to 1799.
- The top hypothesis has at least one proximate top driver.
- At least one item reaches moderate diagnostic strength.
- The report triages unlinked evidence instead of hiding it.
- The optimality gate says `optimal_for_current_corpus`, or the remaining cap
  names a concrete external source class still missing from the packet.

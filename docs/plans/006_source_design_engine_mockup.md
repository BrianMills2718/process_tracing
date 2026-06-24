# Plan 006 Source Design Engine Mockup

This mockup is the pre-implementation seam artifact for the source-design
engine slice. It shows the minimum artifact shapes the implementation must
produce.

## 1. Initial Source Design State

```json
{
  "case_name": "18 Brumaire",
  "research_question": "Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup?",
  "iteration": 1,
  "source_candidates": [
    {
      "source_id": "source_f_bourrienne_memoirs",
      "source_kind": "memoir",
      "expected_observability": "Private planning sequence and actor coordination"
    }
  ],
  "source_gaps": [
    {
      "missing_source_class": "Private correspondence among conspirators",
      "priority": "high"
    }
  ],
  "acquisition_actions": [],
  "review_log": []
}
```

## 2. Acquisition Action Output

```json
{
  "action_id": "acq_gap_1",
  "target_id": "acq_gap_1",
  "status": "proposed",
  "target_source_class": "Private correspondence among conspirators",
  "evidence_need": "Private planning sequence, coalition bargaining, and whether Bonaparte converted the plot into personalist rule.",
  "inferential_payoff": "Determines whether the main claim-scope cap can be lifted or must remain.",
  "stop_rule": "Stop when direct correspondence is acquired or repeated searches only return adjacent substitutes."
}
```

## 3. Candidate Review Output

```json
{
  "candidate_id": "cand_001",
  "action_id": "acq_gap_1",
  "decision": "reject_as_adjacent",
  "rationale": "Memoir evidence adds planning detail but does not satisfy the direct-correspondence gap.",
  "admitted_source_id": null
}
```

## 4. Updated Gap Disposition

```json
{
  "missing_source_class": "Private correspondence among conspirators",
  "status": "partially_mitigated",
  "claim_implications": "Private-planning sequence is better observed, but publication-strength intent claims remain capped.",
  "disposition_reason": "An adjacent memoir source improved observability, but the direct source class remains unresolved."
}
```

## 5. Reviewer Questions

- Can I see which gaps are still open, which were searched, and why the search stopped?
- Can I tell the difference between a retrieved candidate, an admitted source, and actual evidence in the trace?
- Can I compare the source-design state before and after a rerun without reading code?

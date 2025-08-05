Process Tracing Ontology - Phased Implementation

## Current Implementation Status

**Phase 0 (Current)**: Basic Event-Evidence-Hypothesis patterns ✅  
**Phase 1 (Next)**: Core Mechanism Analysis 🚧  
**Phase 2 (Planned)**: Counter-evidence & Alternatives 📋  
**Phase 3 (Future)**: Agency & Context 📋  

---

## CORE NODE TYPES (Phase 0 - Implemented)

  Event 🔵 (#66b3ff)

  - description (string, required): Event description
  - timestamp (datetime, optional): Precise timestamp
  - date, start_date, end_date (string, optional): Date information
  - location (string, optional): Where event occurred
  - certainty (float 0.0-1.0, optional): Confidence event occurred
  - type (optional): "triggering", "intermediate", "outcome", "unspecified"
  - is_point_in_time (boolean, optional): Whether event is instantaneous

  Hypothesis 🟡 (#ffcc00)

  - description (string, required): Testable causal statement
  - prior_probability (float 0.0-1.0, optional): Probability before evidence
  - posterior_probability (float 0.0-1.0, optional): Probability after evidence
  - status (optional): "active", "supported", "partially_supported", "refuted", "undetermined"

  Evidence 🔴 (#ff6666)

  - description (string, required): Evidence description
  - type (required): "hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "bayesian", "general"
  - certainty (float 0.0-1.0, optional): Confidence in evidence
  - source (string, optional): Source identification
  - credibility (float 0.0-1.0, optional): Source credibility

## PHASE 1 ADDITION: Core Mechanism Analysis 🚧

  Causal_Mechanism 🟢 (#99ff99)

  - description (string, required): How the mechanism works step-by-step
  - confidence (float 0.0-1.0, optional): Confidence mechanism operates as described
  - completeness (float 0.0-1.0, optional): How complete our understanding is
  - status (optional): "hypothetical", "supported", "refuted", "partial", "unspecified"
  - testable_predictions (array, optional): What evidence would confirm/refute operation

## PHASE 2 ADDITIONS: Counter-evidence & Alternatives 📋

  Alternative_Explanation 🟠 (#ff9966)
  
  - description (string, required): Alternative causal explanation
  - probability (float 0.0-1.0, optional): Estimated likelihood
  - status (optional): "active", "eliminated", "supported", "undetermined"
  - key_predictions (array, optional): Distinguishing implications

## PHASE 3 ADDITIONS: Agency & Context 📋

  Actor 🩷 (#ff99cc)
  
  - name (string, required): Actor identification
  - role (string, optional): Position or function
  - intentions (string, optional): Goals and motivations
  - beliefs (string, optional): What they believe to be true
  - constraints (string, optional): Limitations on their actions
  - capabilities (string, optional): What they can actually do

  Condition 🟣 (#ccccff)
  
  - description (string, required): Condition description
  - type (required): "background", "enabling", "constraining", "scope"
  - necessity (float 0.0-1.0, optional): How necessary for outcomes
  - temporal_scope (string, optional): When condition applies

## RESEARCH-LEVEL EXTENSIONS (Future Consideration)

  - Data_Source 🔷 (#c2c2f0): Source documents/interviews/observations
  - Inference_Rule 🟪 (#cc99ff): Logical reasoning rules
  - Inferential_Test 🟤 (#ffb366): Formal hypothesis tests

---

## EDGE TYPES BY IMPLEMENTATION PHASE

### PHASE 0 EDGES (Currently Implemented) ✅

**Basic Evidence-Hypothesis Testing:**

  supports

  - Domain: [Evidence, Event]
  - Range: [Hypothesis, Event, Causal_Mechanism]
  - Properties:
    - probative_value (0.0-1.0), certainty (0.0-1.0)
    - source_text_quote, description
    - diagnostic_type: "hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "general"
    - target_type: "event_occurrence", "causal_relationship", "mechanism_operation", "general"

  refutes

  - Domain: [Evidence, Event]
  - Range: [Hypothesis, Event, Causal_Mechanism]
  - Properties: Same as supports + diagnostic_type + target_type

  tests_hypothesis

  - Domain: [Evidence, Event]
  - Range: [Hypothesis]
  - Properties: probative_value, test_result ("passed", "failed", "ambiguous"), diagnostic_type

  tests_mechanism

  - Domain: [Evidence, Event]
  - Range: [Causal_Mechanism]
  - Properties: probative_value, test_result, diagnostic_type

  🔥 NEW ACADEMIC EDGE TYPES

  confirms_occurrence

  - Domain: [Evidence]
  - Range: [Event]
  - Properties: certainty, source_text_quote, diagnostic_type
  - Purpose: Evidence confirms an event actually happened

  disproves_occurrence

  - Domain: [Evidence]
  - Range: [Event]
  - Properties: certainty, source_text_quote, diagnostic_type
  - Purpose: Evidence disproves an event happened

  provides_evidence_for

  - Domain: [Event]
  - Range: [Hypothesis, Causal_Mechanism]
  - Properties: probative_value, reasoning, diagnostic_type
  - Purpose: Events serve as evidence for broader claims

  🔥 TRADITIONAL CAUSAL CONNECTIONS

  causes

  - Domain: [Event] → Range: [Event]
  - Properties: certainty, mechanism_id, type ("direct", "indirect")

### PHASE 1 EDGES (Mechanism Analysis) 🚧

  **part_of_mechanism**
  
  - Domain: [Event] → Range: [Causal_Mechanism]
  - Properties: 
    - role (string): "trigger", "intermediate", "outcome", "facilitating"
    - sequence_position (int, optional): Step number in mechanism
    - necessity (float 0.0-1.0, optional): How necessary for mechanism
  - Purpose: Links events as components of causal mechanisms

  **tests_mechanism**
  
  - Domain: [Evidence, Event] → Range: [Causal_Mechanism]  
  - Properties:
    - probative_value (float 0.0-1.0): Strength of mechanism test
    - test_result (string): "passed", "failed", "ambiguous", "inconclusive"
    - diagnostic_type: Van Evera types for mechanism testing
    - mechanism_aspect (string): "existence", "operation", "completeness"
  - Purpose: Evidence testing whether mechanisms operate as theorized

  **explains_mechanism**
  
  - Domain: [Hypothesis] → Range: [Causal_Mechanism]
  - Properties: 
    - certainty (float 0.0-1.0): Confidence in explanation
    - type_of_claim (string): "existence", "operation", "necessity", "sufficiency"
    - scope (string, optional): Under what conditions explanation applies
  - Purpose: Links theoretical claims to specific mechanisms

### PHASE 2 EDGES (Counter-evidence & Alternatives) 📋

  **refutes**
  
  - Domain: [Evidence, Event] → Range: [Hypothesis, Causal_Mechanism, Alternative_Explanation]
  - Properties: Same as supports + refutation strength
  - Purpose: Evidence that contradicts or undermines claims

  **disproves_occurrence** 
  
  - Domain: [Evidence] → Range: [Event]
  - Properties: certainty, source_text_quote, diagnostic_type
  - Purpose: Evidence showing events did NOT happen

  **supports_alternative / refutes_alternative**
  
  - Domain: [Evidence] → Range: [Alternative_Explanation]  
  - Properties: probative_value, diagnostic_type
  - Purpose: Evidence for/against competing explanations

### PHASE 3 EDGES (Agency & Context) 📋

  **initiates**
  
  - Domain: [Actor] → Range: [Event]
  - Properties: intentionality, capability_assessment
  - Purpose: Actor-driven event causation

  - Domain: [Condition] → Range: [Event, Causal_Mechanism]
  - Properties: necessity, temporal_scope, condition_type
  - Purpose: Background/enabling/constraining factors

### RESEARCH-LEVEL EDGES (Future Consideration)

  **updates_probability** - Bayesian evidence updating
  **contradicts** - Evidence conflicts  
  **infers** - Logical inference patterns
  **provides_evidence** - Source-evidence attribution

---

## 🎯 IMPLEMENTATION PRIORITIES & METHODOLOGY COVERAGE

### Phase 0 (Current): 60% Process Tracing Coverage
- ✅ Van Evera diagnostic tests
- ✅ Basic evidence-hypothesis testing  
- ✅ Event causation chains
- ❌ No mechanism analysis
- ❌ No counter-evidence patterns

### Phase 1 (Mechanism Sprint): 85% Process Tracing Coverage  
- ✅ All Phase 0 capabilities
- ➕ Causal mechanism modeling
- ➕ Mechanism operation testing
- ➕ Event-mechanism linkage
- ➕ Beach & Pedersen theory-testing support

### Phase 2 (Counter-evidence Sprint): 95% Process Tracing Coverage
- ✅ All Phase 1 capabilities  
- ➕ Alternative explanation testing
- ➕ Systematic counter-evidence evaluation
- ➕ George & Bennett congruence method support
- ➕ Explaining-outcome process tracing

### Phase 3 (Agency Sprint): 100% Process Tracing Coverage
- ✅ All Phase 2 capabilities
- ➕ Actor intention/belief modeling
- ➕ Strategic interaction analysis
- ➕ Scope condition specification
- ➕ Complete academic methodology suite

---

## 🎯 KEY ACADEMIC FEATURES

  Van Evera Diagnostic Tests (available on most evidence connections):
  - hoop: Necessary but not sufficient (hypothesis fails if absent)
  - smoking_gun: Sufficient but not necessary (hypothesis confirmed if present)
  - straw_in_the_wind: Neither necessary nor sufficient (weak indicator)
  - doubly_decisive: Both necessary and sufficient (critical evidence)

  Target Types (what's being tested):
  - event_occurrence: Whether an event happened
  - causal_relationship: Whether X caused Y
  - mechanism_operation: How the causation works

  This ontology now supports the full range of academic process tracing methodology with flexible Evidence↔Event↔Hypothesis connections!

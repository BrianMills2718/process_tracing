# Methodological Value Analysis: Missing Edge Types in Process Tracing

## Analysis of 5 Critical Edge Types

Based on the current structured extraction results showing 6/21 edge types captured, this analysis examines the methodological importance of 5 specific missing edge types in process tracing frameworks.

## 1. `confirms_occurrence` - Evidence → Event

**Methodological Framework**: Van Evera (1997) "Guide to Methods for Students of Political Science"
**Process Tracing Context**: Beach & Pedersen (2013) "Process-Tracing Methods"

**Academic Value**:
- **Core Van Evera Function**: Establishes that events actually occurred (vs. being assumed)
- **Diagnostic Purpose**: Provides positive confirmation of event occurrence through documentary/physical evidence
- **Distinguished from `supports`**: More specific - confirms factual occurrence rather than supporting theoretical claims

**Example Usage**:
```
Archaeological evidence → confirms_occurrence → Boston Tea Party event
Ship manifests → confirms_occurrence → Tea dumping event
```

**Why Critical**: Van Evera emphasizes that process tracing requires establishing basic facts before causal analysis. This edge type captures the foundational evidential step.

---

## 2. `infers` - Evidence/Event → Hypothesis/Mechanism

**Methodological Framework**: George & Bennett (2005) "Case Studies and Theory Development"
**Process Tracing Context**: Checkel (2008) "Tracing Causal Mechanisms"

**Academic Value**:
- **Inferential Logic**: Captures logical inference processes from observable evidence to theoretical claims
- **Abductive Reasoning**: Represents "inference to best explanation" methodology
- **Distinguished from `supports`**: More cognitively specific - captures the inferential process itself

**Example Usage**:
```
Communication patterns → infers → Coordination mechanism
Missing correspondence → infers → Isolation hypothesis
```

**Why Critical**: Process tracing methodology explicitly relies on inferential reasoning chains. This edge type captures the logical bridge between observation and theory.

---

## 3. `refutes` - Evidence → Hypothesis/Mechanism/Alternative

**Methodological Framework**: Van Evera (1997) diagnostic tests, Popper (1959) falsification
**Process Tracing Context**: Collier (2011) "Understanding Process Tracing"

**Academic Value**:
- **Falsification Logic**: Core Popperian methodology for theory testing
- **Van Evera Hoop Tests**: Evidence that definitively eliminates hypotheses
- **Distinguished from `refutes_alternative`**: General refutation vs. alternative-specific

**Example Usage**:
```
Absence of French funds → refutes → External influence hypothesis
Working-class participation → refutes → Elite manipulation theory
```

**Why Critical**: Process tracing requires systematic elimination of competing explanations. This is fundamental to establishing causal claims.

---

## 4. `tests_alternative` - Evidence → Alternative_Explanation

**Methodological Framework**: George & Bennett (2005) congruence method
**Process Tracing Context**: Mahoney (2012) "The Logic of Process Tracing"

**Academic Value**:
- **Competitive Testing**: Systematically evaluates competing explanations
- **Congruence Method**: Tests whether evidence fits alternative theoretical predictions
- **Distinguished from general testing**: Specifically targets alternative explanations

**Example Usage**:
```
Merchant records → tests_alternative → Economic self-interest theory
Elite rhetoric patterns → tests_alternative → Elite manipulation theory
```

**Why Critical**: George & Bennett emphasize that process tracing gains strength through competitive testing of alternatives. This edge type captures that methodology.

---

## 5. `weighs_evidence` - Evidence → Evidence/Hypothesis

**Methodological Framework**: Beach & Pedersen (2013) "Process-Tracing Methods"
**Process Tracing Context**: Checkel (2008) evidential weight assessment

**Academic Value**:
- **Evidential Assessment**: Captures the comparative strength evaluation of different evidence pieces
- **Weight Calibration**: Represents Beach & Pedersen's emphasis on evidence quality assessment
- **Meta-Evidential**: Evidence about evidence - second-order analysis

**Example Usage**:
```
Primary source → weighs_evidence → Secondary account (higher credibility)
Contemporary record → weighs_evidence → Later memoir (temporal reliability)
```

**Why Critical**: Process tracing methodology requires explicit assessment of evidential quality and reliability. This edge type captures that meta-analytical process.

---

## Summary Assessment

### Methodological Impact Ranking (High to Low):

1. **`refutes`** (Critical) - Core falsification logic, fundamental to Van Evera diagnostic framework
2. **`confirms_occurrence`** (High) - Basic fact establishment, prerequisite for causal analysis
3. **`tests_alternative`** (High) - Competitive testing, George & Bennett congruence method
4. **`infers`** (Medium-High) - Logical bridging, captures inferential processes
5. **`weighs_evidence`** (Medium) - Meta-analysis, evidence quality assessment

### Implementation Priority:

These 5 edge types represent core methodological functions in process tracing literature. Their absence limits the system's ability to:

- Perform systematic hypothesis elimination (`refutes`)
- Establish basic factual foundations (`confirms_occurrence`) 
- Conduct competitive theory testing (`tests_alternative`)
- Capture inferential reasoning (`infers`)
- Assess evidential quality (`weighs_evidence`)

**Recommendation**: Prioritize implementing `refutes`, `confirms_occurrence`, and `tests_alternative` as they represent the most fundamental process tracing methodologies.
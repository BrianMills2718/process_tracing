# Ontology Refactoring Investigation
# Date: 2025-09-14
# UPDATED: Perfect LLM Academic Expert Extraction Analysis

## Executive Summary - REVISED GOAL

**CLARIFIED OBJECTIVE**: Implement "perfect" LLM extraction where the LLM acts as a full academic expert, providing:
1. **Complete reasoning chains** for all academic judgments
2. **Confidence levels and uncertainty quantification** for all decisions  
3. **Alternative interpretations** and why they were rejected
4. **Mathematical precision** (P(E|H), Bayes factors, etc.) with justification
5. **Step-by-step inference chains** showing how multiple evidence combines

**KEY INSIGHT**: This is not about removing human judgment, but about capturing and making transparent the LLM's academic reasoning process at the level of a domain expert.

This document provides a comprehensive investigation of what would be required to achieve this level of sophisticated LLM academic expertise.

## Current System Overview

### Current Ontology Structure
- **Node types**: 10 (Event, Evidence, Hypothesis, etc.)
- **Edge types**: 19 (supports, tests_hypothesis, etc.) 
- **Key files**: 13+ files directly depend on ontology structure
- **Status**: Academically sound but automation/reproducibility gaps identified

### External Expert Assessment Summary
**Major improvements needed**:
1. Temporal modeling with Allen relations
2. Audit-ready provenance (Citation, Source_Excerpt nodes)
3. N-ary inference steps (replace binary edges with Inference nodes)
4. Evidence/diagnostic type restructuring
5. Mechanism granularity improvements
6. Case scaffolding for comparative work

## Investigation Methodology

### Analysis Approach
1. **Dependency Analysis**: Identify all files using current ontology structure
2. **Breaking Change Assessment**: Determine what changes would break existing code
3. **Implementation Complexity**: Estimate effort for each proposed change
4. **Risk Assessment**: Identify potential issues and mitigation strategies

---

# DETAILED INVESTIGATION RESULTS

## Part 1: Current Ontology Dependencies

### System-Wide Impact Assessment
- **Total Python files scanned**: 7,943
- **Files with ontology dependencies**: 123+
- **Core pipeline files**: 5 critical components
- **Ontology-dependent tests**: 15+ test files

### Critical Pipeline Components

#### 1. core/structured_extractor.py (LLM Extraction)
**Current Evidence extraction dependencies**:
- `diagnostic_type`: 8 direct references 
- Van Evera types (`hoop`, `smoking_gun`, etc.): 40+ references
- Evidence node type patterns: Hardcoded in Pydantic schemas

**Impact of Evidence restructuring**: ❌ **BREAKING CHANGE**
- All LLM prompts expect diagnostic_type on Evidence nodes
- Pydantic validation schemas would need complete rewrite
- JSON schema validation would break

#### 2. core/analyze.py (Main Analysis Pipeline)  
**Current dependencies**:
- `Evidence.*type`: 58+ references throughout analysis logic
- Van Evera diagnostic types: 43+ references
- Evidence properties access: 8+ direct property lookups

**Impact of Evidence restructuring**: ❌ **BREAKING CHANGE**
- Evidence processing logic assumes diagnostic_type is available
- HTML generation expects diagnostic classification on Evidence
- Van Evera testing engine queries Evidence.type directly

#### 3. Event Temporal Field Usage
**Current temporal fields in use**:
- `date`: 65+ references across codebase (most used)
- `timestamp`: 10+ references
- `start_date`/`end_date`: 10+ references combined
- `is_point_in_time`: 3+ references

**Impact of temporal consolidation**: ❌ **BREAKING CHANGE**
- All temporal processing logic would need rewrite
- Date parsing and formatting throughout analysis pipeline
- Timeline generation in HTML output

### Inference System Dependencies
**Current Bayesian/inference patterns**:
- `prior_probability`/`posterior_probability`: 15+ references
- `updates_probability` edge: Used in structured extraction
- Van Evera testing engine: Direct probability calculations

**Impact of N-ary inference nodes**: ❌ **BREAKING CHANGE**
- Current system uses direct Evidence→Hypothesis edges
- Bayesian updating logic would need fundamental redesign
- Van Evera engine assumes binary evidence-hypothesis relationships

## Part 2: Detailed Breaking Change Analysis

### Change Category 1: Evidence/Diagnostic Restructuring

#### Current Architecture:
```json
{
  "type": "Evidence",
  "properties": {
    "description": "Napoleon invaded Russia in winter",
    "type": "smoking_gun",           # ← Diagnostic type here
    "certainty": 0.9
  }
}
```

#### Proposed Architecture:
```json
{
  "type": "Evidence", 
  "properties": {
    "description": "Napoleon invaded Russia in winter",
    "data_modality": "documentary"   # ← New field
  }
}

{
  "type": "Inferential_Test",
  "properties": {
    "type": "smoking_gun",           # ← Moved here
    "evidence_id": "evidence_1",
    "hypothesis_id": "hypothesis_1"
  }
}
```

#### Files Requiring Changes:
1. **core/structured_extractor.py**
   - Rewrite all Evidence extraction schemas
   - Add new Inferential_Test extraction logic
   - Update LLM prompts to extract separate objects
   - **Estimated effort**: 3-5 days

2. **core/analyze.py** 
   - Rewrite evidence processing logic (58+ references)
   - Update Van Evera diagnostic lookup logic
   - Modify HTML generation data preparation
   - **Estimated effort**: 5-7 days

3. **core/html_generator.py**
   - Update evidence display logic
   - Modify diagnostic type rendering
   - Revise evidence categorization
   - **Estimated effort**: 2-3 days

4. **core/van_evera_testing_engine.py**
   - Completely rewrite diagnostic test logic
   - Update Bayesian calculation methods
   - **Estimated effort**: 3-4 days

### Change Category 2: Event Temporal Consolidation

#### Current Architecture:
```json
{
  "type": "Event",
  "properties": {
    "timestamp": "2024-01-15T10:30:00",
    "date": "January 15, 1812", 
    "start_date": "1812-01-15",
    "end_date": "1812-12-15",
    "is_point_in_time": false
  }
}
```

#### Proposed Architecture:
```json
{
  "type": "Event",
  "properties": {
    "time": {
      "earliest": "1812-01-15T00:00:00Z",
      "latest": "1812-12-15T23:59:59Z", 
      "precision": "month",
      "is_interval": true
    }
  }
}
```

#### Files Requiring Changes:
1. **core/structured_extractor.py**
   - Rewrite Event temporal extraction schemas
   - Update LLM prompts for new time object structure
   - **Estimated effort**: 2-3 days

2. **core/analyze.py**
   - Update all temporal processing logic (65+ date references)
   - Rewrite timeline generation
   - Modify temporal analysis functions
   - **Estimated effort**: 4-5 days

3. **HTML generation & visualization**
   - Update timeline rendering 
   - Modify date display formatting
   - **Estimated effort**: 1-2 days

### Change Category 3: N-ary Inference System

#### Current Architecture:
```json
{
  "type": "infers",
  "domain": ["Evidence"], 
  "range": ["Hypothesis"],
  "properties": {
    "certainty": 0.8
  }
}
```

#### Proposed Architecture:
```json
{
  "type": "Inference",
  "properties": {
    "rule": "bayesian_updating",
    "prior": 0.3,
    "posterior": 0.7, 
    "bayes_factor": 2.33
  }
}

// With edges: Evidence → uses_premise → Inference → concludes → Hypothesis
```

#### Impact Assessment:
- **Current inference logic**: Assumes binary Evidence→Hypothesis relationships
- **New requirement**: Multiple Evidence + Rule → single Inference → Hypothesis  
- **Complexity**: Fundamental architectural change to reasoning system

#### Files Requiring Changes:
1. **Bayesian processing logic**: Complete rewrite required
2. **Evidence evaluation**: Must aggregate multiple premises
3. **Van Evera testing**: Must work with N-ary inference nodes
4. **HTML generation**: Must display complex inference chains
5. **All test files**: Must be updated for new inference structure

**Estimated effort**: 10-15 days (most complex change)

## Part 3: Implementation Effort Assessment

### Easy Changes (1-3 days each)
✅ **Additive node types**:
- Add Citation, Source_Excerpt, Prediction nodes
- Add temporal edges (before, after, simultaneous_with)  
- Add hypothesis competition (competes_with, rival_to)
- Add Case scaffolding

**Why easy**: Don't break existing functionality, purely additive

### Medium Changes (1-2 weeks each)
⚠️ **Mechanism granularity**:
- Add Mechanism_Step nodes
- Update mechanism processing logic
- Modify HTML generation for stepwise display

⚠️ **Enhanced provenance**:
- Add paragraph/character offset tracking
- Update source citation structure
- Modify evidence traceability

### Hard Changes (2-4 weeks each)
❌ **Evidence/diagnostic restructuring**:
- **Files affected**: 20+ core files
- **Breaking changes**: LLM extraction, analysis pipeline, HTML generation
- **Testing required**: Complete regression testing
- **Risk**: High - could break entire pipeline

❌ **Event temporal consolidation**:
- **Files affected**: 15+ files with temporal logic
- **Breaking changes**: All date processing, timeline generation
- **Data migration**: All existing Event nodes need conversion
- **Testing required**: Temporal analysis validation

❌ **N-ary inference system**:
- **Files affected**: Entire Bayesian reasoning pipeline
- **Breaking changes**: Fundamental architecture change
- **New complexity**: Multi-premise reasoning logic
- **Testing required**: Complete inference validation

## Part 4: Risk Assessment

### High-Risk Changes
1. **Evidence restructuring**: Could break LLM extraction entirely
2. **N-ary inference**: Fundamental change to reasoning architecture  
3. **Temporal consolidation**: Complex data migration required

### Mitigation Strategies
1. **Phased approach**: Implement easy changes first, validate before hard changes
2. **Dual schemas**: Maintain backward compatibility during transition
3. **Feature flags**: Allow switching between old/new structures
4. **Comprehensive testing**: Full regression test suite required

### Success Criteria
- All existing functionality preserved
- New capabilities working as designed
- Performance maintained or improved
- Complete test coverage for new features

## Part 5: Recommended Implementation Approach

### Phase 1: Low-Risk Additions (Week 1)
✅ Add temporal edges (before, after, simultaneous_with)
✅ Add hypothesis competition (competes_with, rival_to) 
✅ Add Citation and Source_Excerpt nodes
✅ Add Case scaffolding
✅ Update ontology_manager with new queries

**Effort**: 3-5 days
**Risk**: Very low - purely additive
**Value**: Addresses #1 critical gap (temporal ordering)

### Phase 2: Medium-Risk Enhancements (Week 2-3)
⚠️ Add Prediction nodes with P(E|H) calculations
⚠️ Enhance mechanism granularity with steps
⚠️ Implement paragraph-based provenance tracking
⚠️ Add quality control metadata

**Effort**: 1-2 weeks  
**Risk**: Medium - may require schema adjustments
**Value**: Significant automation improvements

### Phase 3: High-Risk Restructuring (Month 2-3) 
❌ Evidence/diagnostic type separation (IF needed)
❌ Event temporal field consolidation (IF needed)  
❌ N-ary inference system (IF needed)

**Effort**: 4-6 weeks
**Risk**: High - could break existing functionality
**Value**: Production-ready automation capabilities

## Part 6: Cost-Benefit Analysis

### Option A: Minimal Changes (Phase 1 only)
**Cost**: 1 week
**Benefit**: Fixes critical temporal gap, adds hypothesis competition
**Risk**: Very low
**Outcome**: 80% of academic benefits for 20% of effort

### Option B: Moderate Changes (Phase 1 + 2)  
**Cost**: 3-4 weeks
**Benefit**: Significant automation improvements, better provenance
**Risk**: Medium
**Outcome**: Production-ready system with automation capabilities

### Option C: Full Restructuring (All phases)
**Cost**: 2-3 months
**Benefit**: Complete production-ready system with all expert recommendations
**Risk**: High - potential to break existing system
**Outcome**: Industry-grade process tracing platform

## Final Recommendations

### Recommended Approach: **Option A + Selective Option B**
1. **Implement Phase 1 immediately** - Low risk, high value
2. **Evaluate results after Phase 1** - Assess system improvement
3. **Implement selected Phase 2 features** - Based on actual needs
4. **Avoid Phase 3 unless absolutely necessary** - Cost/benefit not favorable

### Key Insights
- **80/20 rule applies**: Most benefits achievable with minimal changes
- **Temporal edges alone** address the #1 critical gap both reviews identified  
- **Evidence restructuring** has questionable ROI given implementation cost
- **Current system is academically sound** - focus on targeted improvements

### Next Steps
1. Review this analysis with stakeholders
2. Get approval for Phase 1 implementation
3. Plan Phase 1 development timeline
4. Begin with temporal edge implementation

---

# PART 7: PERFECT LLM EXTRACTION INVESTIGATION

## Implementation Requirements & Design Questions

**ASSUMPTIONS**: 
- LLM is capable of expert-level academic reasoning with proper prompting
- LLM can provide consistent step-by-step reasoning explanations  
- LLM can generate precise numerical probability assessments (P(E|H) = 0.85, not ranges)
- LLM can systematically generate and evaluate alternative interpretations

**FOCUS**: Implementation design and technical architecture, not capability validation

### A. REASONING CHAIN ARCHITECTURE

#### Uncertainty 1: How to Structure Multi-Step Reasoning
**Question**: How do we capture complex reasoning chains like: 
`Evidence1 + Evidence2 → Intermediate Inference → Evidence3 validates → Final Hypothesis`

**Current approach**: Direct Evidence → Hypothesis edges
**Proposed approach**: Evidence → Reasoning_Step → Reasoning_Step → Hypothesis

**Investigation needed**:
- How many reasoning steps are typical in process tracing?
- Should reasoning steps be linear or tree-structured? 
- How does LLM naturally structure its reasoning process?
- What granularity of reasoning steps is useful vs overwhelming?

#### Uncertainty 2: N-ary Evidence Combination Logic  
**Question**: How does LLM combine multiple pieces of evidence?

**Example reasoning**: "Evidence A (Napoleon invaded) + Evidence B (winter was harsh) + Evidence C (no winter supplies) → Combined conclusion (overconfidence hypothesis)"

**Investigation needed**:
- How does LLM weight different evidence when combining?
- Should combination be additive, multiplicative, or more complex?
- How to handle contradictory evidence in combinations?
- What logical rules should govern evidence combination?

#### Uncertainty 3: Confidence Propagation Through Chains
**Question**: How does confidence flow through multi-step reasoning?

**Example**: 
- Evidence A (confidence 0.8) → 
- Intermediate inference (confidence ?) → 
- Final hypothesis (confidence ?)

**Investigation needed**:
- Mathematical models for confidence propagation
- How uncertainty compounds through reasoning chains
- Whether to use Bayesian, fuzzy logic, or other approaches
- How LLM naturally assesses confidence at each step

### B. LLM EXTRACTION CAPABILITIES

#### Requirement 4: LLM Reasoning Transparency Implementation
**Assumption**: Current LLMs can reliably explain their reasoning process with proper prompting.

**Target capability**: Detailed step-by-step reasoning with alternatives considered
**Expected output**: "I classified this as 'smoking gun' because winter retreat is sufficient evidence for overconfidence hypothesis, regardless of other factors. Alternative interpretations considered: strategic withdrawal (rejected due to lack of planning evidence), weather-forced retreat (rejected due to timing inconsistencies)."

**Implementation needed**:
- Design optimal prompting strategies for consistent multi-step reasoning
- Create templates for LLM reasoning explanation formats
- Implement confidence assessment prompting strategies  
- Develop validation frameworks for reasoning explanation quality

#### Requirement 5: Mathematical Precision Implementation
**Requirement**: LLM must provide precise numerical probability assessments.

**Target capability**: 
- LLM says "Evidence strongly supports hypothesis" 
- System requires P(E|H) = 0.85 (precise single value, not ranges or qualitative)

**Implementation needed**:
- Design prompts that elicit consistent numerical probability assignments
- Implement validation for LLM numerical vs qualitative consistency 
- Ensure appropriate precision levels (2 decimal places: 0.85, 0.73, etc.)
- Validate LLM Bayes factor calculations for mathematical accuracy

#### Requirement 6: Alternative Interpretation Generation Implementation
**Assumption**: LLM can reliably generate and evaluate alternative interpretations with proper prompting.

**Target capability**: 
"Evidence X could support Hypothesis A (smoking gun test) but alternatively could support Hypothesis B (hoop test) if we assume Y. I chose A because Z."

**Implementation needed**:
- Design prompts that consistently elicit plausible alternatives (target: 2-3 alternatives per evidence)
- Create structured formats for alternative evaluation and rejection reasoning
- Implement prompting strategies to minimize confirmation bias
- Develop templates for comparative analysis between interpretations

### C. ONTOLOGY SCHEMA DESIGN

#### Uncertainty 7: Granularity of Reasoning Steps
**Question**: What level of granularity is optimal for reasoning steps?

**Too coarse**: "Evidence → Hypothesis" (current)
**Too fine**: "Evidence → micro-inference1 → micro-inference2 → ... → Hypothesis"
**Optimal**: ?

**Investigation needed**:
- Analyze expert historian reasoning patterns for typical granularity
- Test different granularity levels with LLM extraction quality
- Determine cognitive load implications for human review
- Balance between detail and usability

#### Uncertainty 8: Schema Structure for Complex Reasoning
**Question**: How to structure ontology for multi-evidence, multi-step reasoning?

**Options considered**:
1. **Inference nodes**: Evidence → Inference_Node → Hypothesis
2. **Reasoning chains**: Evidence → Reasoning_Step1 → Reasoning_Step2 → Hypothesis  
3. **Argument structures**: Evidence → Argument → Claim → Hypothesis
4. **Hybrid approaches**: Combination of above

**Investigation needed**:
- Prototype different schema approaches
- Test LLM extraction quality with each approach
- Assess human reviewability of each structure
- Determine computational complexity implications

#### Requirement 9: Confidence and Uncertainty Representation
**Decision**: Use single precise confidence scores for all LLM assessments.

**Required format**: confidence = 0.8 (single numerical value, 2 decimal precision)
**Rejected alternatives**: Confidence intervals, multiple uncertainty types, qualitative uncertainty

**Implementation needed**:
- Design prompts that consistently elicit single confidence scores
- Implement calibration validation (ensure 80% confidence correlates with 80% accuracy)
- Create confidence score validation ranges (0.0-1.0, required precision)
- Develop human review interfaces optimized for single confidence scores

### D. IMPLEMENTATION ARCHITECTURE

#### Uncertainty 10: LLM Processing Pipeline Design
**Question**: How should the LLM extraction pipeline work for complex reasoning?

**Current pipeline**: Text → LLM → Evidence/Hypothesis nodes → Analysis
**Proposed options**:
1. **Single-pass extraction**: Text → LLM extracts everything at once
2. **Multi-pass extraction**: Text → Extract basic elements → LLM adds reasoning → LLM adds confidence
3. **Iterative refinement**: Text → Initial extraction → LLM self-critique → Refined extraction
4. **Ensemble approaches**: Multiple LLM passes with consensus building

**Investigation needed**:
- Test extraction quality with different pipeline approaches
- Assess computational cost and time requirements
- Determine error propagation characteristics
- Validate consistency across pipeline approaches

#### Uncertainty 11: Human-LLM Interaction Model
**Question**: How should humans interact with LLM academic reasoning?

**Options**:
1. **Full automation**: LLM produces complete reasoning, human reviews
2. **Interactive reasoning**: Human guides LLM reasoning process step-by-step
3. **Collaborative reasoning**: LLM proposes, human modifies, iterative refinement
4. **Quality assurance**: LLM reasoning with human validation checkpoints

**Investigation needed**:
- Test user experience with different interaction models
- Assess accuracy improvements from human involvement
- Determine optimal balance between automation and human expertise
- Validate workflow efficiency for academic research

#### Uncertainty 12: Validation and Quality Control
**Question**: How do we validate that LLM academic reasoning is sound?

**Validation needs**:
- Reasoning chain logical consistency
- Mathematical calculation accuracy
- Historical/domain knowledge accuracy
- Bias detection and mitigation
- Confidence calibration validation

**Investigation needed**:
- Develop validation metrics for reasoning quality
- Create test suites for LLM academic performance
- Design bias detection and correction mechanisms
- Establish benchmarks against human expert performance

### E. TECHNICAL FEASIBILITY

#### Uncertainty 13: Current LLM Limitations
**Question**: What are the hard limits of current LLM capabilities for academic reasoning?

**Known challenges**:
- Hallucination in historical facts
- Inconsistency across similar problems
- Overconfidence in incorrect judgments
- Limited working memory for complex reasoning chains

**Investigation needed**:
- Systematic testing of LLM academic reasoning capabilities
- Identify failure modes and mitigation strategies
- Determine which aspects require human oversight
- Assess improvement potential with prompt engineering

#### Uncertainty 14: Scalability and Performance
**Question**: How will complex reasoning extraction scale with document size and complexity?

**Concerns**:
- Token limits for complex reasoning chains
- Processing time for multi-step extraction
- Memory requirements for large reasoning graphs
- Cost implications of detailed LLM reasoning

**Investigation needed**:
- Benchmark performance with complex historical documents
- Test scalability limits of reasoning chain extraction
- Assess cost-benefit of different reasoning granularities
- Determine optimal chunking strategies for large documents

#### Uncertainty 15: Integration with Existing System
**Question**: How do we migrate from current simple extraction to complex reasoning chains?

**Migration challenges**:
- Backward compatibility with existing graphs
- Data migration for enhanced reasoning structures  
- User interface updates for reasoning chain display
- Testing and validation during transition

**Investigation needed**:
- Design migration pathway with minimal disruption
- Create dual-mode operation during transition
- Develop testing protocols for migration validation
- Plan user training for enhanced reasoning features

## Investigation Methodology

### Phase 1: LLM Implementation Design (Week 1-2)
1. **Reasoning transparency prompting**: Design prompts for step-by-step reasoning extraction
2. **Confidence calibration implementation**: Create precise numerical confidence elicitation
3. **Alternative generation prompting**: Design prompts for alternative interpretation generation
4. **Mathematical precision validation**: Ensure consistent numerical probability assignments

### Phase 2: Schema Prototyping (Week 2-3)
1. **Design reasoning chain schemas**: Test different ontological structures
2. **Prototype extraction pipelines**: Test single-pass vs multi-pass approaches
3. **Validate human reviewability**: Ensure complex reasoning is comprehensible
4. **Assess computational requirements**: Determine performance implications

### Phase 3: Integration Planning (Week 3-4)  
1. **Migration strategy design**: Plan transition from simple to complex extraction
2. **Quality control framework**: Develop validation and testing approaches
3. **User experience design**: Plan interfaces for reasoning chain interaction
4. **Risk mitigation planning**: Identify and address potential failure modes

## Success Criteria for Investigation

### Technical Success Criteria
- [ ] LLM provides consistent, logical reasoning chains with proper prompting
- [ ] Confidence assessments are precise numerical values (0.85, not ranges)
- [ ] Mathematical calculations (P(E|H), Bayes factors) are precise and justified
- [ ] Alternative interpretations are generated and systematically evaluated
- [ ] Schema successfully represents complex multi-step reasoning chains

### Academic Success Criteria  
- [ ] Reasoning chains match expert historian thinking patterns
- [ ] Domain knowledge accuracy is maintained or improved
- [ ] Bias detection and mitigation is effective
- [ ] Academic transparency and auditability is achieved
- [ ] Research workflow efficiency is maintained or improved

### Implementation Success Criteria
- [ ] Migration pathway preserves existing functionality
- [ ] Performance remains acceptable with complex reasoning
- [ ] Human-LLM interaction is intuitive and efficient
- [ ] Quality control and validation systems are robust
- [ ] System reliability meets academic research standards

## Next Immediate Actions

1. **Design LLM reasoning capability tests** (Week 1 Priority)
2. **Create reasoning chain schema prototypes** (Week 1 Priority)  
3. **Test LLM mathematical precision and consistency** (Week 1 Priority)
4. **Prototype multi-step reasoning extraction** (Week 2 Priority)
5. **Validate reasoning chain comprehensibility with users** (Week 2 Priority)

---
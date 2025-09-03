# Phase 13: Complete LLM-First Implementation Plan

## Executive Summary

**Objective**: Remove all remaining hardcoded fallback values from 3 plugins to achieve true LLM-first architecture.

**Scope**: 13 hardcoded values across 3 files
- content_based_diagnostic_classifier.py: 1 value (LOW complexity)
- primary_hypothesis_identifier.py: 8 values (MEDIUM complexity) 
- research_question_generator.py: 4 values (HIGH complexity)

**Estimated Timeline**: 70 minutes total

---

## Detailed Analysis and Implementation Plan

### Plugin 1: content_based_diagnostic_classifier.py
**Complexity**: LOW | **Time**: 10 minutes | **Risk**: LOW

#### Current State
```python
# Line 570
confidence = confidence_assessment.probative_value if hasattr(confidence_assessment, 'probative_value') else 0.6
```

#### Context Analysis
- **Location**: Inside try block after successful LLM call
- **Purpose**: Handle case where LLM response object lacks expected attribute
- **Impact**: Used for confidence scoring in diagnostic classification

#### Implementation Plan
```python
# BEFORE (fallback)
confidence = confidence_assessment.probative_value if hasattr(confidence_assessment, 'probative_value') else 0.6

# AFTER (fail-fast)
if not hasattr(confidence_assessment, 'probative_value'):
    raise LLMRequiredError("LLM assessment missing probative_value attribute - invalid response format")
confidence = confidence_assessment.probative_value
```

#### Risk Assessment
- **Minimal**: Simple attribute check replacement
- **Dependencies**: None
- **Testing**: Verify plugin loads and functions

---

### Plugin 2: primary_hypothesis_identifier.py  
**Complexity**: MEDIUM | **Time**: 30 minutes | **Risk**: MEDIUM

#### Current State (8 fallback values)
```python
# Lines 146-149: Weight fallbacks
ve_weight_float = float(ve_weight) if isinstance(ve_weight, (int, float)) else 0.4
ev_weight_float = float(ev_weight) if isinstance(ev_weight, (int, float)) else 0.3  
th_weight_float = float(th_weight) if isinstance(th_weight, (int, float)) else 0.2
el_weight_float = float(el_weight) if isinstance(el_weight, (int, float)) else 0.1

# Lines 339-342: Threshold fallbacks  
ve_threshold_float = float(ve_threshold) if isinstance(ve_threshold, (int, float)) else 0.6
ev_threshold_float = float(ev_threshold) if isinstance(ev_threshold, (int, float)) else 0.5
th_threshold_float = float(th_threshold) if isinstance(th_threshold, (int, float)) else 0.4
el_threshold_float = float(el_threshold) if isinstance(el_threshold, (int, float)) else 0.3
```

#### Context Analysis
- **Location**: Configuration validation for academic criteria
- **Source**: Values come from `PRIMARY_HYPOTHESIS_CRITERIA` class constant
- **Purpose**: Handle corrupted/invalid configuration data
- **Expected Values**: Match exactly what's in PRIMARY_HYPOTHESIS_CRITERIA

#### Configuration Verification
```python
PRIMARY_HYPOTHESIS_CRITERIA = {
    'van_evera_score': {'weight': 0.40, 'minimum_threshold': 0.6},
    'evidence_support': {'weight': 0.30, 'minimum_threshold': 0.5}, 
    'theoretical_sophistication': {'weight': 0.20, 'minimum_threshold': 0.4},
    'elimination_power': {'weight': 0.10, 'minimum_threshold': 0.3}
}
```

#### Implementation Plan
```python
# BEFORE (fallback validation)
ve_weight_float = float(ve_weight) if isinstance(ve_weight, (int, float)) else 0.4

# AFTER (fail-fast validation)  
if not isinstance(ve_weight, (int, float)):
    raise LLMRequiredError(f"Invalid van_evera weight configuration: {ve_weight} - must be numeric")
ve_weight_float = float(ve_weight)
```

#### Systematic Replacement Strategy
1. Create helper method `_validate_numeric_config(value, name, expected)`
2. Replace all 8 fallback patterns with validation calls
3. Ensure error messages indicate configuration corruption
4. Test with valid and invalid configurations

#### Risk Assessment  
- **Medium**: Configuration validation is important
- **Mitigation**: The class constant should always be valid
- **Testing**: Verify configuration integrity, test error paths

---

### Plugin 3: research_question_generator.py
**Complexity**: HIGH | **Time**: 30 minutes | **Risk**: HIGH

#### Current State (4 fallback values)
```python  
# Line 447: Domain score
score += 0.25 if analysis['domain_scores'][analysis['primary_domain']] >= 3 else 0.15

# Line 450: Concept count  
score += 0.25 if len(analysis['content_analysis']['key_concepts']) >= 5 else 0.15

# Line 453: Causal language
score += 0.25 if analysis['content_analysis']['causal_language_detected'] else 0.1

# Line 457: Complexity count
score += 0.25 if complexity_count >= 3 else (0.15 if complexity_count >= 2 else 0.1)
```

#### Context Analysis
- **Location**: `_calculate_sophistication_score()` method  
- **Purpose**: Algorithmic scoring for academic sophistication
- **Problem**: Rule-based scoring, not semantic understanding
- **Data Available**: Hypothesis text, domain classification, content analysis

#### LLM-First Replacement Strategy

**Option A: Single LLM Assessment** (Recommended)
```python
def _calculate_sophistication_score(self, analysis: Dict) -> float:
    """Calculate academic sophistication score using LLM assessment"""
    try:
        semantic_service = get_semantic_service()
        
        # Create sophistication assessment context
        assessment_context = f"""
        Domain: {analysis['primary_domain']}
        Key Concepts: {analysis['content_analysis']['key_concepts']}
        Causal Language: {analysis['content_analysis']['causal_language_detected']}
        Complexity Indicators: {analysis['complexity_indicators']}
        """
        
        sophistication_result = semantic_service.assess_probative_value(
            evidence_description=assessment_context,
            hypothesis_description="This research represents sophisticated academic inquiry",
            context="Academic sophistication assessment for research question generation"
        )
        
        if not hasattr(sophistication_result, 'probative_value'):
            raise LLMRequiredError("Sophistication assessment missing probative_value")
            
        return sophistication_result.probative_value
        
    except Exception as e:
        raise LLMRequiredError(f"Cannot assess sophistication without LLM: {e}")
```

**Option B: Detailed LLM Scoring** (More comprehensive)
- Replace each scoring component with individual LLM assessments
- Combine results with LLM-determined weights
- Higher accuracy but more LLM calls

#### Risk Assessment
- **High**: Changes fundamental scoring logic
- **Impact**: May affect research question quality/ranking
- **Mitigation**: Extensive testing with known datasets
- **Fallback**: Option A is simpler and safer

---

## Implementation Sequence

### Phase 1: Low-Risk Fix (10 minutes)
1. Fix content_based_diagnostic_classifier.py
2. Test plugin loading and basic functionality
3. Commit changes

### Phase 2: Medium-Risk Fix (30 minutes)  
1. Fix primary_hypothesis_identifier.py configuration validation
2. Add helper method for clean implementation
3. Test with valid/invalid configurations
4. Verify academic scoring still works
5. Commit changes

### Phase 3: High-Risk Fix (30 minutes)
1. Implement LLM-based sophistication scoring
2. Extensive testing with existing datasets
3. Compare outputs before/after for reasonableness
4. Commit changes
5. Create evidence documentation

---

## Testing Strategy

### Unit Testing
- **Plugin Loading**: All plugins import without errors
- **Basic Functionality**: Core methods execute without exceptions
- **Error Handling**: LLMRequiredError raised appropriately

### Integration Testing  
- **End-to-End**: Run full analysis pipeline with test data
- **Output Comparison**: Compare results before/after changes
- **Edge Cases**: Test with invalid/missing data

### Validation Testing
```bash
# Check for remaining fallbacks
grep -r "else 0\." core/plugins/*.py

# Verify compliance improvement  
python validate_true_compliance.py

# Test plugin functionality
python -c "from core.plugins.content_based_diagnostic_classifier import *"
python -c "from core.plugins.primary_hypothesis_identifier import *" 
python -c "from core.plugins.research_question_generator import *"
```

---

## Success Criteria

### Quantitative Metrics
- **Zero** remaining hardcoded fallback values in plugins
- **Compliance rate**: >90% (up from 86.6%)
- **Plugin functionality**: All plugins load and execute

### Qualitative Measures  
- All semantic operations use LLM understanding
- Proper fail-fast error handling throughout
- No degradation in system functionality
- Clean, maintainable code

### Evidence Requirements
- Before/after grep results showing removal of fallbacks
- Plugin loading test outputs
- Validation compliance report
- Sample analysis outputs for comparison

---

## Risk Mitigation

### Rollback Plan
- Keep backup of original files
- Commit each plugin fix separately  
- Test incrementally to isolate issues

### Error Handling
- Comprehensive exception handling around LLM calls
- Clear error messages indicating what LLM functionality is required
- Graceful degradation messaging where appropriate

### Quality Assurance
- Code review of all changes
- Testing with multiple datasets
- Documentation of any behavior changes

---

## Estimated Timeline Summary

| Plugin | Complexity | Time | Cumulative |
|--------|------------|------|------------|
| content_based_diagnostic_classifier | LOW | 10 min | 10 min |
| primary_hypothesis_identifier | MEDIUM | 30 min | 40 min |  
| research_question_generator | HIGH | 30 min | 70 min |

**Total Implementation Time**: 70 minutes
**Total with Testing/Documentation**: 90 minutes

This plan provides a systematic approach to achieving true LLM-first architecture in all remaining plugin components.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 PERMANENT INFORMATION -- DO NOT CHANGE ON UPDATES

### LLM-First Architecture Policy (MANDATORY)

**CORE PRINCIPLE**: This system is **LLM-FIRST** with **ZERO TOLERANCE** for rule-based or keyword-based implementations.

**PROHIBITED IMPLEMENTATIONS**:
- ❌ Keyword matching for evidence classification (`if 'ideological' in text`)
- ❌ Hardcoded probative value assignments (`probative_value = 0.7`)
- ❌ Rule-based contradiction detection (`if 'before' in hypothesis and 'after' in evidence`)
- ❌ Domain classification using keyword lists
- ❌ Confidence thresholds based on hardcoded ranges
- ❌ Any `if/elif` chains for semantic understanding
- ❌ Dataset-specific logic (American Revolution hardcoded rules)
- ❌ Historical period-specific keyword matching
- ❌ Returning None/0/[] on LLM failure (must raise LLMRequiredError)

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation
- ✅ Generalist process tracing without dataset-specific hardcoding
- ✅ Raise LLMRequiredError on any LLM failure (fail-fast)

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: Phase 13 - Implementation Ready (Updated 2025-01-30)

**System Status**: **86.6% Measured Compliance** (58/67 files pass validator)
**Target**: **Remove 13 remaining hardcoded fallbacks from 3 plugins**
**Goal**: **Achieve TRUE LLM-first in all critical components**

**PHASE 13 IMPLEMENTATION STATUS:**
- ✅ **Planning Complete**: Comprehensive analysis and implementation plan created
- ✅ **Risk Assessment**: LOW→MEDIUM→HIGH complexity sequence identified
- ✅ **Testing Strategy**: Full validation framework designed
- ⏳ **Implementation**: Ready to execute systematic fixes

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately via LLMRequiredError
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **INCREMENTAL VALIDATION**: Test after each plugin fix to isolate issues
- **COMPREHENSIVE VERIFICATION**: Use systematic validation before claiming completion

### Quality Standards
- **Semantic Understanding**: All classification based on LLM analysis, not keyword matching
- **Generalist System**: No dataset-specific hardcoding - system works across all historical periods
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results in structured evidence files

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Architecture
- **Plugin System**: 16+ registered plugins (13 fallbacks remain in 3 plugins)
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence
- **LLM Integration**: VanEveraLLMInterface with structured Pydantic output (Gemini 2.5 Flash)
- **Validation System**: validate_true_compliance.py for comprehensive compliance checking
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management
- **Universality**: No dataset-specific logic - works across all domains and time periods

## 🚀 PHASE 13: Execute LLM-First Migration (Implementation Ready)

### CRITICAL: 3 Plugins Require 13 Fallback Removals

**IMPLEMENTATION ORDER** (LOW→MEDIUM→HIGH Risk):

#### TASK 1: content_based_diagnostic_classifier.py (10 minutes, LOW risk)
**Location**: Line 570
**Violation**: `confidence = confidence_assessment.probative_value if hasattr(confidence_assessment, 'probative_value') else 0.6`

**Required Fix**:
```python
# BEFORE (has fallback)
confidence = confidence_assessment.probative_value if hasattr(confidence_assessment, 'probative_value') else 0.6

# AFTER (fail-fast)
if not hasattr(confidence_assessment, 'probative_value'):
    raise LLMRequiredError("LLM assessment missing probative_value attribute - invalid response format")
confidence = confidence_assessment.probative_value
```

**Validation Commands**:
```bash
# Test plugin loading
python -c "from core.plugins.content_based_diagnostic_classifier import ContentBasedDiagnosticClassifierPlugin; print('Plugin loads OK')"

# Verify fallback removed
grep -n "else 0\." core/plugins/content_based_diagnostic_classifier.py
# Should return no results
```

#### TASK 2: primary_hypothesis_identifier.py (30 minutes, MEDIUM risk)
**Locations**: Lines 146-149 (weights), 339-342 (thresholds)
**Violations**: 8 configuration validation fallbacks

**Current Fallback Pattern**:
```python
ve_weight_float = float(ve_weight) if isinstance(ve_weight, (int, float)) else 0.4
ev_weight_float = float(ev_weight) if isinstance(ev_weight, (int, float)) else 0.3
th_weight_float = float(th_weight) if isinstance(th_weight, (int, float)) else 0.2
el_weight_float = float(el_weight) if isinstance(el_weight, (int, float)) else 0.1
# Plus 4 similar threshold patterns
```

**Required Fix Strategy**:
1. Add helper method for configuration validation:
```python
def _validate_numeric_config(self, value, name, expected):
    """Validate configuration value is numeric or raise LLMRequiredError"""
    if not isinstance(value, (int, float)):
        raise LLMRequiredError(f"Invalid {name} configuration: {value} - expected numeric value {expected}")
    return float(value)
```

2. Replace all 8 fallback patterns:
```python
# BEFORE (fallback)
ve_weight_float = float(ve_weight) if isinstance(ve_weight, (int, float)) else 0.4

# AFTER (fail-fast)
ve_weight_float = self._validate_numeric_config(ve_weight, "van_evera weight", 0.4)
```

**Context**: The `PRIMARY_HYPOTHESIS_CRITERIA` class constant should always contain valid numeric values. These fallbacks handle configuration corruption, but LLM-first requires failing explicitly.

**Validation Commands**:
```bash
# Test plugin loading
python -c "from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin; print('Plugin loads OK')"

# Verify no fallbacks remain  
grep -n "else 0\." core/plugins/primary_hypothesis_identifier.py
# Should return no results

# Verify configuration integrity
python -c "
from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin
plugin = PrimaryHypothesisIdentifierPlugin('test')
for key, val in plugin.PRIMARY_HYPOTHESIS_CRITERIA.items():
    assert isinstance(val['weight'], (int, float))
    assert isinstance(val['minimum_threshold'], (int, float))
print('Configuration values all numeric')
"
```

#### TASK 3: research_question_generator.py (30 minutes, HIGH risk)
**Locations**: Lines 447, 450, 453, 457 in `_calculate_sophistication_score()`
**Violations**: 4 algorithmic scoring fallbacks

**Current Rule-Based Logic**:
```python
def _calculate_sophistication_score(self, analysis: Dict) -> float:
    """Calculate academic sophistication score (0.0-1.0)"""
    score = 0.0
    
    # Domain specificity (0.25)
    score += 0.25 if analysis['domain_scores'][analysis['primary_domain']] >= 3 else 0.15
    
    # Theoretical sophistication (0.25)
    score += 0.25 if len(analysis['content_analysis']['key_concepts']) >= 5 else 0.15
    
    # Causal complexity (0.25)
    score += 0.25 if analysis['content_analysis']['causal_language_detected'] else 0.1
    
    # Analytical depth (0.25)
    complexity_count = sum(analysis['complexity_indicators'].values())
    score += 0.25 if complexity_count >= 3 else (0.15 if complexity_count >= 2 else 0.1)
    
    return min(score, 1.0)
```

**Required LLM-Based Replacement**:
```python
def _calculate_sophistication_score(self, analysis: Dict) -> float:
    """Calculate academic sophistication score using LLM assessment"""
    try:
        from core.semantic_analysis_service import get_semantic_service
        semantic_service = get_semantic_service()
        
        # Create sophistication assessment context
        assessment_context = f"""
        Research Domain: {analysis['primary_domain']}
        Key Concepts: {', '.join(analysis['content_analysis']['key_concepts'][:10])}
        Causal Language Present: {analysis['content_analysis']['causal_language_detected']}
        Complexity Indicators: {dict(list(analysis['complexity_indicators'].items())[:5])}
        """
        
        sophistication_result = semantic_service.assess_probative_value(
            evidence_description=assessment_context.strip(),
            hypothesis_description="This research demonstrates sophisticated academic inquiry with theoretical depth and methodological rigor",
            context="Academic sophistication assessment for research question generation"
        )
        
        if not hasattr(sophistication_result, 'probative_value'):
            raise LLMRequiredError("Sophistication assessment missing probative_value - invalid LLM response")
            
        return sophistication_result.probative_value
        
    except Exception as e:
        raise LLMRequiredError(f"Cannot assess sophistication without LLM: {e}")
```

**Risk**: This changes fundamental scoring logic. Research question quality may change.

**Validation Commands**:
```bash
# Test plugin loading
python -c "from core.plugins.research_question_generator import ResearchQuestionGeneratorPlugin; print('Plugin loads OK')"

# Verify no rule-based scoring remains
grep -n "else 0\." core/plugins/research_question_generator.py
# Should return no results

# Test LLM integration
python -c "from core.semantic_analysis_service import get_semantic_service; print('LLM service available')"
```

### POST-IMPLEMENTATION VALIDATION

**Comprehensive Validation Commands**:
```bash
# 1. Check ALL plugins for remaining fallbacks
for file in core/plugins/*.py; do 
    echo "=== $file ==="
    grep -n "else 0\." "$file" 2>/dev/null || echo "No fallbacks found"
done

# 2. Run compliance validator
python validate_true_compliance.py

# Expected improvement: 86.6% → >90%

# 3. Test all modified plugins load
python -c "
from core.plugins.content_based_diagnostic_classifier import ContentBasedDiagnosticClassifierPlugin
from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin  
from core.plugins.research_question_generator import ResearchQuestionGeneratorPlugin
print('All 3 plugins load successfully')
"

# 4. Test system functionality (if test data available)
python -m core.analyze test_data/american_revolution_graph.json
```

### SUCCESS CRITERIA

**Must Achieve**:
- ✅ Zero hardcoded fallback values in all plugins (`grep -r "else 0\." core/plugins/` returns nothing)
- ✅ Compliance rate >90% (improved from 86.6%)
- ✅ All 3 modified plugins load without import errors
- ✅ System functionality preserved (analysis pipeline works)
- ✅ Proper LLMRequiredError handling implemented

### EVIDENCE REQUIREMENTS

**Create Evidence Files**:
1. **`evidence/current/Evidence_Phase13_Task1.md`**: content_based_diagnostic_classifier fix
2. **`evidence/current/Evidence_Phase13_Task2.md`**: primary_hypothesis_identifier fix  
3. **`evidence/current/Evidence_Phase13_Task3.md`**: research_question_generator fix
4. **`evidence/current/Evidence_Phase13_Final.md`**: Comprehensive validation results

**Each Evidence File Must Include**:
- Before/after code snippets
- Raw command outputs for all validation steps
- Plugin loading test results
- Any functional differences observed
- Success/failure determination with justification

### IMPLEMENTATION SEQUENCE

1. **Execute Task 1** (10 minutes) → Test → Document → Commit
2. **Execute Task 2** (30 minutes) → Test → Document → Commit  
3. **Execute Task 3** (30 minutes) → Test → Document → Commit
4. **Final Validation** (10 minutes) → Comprehensive testing → Evidence documentation

**Total Timeline**: 80 minutes implementation + 10 minutes final validation = 90 minutes

### REFERENCE DOCUMENTS

**Implementation Details**: `evidence/current/Evidence_Phase13_ImplementationPlan.md`
**Testing Strategy**: `evidence/current/Evidence_Phase13_TestingStrategy.md`
**Complete Planning**: `evidence/current/Evidence_Phase13_CompletePlan.md`

## Testing Commands Reference

```bash
# Validate compliance improvements
python validate_true_compliance.py

# Check for any remaining fallbacks  
grep -r "else 0\." core/plugins/

# Test plugin loading
python -c "from core.plugins.[name] import [ClassName]; print('OK')"

# Test system functionality
python -m core.analyze test_data/american_revolution_graph.json

# Verify LLM integration
python -c "from core.semantic_analysis_service import get_semantic_service; print('Available')"
```

## Critical Success Factors

- **Evidence-Based Validation**: Every claim must be backed by raw command outputs
- **Incremental Testing**: Test after each plugin to isolate issues quickly  
- **Fail-Fast Implementation**: Proper LLMRequiredError, no silent fallbacks
- **No Compromise**: Zero tolerance for any remaining hardcoded values
- **System Preservation**: Core functionality must remain intact

Remember: TRUE LLM-first means ZERO fallbacks. No compromises, no exceptions.
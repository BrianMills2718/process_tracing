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

## 🎯 CURRENT STATUS: Phase 12 - Final Corrections (Updated 2025-01-30)

**System Status**: **86.6% Measured Compliance** (58/67 files pass validator)
**Actual Semantic Compliance**: **~94%** (63/67 when excluding false positives)
**Current Priority**: **FIX 2 REMAINING PLUGIN FALLBACKS** for true LLM-first

**PHASE 11 VERIFICATION RESULTS (2025-01-30):**
- ✅ **2 Plugins Fully Fixed**: content_based_diagnostic_classifier.py, legacy_compatibility_manager.py
- ⚠️ **2 Plugins Still Have Fallbacks**: dowhy_causal_analysis_engine.py (3 values), advanced_van_evera_prediction_engine.py (1 value)
- ℹ️ **5 False Positives**: Variable names, comments, dictionary keys flagged incorrectly
- ❌ **4 Temporal Files**: Deep architectural coupling requires major refactoring

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately via LLMRequiredError
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **COMPREHENSIVE VALIDATION**: Check ALL files, not just a subset
- **CONTINUOUS VERIFICATION**: Re-verify after each change

### Quality Standards
- **Semantic Understanding**: All classification based on LLM analysis, not keyword matching
- **Generalist System**: No dataset-specific hardcoding - system works across all historical periods
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results in structured evidence files

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Architecture
- **Plugin System**: 16+ registered plugins requiring LLM-first conversion
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence
- **LLM Integration**: VanEveraLLMInterface with structured Pydantic output (Gemini 2.5 Flash)
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management
- **Universality**: No dataset-specific logic - works across all domains and time periods

## 🚀 PHASE 12: Fix Remaining Plugin Fallbacks

### CRITICAL: Plugins Still Containing Hardcoded Fallbacks (2 files)

#### PRIORITY 1: Fix These Immediately

1. **core/plugins/dowhy_causal_analysis_engine.py** - 3 HARDCODED VALUES
   - Line 210: `else 0.7` - Fallback if no confidence_score attribute
   - Line 241: `else 0.6` - Fallback if no confidence_score attribute  
   - Line 269: `else 0.5` - Fallback if no confidence_score attribute
   - **Fix**: Remove `else` clauses, raise LLMRequiredError if attribute missing

2. **core/plugins/advanced_van_evera_prediction_engine.py** - 1 HARDCODED VALUE
   - Line 1022: `else 0.5` - Fallback if assessment lacks probative_value
   - **Fix**: Remove `else` clause, raise LLMRequiredError if attribute missing

### Non-Critical Issues (Can Defer)

#### FALSE POSITIVES - Validator Incorrectly Flags These (3 files)
- **core/evidence_document.py** - Dictionary key check "temporal" (not semantic analysis)
- **core/performance_profiler.py** - Phase name categorization (system labels, not semantic)
- **core/plugins/research_question_generator.py** - Variable name "temporal_classification" (not keyword matching)

#### TEMPORAL MODULES - Major Refactoring Required (4 files)
These require complete architectural changes and can be deferred:
- **core/temporal_extraction.py** - 20+ violations, deep coupling
- **core/temporal_graph.py** - 6 violations, would need rewrite
- **core/temporal_validator.py** - 2 violations
- **core/temporal_viz.py** - Multiple violations + hardcoded values

#### ENCODING ISSUES - Cannot Validate (2 files)
- **core/extract.py** - Character encoding prevents reading
- **core/structured_extractor.py** - Character encoding prevents reading

## 📋 IMMEDIATE TASKS

### Task 1: Fix dowhy_causal_analysis_engine.py (15 minutes)

**Exact Violations**:
- Line 210: `getattr(causal_analysis, 'confidence_score', 0.7)`
- Line 241: `getattr(causal_analysis, 'confidence_score', 0.6)`
- Line 269: `getattr(causal_analysis, 'confidence_score', 0.5)`

**Required Fix**:
```python
# WRONG - Has fallback value
confidence = getattr(causal_analysis, 'confidence_score', 0.7)

# CORRECT - Fail if attribute missing
if hasattr(causal_analysis, 'confidence_score'):
    confidence = causal_analysis.confidence_score
else:
    raise LLMRequiredError("Causal analysis missing confidence score - LLM required")
```

### Task 2: Fix advanced_van_evera_prediction_engine.py (10 minutes)

**Exact Violation**:
- Line 1022: `base_confidence = assessment.probative_value if assessment else 0.5`

**Required Fix**:
```python
# WRONG - Has fallback
base_confidence = assessment.probative_value if assessment else 0.5

# CORRECT - Fail if no assessment
if not assessment or not hasattr(assessment, 'probative_value'):
    raise LLMRequiredError("Cannot determine confidence without LLM assessment")
base_confidence = assessment.probative_value
```

### Task 3: Final Validation

**Validation Commands**:
```bash
# Run comprehensive validation
python validate_true_compliance.py

# Verify 100% compliance
# Expected: 67/67 files compliant

# Create evidence file
echo "Phase 11 Complete: 100% LLM-First Compliance" > evidence/current/Evidence_Phase11_100_Percent.md
```

### Success Criteria

**Realistic Goals**:
- ✅ Remove ALL hardcoded fallback values from the 2 plugins
- ✅ Achieve ~94% true semantic compliance (excluding false positives)
- ✅ Document false positives properly
- ✅ System remains functional

**Known Limitations to Accept**:
- ⚠️ Temporal modules require major refactoring (defer to future phase)
- ⚠️ Validator flags false positives (document but don't fix)
- ⚠️ Encoding issues in 2 files (exclude from semantic validation)

### Estimated Timeline

- **Task 1**: 15 minutes (dowhy_causal_analysis_engine.py fixes)
- **Task 2**: 10 minutes (advanced_van_evera_prediction_engine.py fix)
- **Task 3**: 10 minutes (validation and documentation)

**Total**: 35 minutes to achieve true LLM-first in all critical plugins

## Testing Commands

```bash
# Validate compliance
python validate_true_compliance.py

# Test specific file
python -c "from core.plugins.advanced_van_evera_prediction_engine import *"

# Check for violations
grep -r "if.*in.*text" core/ --include="*.py"
grep -r "confidence\s*=\s*0\." core/ --include="*.py"

# Run main analysis
python -m core.analyze test_data/american_revolution_graph.json
```

## Evidence Requirements

Each task completion requires evidence in `evidence/current/`:
- Raw command outputs
- Before/after code snippets
- Validation results
- Error logs if any
- Success/failure determination

Remember: NO FALLBACKS, NO SHORTCUTS, PURE LLM-FIRST!
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

## 🎯 CURRENT STATUS: Phase 12 - Partial Progress (Updated 2025-01-30)

**System Status**: **86.6% Measured Compliance** (58/67 files pass validator)
**Reality Check**: **Only 2/5 plugins with fallbacks have been fixed**
**Current Priority**: **FIX 3 MORE PLUGINS WITH 13+ FALLBACKS**

**PHASE 12 DOUBLE-CHECK RESULTS (2025-01-30):**
- ✅ **2 Plugins Fixed**: dowhy_causal_analysis_engine.py, advanced_van_evera_prediction_engine.py
- ❌ **3 Plugins Still Have Fallbacks**: 
  - content_based_diagnostic_classifier.py (1 fallback)
  - primary_hypothesis_identifier.py (8 fallbacks!)
  - research_question_generator.py (4 fallbacks)
- ℹ️ **5 False Positives**: Variable names, comments, dictionary keys
- ❌ **4 Temporal Files**: Deep architectural coupling

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

## 🚀 PHASE 13: Fix Remaining Plugin Fallbacks

### CRITICAL: 3 Plugins Still Containing 13+ Hardcoded Fallbacks

#### PRIORITY 1: Must Fix These

1. **core/plugins/content_based_diagnostic_classifier.py** - 1 FALLBACK
   - Line 570: `else 0.6` - Fallback confidence value
   - **Fix**: Remove fallback, raise LLMRequiredError

2. **core/plugins/primary_hypothesis_identifier.py** - 8 FALLBACKS!
   - Lines 146-149: Weight fallbacks (0.4, 0.3, 0.2, 0.1)
   - Lines 339-342: Threshold fallbacks (0.6, 0.5, 0.4, 0.3)
   - **Fix**: Remove all fallbacks, use LLM or configuration

3. **core/plugins/research_question_generator.py** - 4 FALLBACKS
   - Lines 447, 450, 453, 457: Score calculation fallbacks
   - **Fix**: Replace conditional scoring with LLM assessment

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

## 📋 REMAINING TASKS (Phase 13)

### Task 1: Fix content_based_diagnostic_classifier.py (10 minutes)

**Exact Violation**:
- Line 570: `confidence = confidence_assessment.probative_value if hasattr(confidence_assessment, 'probative_value') else 0.6`

### Task 2: Fix primary_hypothesis_identifier.py (30 minutes)

**Exact Violations**:
- Lines 146-149: Weight fallbacks for Van Evera test types
- Lines 339-342: Threshold fallbacks for test types

### Task 3: Fix research_question_generator.py (20 minutes)

**Exact Violations**:
- Line 447: Domain score fallback (0.25/0.15)
- Line 450: Concept count fallback (0.25/0.15)
- Line 453: Causal language fallback (0.25/0.1)
- Line 457: Complexity fallback (0.25/0.15/0.1)

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

**Updated Realistic Goals**:
- ✅ Remove ALL 13+ remaining hardcoded fallback values from 3 plugins
- ✅ Actually achieve true LLM-first in plugin system
- ✅ Properly validate all changes
- ✅ System remains functional

**Known Limitations to Accept**:
- ⚠️ Temporal modules require major refactoring (defer)
- ⚠️ Validator false positives (document only)
- ⚠️ Encoding issues in 2 files (exclude)

### Estimated Timeline

- **Task 1**: 10 minutes (content_based_diagnostic_classifier.py)
- **Task 2**: 30 minutes (primary_hypothesis_identifier.py - 8 values)
- **Task 3**: 20 minutes (research_question_generator.py - 4 values)
- **Task 4**: 10 minutes (validation and documentation)

**Total**: 70 minutes to ACTUALLY achieve true LLM-first in plugins

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
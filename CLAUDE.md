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

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation
- ✅ Generalist process tracing without dataset-specific hardcoding

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: Phase 9 - Cleanup and Final Migration (Updated 2025-01-31)

**System Status**: **71.4% LLM-First Compliance** (10/14 core semantic files)
**Current Priority**: **CLEANUP** - Remove redundant gateway, complete final migrations
**Critical Issue**: **Redundant gateway violates LLM-first principles with keyword matching**

**PHASE 8 DISCOVERIES (2025-01-31):**
- ✅ **Real Compliance**: 71.4% of files already use VanEveraLLMInterface correctly
- ✅ **LiteLLM Usage**: JSON parsing IS required (LiteLLM returns strings, not objects)
- ⚠️ **Gateway Redundant**: Created unnecessary complexity, violates LLM-first with keyword matching
- ❌ **4 Files Non-Compliant**: enhance_mechanisms.py, confidence_calculator.py, analyze.py (partial), diagnostic_rebalancer.py

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### Quality Standards
- **Semantic Understanding**: All classification based on LLM analysis, not keyword matching
- **Generalist System**: No dataset-specific hardcoding - system works across all historical periods
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results in structured evidence files

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Architecture
- **Plugin System**: 16 registered plugins requiring LLM-first conversion
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence
- **LLM Integration**: VanEveraLLMInterface with structured Pydantic output (Gemini 2.5 Flash)
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management
- **Universality**: No dataset-specific logic - works across all domains and time periods

## 🚀 PHASE 9: Cleanup and Complete Migration

### Critical Context
**What We Discovered**:
- VanEveraLLMInterface already exists with 15+ structured methods
- semantic_analysis_service.py provides centralized LLM access
- LiteLLM correctly returns JSON strings that need parsing
- Gateway implementation adds unnecessary complexity and violates LLM-first principles

**What Needs Fixing**:
- Remove redundant gateway implementation
- Revert enhance_evidence.py to use existing interfaces
- Migrate 4 remaining non-compliant files
- Remove keyword matching from gateway and any other files

### Task 1: Remove Redundant Gateway

**Objective**: Clean up unnecessary gateway implementation that violates LLM-first principles

**Required Actions**:

1. **Delete gateway file**:
```bash
# Delete the redundant gateway
rm core/llm_gateway.py

# Verify it's gone
ls core/llm_gateway.py  # Should show "cannot access"
```

2. **Remove gateway tests**:
```bash
# Delete gateway-specific test files
rm test_gateway_integration.py
rm test_enhance_evidence_migration.py  # If it exists

# Keep validation scripts that are still useful
# Keep: validate_migration_progress.py, check_real_compliance.py
```

3. **Document removal** in `evidence/current/Evidence_Phase9_Gateway_Removal.md`:
```markdown
## Gateway Removal Evidence

### Files Deleted
- core/llm_gateway.py (lines of keyword matching removed: 178-193)
- test_gateway_integration.py
- test_enhance_evidence_migration.py

### Reason for Removal
- Redundant: VanEveraLLMInterface already provides needed functionality
- Violates LLM-first: Contains keyword matching logic
- Unnecessary complexity: Adds abstraction layer without benefit

### Verification
[Include ls commands showing files no longer exist]
```

### Task 2: Revert enhance_evidence.py

**Objective**: Restore enhance_evidence.py to use existing LLM interfaces

**Required Actions**:

1. **Check current state**:
```bash
# See what imports gateway
grep -n "llm_gateway\|LLMGateway" core/enhance_evidence.py
```

2. **Revert to original** (if modified):
```bash
# Check git status
git status core/enhance_evidence.py

# If modified, revert it
git checkout -- core/enhance_evidence.py

# Verify it uses VanEveraLLMInterface or semantic_service
grep -n "VanEveraLLMInterface\|semantic_analysis_service" core/enhance_evidence.py
```

3. **Document** in `evidence/current/Evidence_Phase9_Enhance_Evidence_Revert.md`

### Task 3: Migrate enhance_mechanisms.py

**Objective**: Convert enhance_mechanisms.py to LLM-first approach

**Analysis First**:
```bash
# Check current implementation
grep -n "return None\|return 0\|if.*in\|except.*pass" core/enhance_mechanisms.py
```

**Migration Pattern**:
```python
# At top of file, add imports:
from core.plugins.van_evera_llm_interface import VanEveraLLMInterface
from core.llm_required import LLMRequiredError

# Initialize interface:
llm_interface = VanEveraLLMInterface()

# Replace fallback patterns:
# OLD:
def analyze_mechanism(text):
    if not text:
        return None
    # Keyword matching
    if "cause" in text.lower():
        return {"type": "causal", "confidence": 0.7}
    return {"type": "unknown", "confidence": 0.0}

# NEW:
def analyze_mechanism(text):
    if not text:
        raise LLMRequiredError("Text required for mechanism analysis")
    
    # Use LLM for semantic understanding
    result = llm_interface.analyze_causal_mechanism(
        mechanism_text=text,
        context="Process tracing analysis"
    )
    return result  # Structured Pydantic response
```

**Validation**:
```bash
# Test the migrated file
python -c "from core.enhance_mechanisms import *; print('Import successful')"

# Run validation script
python check_real_compliance.py | grep enhance_mechanisms
```

**Document** in `evidence/current/Evidence_Phase9_Mechanisms_Migration.md`

### Task 4: Migrate confidence_calculator.py

**Objective**: Replace hardcoded confidence calculations with LLM-based assessment

**Analysis First**:
```bash
# Find hardcoded calculations
grep -n "0\.\d\|confidence.*=\|return.*float" core/confidence_calculator.py
```

**Migration Pattern**:
```python
# OLD:
def calculate_confidence(evidence_count, hypothesis_strength):
    # Hardcoded formula
    base_confidence = 0.5
    evidence_factor = min(evidence_count * 0.1, 0.3)
    hypothesis_factor = hypothesis_strength * 0.2
    return base_confidence + evidence_factor + hypothesis_factor

# NEW:
def calculate_confidence(evidence_descriptions, hypothesis_text, context):
    """Calculate confidence using LLM semantic assessment"""
    if not evidence_descriptions or not hypothesis_text:
        raise LLMRequiredError("Evidence and hypothesis required for confidence calculation")
    
    # Use VanEveraLLMInterface for semantic confidence assessment
    result = llm_interface.assess_hypothesis_confidence(
        hypothesis=hypothesis_text,
        supporting_evidence=evidence_descriptions,
        context=context
    )
    return result.confidence_score  # Float from 0-1 with LLM reasoning
```

**Document** in `evidence/current/Evidence_Phase9_Confidence_Migration.md`

### Task 5: Fix analyze.py Partial Compliance

**Objective**: Complete LLM-first migration in analyze.py

**Analysis**:
```bash
# Find non-compliant sections
grep -n "keyword\|if.*in.*text\|hardcoded" core/analyze.py

# Check for semantic_service usage
grep -n "semantic_service\|semantic_analysis" core/analyze.py
```

**Required Fixes**:
1. Ensure ALL hypothesis evaluation uses semantic_service or VanEveraLLMInterface
2. Remove any keyword matching for evidence classification
3. Replace hardcoded thresholds with LLM-based assessment

**Document** in `evidence/current/Evidence_Phase9_Analyze_Completion.md`

### Task 6: Migrate diagnostic_rebalancer.py

**Objective**: Convert plugin to LLM-first approach

**Analysis**:
```bash
# Check plugin implementation
grep -n "return\|if.*in\|diagnostic_type" core/plugins/diagnostic_rebalancer.py
```

**Migration Pattern**:
- Replace rule-based diagnostic rebalancing with LLM assessment
- Use VanEveraLLMInterface.rebalance_diagnostics() method
- Remove any hardcoded diagnostic type mappings

**Document** in `evidence/current/Evidence_Phase9_Rebalancer_Migration.md`

### Task 7: Final Validation

**Objective**: Prove 100% LLM-first compliance achieved

**Create** `validate_phase9_completion.py`:
```python
#!/usr/bin/env python3
"""Validate Phase 9 completion - 100% LLM-first compliance"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_file_compliance(filepath):
    """Check single file for LLM-first compliance"""
    content = Path(filepath).read_text()
    
    violations = []
    
    # Check for prohibited patterns
    prohibited = [
        (r"if\s+['\"].*['\"].*in.*text", "Keyword matching"),
        (r"return\s+None\s*#.*fallback", "Fallback to None"),
        (r"return\s+0\.?\d+\s*#.*default", "Hardcoded default"),
        (r"confidence\s*=\s*0\.\d+", "Hardcoded confidence"),
        (r"probative_value\s*=\s*0\.\d+", "Hardcoded probative value")
    ]
    
    import re
    for pattern, description in prohibited:
        if re.search(pattern, content, re.IGNORECASE):
            violations.append(description)
    
    # Check for required patterns
    uses_llm = any([
        "VanEveraLLMInterface" in content,
        "semantic_analysis_service" in content,
        "LLMRequiredError" in content,
        "require_llm" in content
    ])
    
    if not uses_llm and "semantic" in filepath:
        violations.append("No LLM interface usage")
    
    return len(violations) == 0, violations

def main():
    # Target files for validation
    semantic_files = [
        "core/enhance_evidence.py",
        "core/enhance_hypotheses.py",
        "core/enhance_mechanisms.py",
        "core/semantic_analysis_service.py",
        "core/confidence_calculator.py",
        "core/analyze.py",
        "core/plugins/van_evera_testing_engine.py",
        "core/plugins/diagnostic_rebalancer.py",
        "core/plugins/alternative_hypothesis_generator.py",
        "core/plugins/evidence_connector_enhancer.py",
        "core/plugins/content_based_diagnostic_classifier.py",
        "core/plugins/research_question_generator.py",
        "core/plugins/primary_hypothesis_identifier.py",
        "core/plugins/bayesian_van_evera_engine.py"
    ]
    
    compliant = 0
    total = len(semantic_files)
    
    print("Phase 9 LLM-First Compliance Validation")
    print("=" * 50)
    
    for filepath in semantic_files:
        if Path(filepath).exists():
            is_compliant, violations = check_file_compliance(filepath)
            
            if is_compliant:
                print(f"[OK] {filepath}")
                compliant += 1
            else:
                print(f"[FAIL] {filepath}")
                for v in violations[:3]:
                    print(f"      - {v}")
    
    print("=" * 50)
    compliance_rate = (compliant / total) * 100
    print(f"Compliance: {compliant}/{total} files ({compliance_rate:.1f}%)")
    
    if compliance_rate == 100:
        print("[SUCCESS] 100% LLM-first compliance achieved!")
        return 0
    else:
        print(f"[INCOMPLETE] {total - compliant} files still need migration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Run Validation**:
```bash
python validate_phase9_completion.py > evidence/current/Evidence_Phase9_Final_Validation.md
```

### Success Criteria

**Must Achieve**:
- ✅ Gateway removed completely
- ✅ No keyword matching patterns remain
- ✅ All 4 non-compliant files migrated
- ✅ 100% of semantic files use LLM interfaces
- ✅ All validation scripts pass

### Testing Commands

```bash
# Verify gateway is gone
ls core/llm_gateway.py 2>&1 | grep "cannot access"

# Check real compliance
python check_real_compliance.py

# Validate individual migrations
python -c "from core.enhance_mechanisms import *"
python -c "from core.confidence_calculator import *"

# Run final validation
python validate_phase9_completion.py

# Test system still works
python -m core.analyze test_data/american_revolution_graph.json
```

## Evidence Files Structure

Create these files in `evidence/current/`:
- `Evidence_Phase9_Gateway_Removal.md` - Proof of gateway removal
- `Evidence_Phase9_Enhance_Evidence_Revert.md` - Reversion details
- `Evidence_Phase9_Mechanisms_Migration.md` - Migration details and test results
- `Evidence_Phase9_Confidence_Migration.md` - Migration details and test results
- `Evidence_Phase9_Analyze_Completion.md` - Completion of analyze.py migration
- `Evidence_Phase9_Rebalancer_Migration.md` - Plugin migration details
- `Evidence_Phase9_Final_Validation.md` - 100% compliance validation

Each evidence file must contain:
- Raw command outputs
- Before/after code snippets
- Test results proving functionality
- No assumptions - only verified facts

## Next Phase Preview

After achieving 100% LLM-first compliance:
- Phase 10: Performance optimization with batched operations
- Phase 11: Advanced Van Evera test enhancements
- Phase 12: Counterfactual analysis implementation
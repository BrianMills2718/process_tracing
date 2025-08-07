# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: EVIDENCE-BASED TASK COMPLETION (Updated 2025-01-07)

**System Status**: **Functional** - Van Evera pipeline operational with 67.9-71.5% academic compliance  
**Current Priority**: **Complete Remaining Evidence-Based Tasks** - Address verified technical issues with systematic validation  
**Academic Quality**: **67.9-71.5% Van Evera Compliance** achieved through working pipeline  

**COMPLETED TASKS (2025-01-07)**:
- ✅ **Pydantic Enum Bug**: Fixed `evidence_quality="LOW"` to `"low"` in van_evera_llm_interface.py:286
- ✅ **System Validation**: Confirmed Van Evera pipeline executes successfully
- ✅ **Fictional Problem Identification**: Verified claimed "timeout" and "hanging" issues don't exist

**Evidence-Based Status Assessment**:
- ✅ **Core Van Evera Pipeline**: Working (9-step workflow executes successfully)
- ✅ **Plugin Architecture**: All 16 plugins functional and registered
- ✅ **LLM Integration**: Gemini 2.5 Flash operational with structured output
- ✅ **Q/H1/H2/H3 Structure**: Working (Q_H1 identification successful)
- ✅ **LLM Interface**: Pydantic validation now passes correctly
- ❌ **CLAUDE.md Violations**: 6 test files violate "no unnecessary files" requirement
- ❌ **Type Checking**: mypy available but not run (119 errors found)
- ❌ **Performance Assessment**: No proper baseline established

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **VERIFY BEFORE FIXING**: Every problem must be reproduced through direct testing before attempting solutions
- **REAL PROBLEMS ONLY**: Address only technical issues confirmed through evidence-based validation
- **MINIMAL INTERVENTION**: Use surgical fixes for confirmed bugs, avoid system overhauls
- **EVIDENCE-BASED VALIDATION**: All claims backed by measurable, reproducible testing
- **PROCESS COMPLIANCE**: Follow CLAUDE.md requirements exactly (no unnecessary files)

### Quality Standards
- **Academic Functionality**: Maintain ≥60% Van Evera compliance (currently 67.9-71.5%)
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results

## Project Overview

This is an LLM-enhanced Process Tracing Toolkit with **complete Van Evera plugin architecture** achieving **67.9-71.5% academic quality**. Core functionality is operational. Remaining tasks address verified violations and establish proper evidence-based baselines.

## 🚨 REMAINING TASKS: EVIDENCE-BASED COMPLETION

### **TASK 1: Remove Process Violation Files (CRITICAL)**

**Problem**: 6 test files exist that violate CLAUDE.md "no unnecessary files" requirement
**Evidence**: Files confirmed to exist and violate stated principles

**Files to Remove**:
```bash
rm test_critical_fix_validation.py
rm test_diagnostic_rebalancer.py  
rm test_phase4_complete.py
rm test_phase4_final.py
rm test_phase4_validation.py
rm test_q_h_structure_implementation.py
```

**Validation Requirements**:
1. Verify files exist before removal:
```bash
ls test_*.py 2>/dev/null && echo "Violation files found" || echo "No violations"
```

2. Remove confirmed violation files (DO NOT remove legitimate system tests)

3. Validate successful removal:
```bash
ls test_*.py 2>/dev/null && echo "FAILED: Files still exist" || echo "SUCCESS: Violations cleaned"
```

**Success Criteria**: 
- All 6 specified violation files removed
- No legitimate test files (if any exist) are removed
- Process compliance achieved

### **TASK 2: Complete Type Checking Validation (CRITICAL)**

**Problem**: mypy is available but wasn't used, found 119 type errors across 17 files
**Evidence**: `python -c "import mypy"` succeeds, but type checking was skipped

**Required Actions**:

1. **Confirm mypy availability**:
```bash
python -c "import mypy" 2>/dev/null && echo "mypy IS available" || echo "mypy not available"
```

2. **Run comprehensive type checking**:
```bash
python -m mypy core/plugins/ --ignore-missing-imports --show-error-codes
```

3. **Document type checking results**:
- Record number of errors found
- Identify which files have type issues
- Note: DO NOT attempt to fix all 119 errors - just document the baseline

4. **Validate core imports still work**:
```bash
python -c "
try:
    from core.plugins.van_evera_llm_interface import VanEveraLLMInterface
    from core.plugins.van_evera_workflow import execute_van_evera_analysis
    print('SUCCESS: Core imports functional despite type issues')
except ImportError as e:
    print(f'CRITICAL: Import failure - {e}')
"
```

**Success Criteria**:
- mypy executed successfully on core/plugins/
- Type error baseline documented (number and scope)
- Core functionality confirmed to remain operational
- Evidence-based assessment of type checking status completed

### **TASK 3: Establish Real Performance Baseline (IMPORTANT)**

**Problem**: Previous performance claims were scientifically invalid (impossible timing measurements)
**Evidence**: Claimed 0.01-0.02s execution times are 100-250x faster than physically possible

**Required Methodology**:

1. **Proper Performance Testing**:
```python
import time
import sys
from core.plugins.van_evera_workflow import execute_van_evera_analysis

def measure_performance():
    """Establish evidence-based performance baseline with proper methodology"""
    
    # Test datasets with known characteristics
    test_cases = [
        {
            'name': 'Small',
            'nodes': 5,  # 2 hypotheses + 3 evidence
            'edges': 6   # 3 support edges + 3 addresses_research_question
        },
        {
            'name': 'Medium', 
            'nodes': 20,  # 5 hypotheses + 15 evidence
            'edges': 25   # 15 support + 5 addresses_research_question + 5 alternatives
        },
        {
            'name': 'Large',
            'nodes': 50,  # 10 hypotheses + 40 evidence  
            'edges': 70   # 40 support + 10 addresses + 20 alternatives
        }
    ]
    
    results = []
    for case in test_cases:
        # Generate test data
        data = generate_test_data(case['nodes'], case['edges'])
        
        # Measure execution time with proper methodology
        start_time = time.perf_counter()
        try:
            result = execute_van_evera_analysis(data, f'baseline_{case["name"].lower()}')
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            academic_score = result.get('academic_quality_assessment', {}).get('overall_score', 0)
            
            results.append({
                'case': case['name'],
                'nodes': case['nodes'],
                'edges': case['edges'],
                'execution_time_seconds': round(execution_time, 3),
                'academic_score_percent': round(academic_score, 1),
                'success': bool(result.get('van_evera_analysis'))
            })
            
            print(f"{case['name']} Dataset ({case['nodes']} nodes, {case['edges']} edges):")
            print(f"  Execution Time: {execution_time:.3f}s")
            print(f"  Academic Score: {academic_score:.1f}%")
            print(f"  Success: {bool(result.get('van_evera_analysis'))}")
            
        except Exception as e:
            print(f"{case['name']} dataset failed: {e}")
            results.append({
                'case': case['name'],
                'error': str(e),
                'success': False
            })
    
    return results

# Execute performance baseline
print("=== EVIDENCE-BASED PERFORMANCE BASELINE ===")
baseline_results = measure_performance()
print("=== BASELINE ESTABLISHED ===")
```

2. **Document Realistic Performance Characteristics**:
- Record actual execution times (likely 2-5+ seconds for real datasets)
- Note memory usage patterns
- Document academic quality scores for different dataset sizes
- Establish what the system actually achieves vs. documentation claims

**Success Criteria**:
- Proper timing methodology used (time.perf_counter())
- Multiple dataset sizes tested 
- Realistic execution times documented (no impossible claims)
- Academic quality baseline established with evidence

### **TASK 4: Final Evidence-Based Validation**

**Complete systematic validation of all task completion**:

1. **Syntax Validation**:
```bash
python -m py_compile core/plugins/van_evera_llm_interface.py
echo "Syntax validation exit code: $?"
```

2. **Import Validation**: 
```bash
python -c "
try:
    from core.plugins.van_evera_llm_interface import VanEveraLLMInterface, create_llm_query_function
    from core.plugins.van_evera_workflow import execute_van_evera_analysis
    print('SUCCESS: All critical imports operational')
except ImportError as e:
    print(f'FAILED: Import error - {e}')
"
```

3. **Pydantic Validation Test**:
```bash
python -c "
from core.plugins.van_evera_llm_interface import VanEveraLLMInterface
from core.plugins.van_evera_llm_schemas import VanEveraPredictionEvaluation

interface = VanEveraLLMInterface()
try:
    fallback = interface._create_fallback_response(VanEveraPredictionEvaluation, 'validation test')
    print(f'SUCCESS: Pydantic enum fix confirmed - evidence_quality={fallback.evidence_quality}')
except Exception as e:
    print(f'FAILED: Pydantic validation error - {e}')
"
```

4. **Process Compliance Verification**:
```bash
# Verify no unnecessary files created
ls *.py | grep -E "^test_" | wc -l
echo "Test files remaining (should be 0 after Task 1): ^"

# Verify CLAUDE.md compliance
echo "Files created during task completion:"
git status --porcelain | grep "^??" || echo "No new files created"
```

## 🎯 SUCCESS CRITERIA (EVIDENCE-BASED)

### **Technical Success Criteria**
1. **Process Violations Resolved**: All 6 test files removed, no unnecessary files remain
2. **Type Checking Completed**: mypy executed, baseline documented (even with errors present)
3. **Performance Baseline Established**: Real execution times measured with proper methodology
4. **System Functionality Maintained**: Core Van Evera pipeline continues operating at 67.9-71.5% compliance

### **Quality Gates**
- ✅ Evidence-based problem identification (verify issues exist before fixing)
- ✅ Minimal intervention approach (surgical fixes, no unnecessary changes)
- ✅ Process compliance (CLAUDE.md requirements followed exactly)
- ✅ Systematic validation (all claims backed by direct testing)

### **What NOT to Implement**
- ❌ **Fictional Problem Solutions**: Do not implement fixes for unverified issues
- ❌ **System Overhauls**: Avoid large-scale changes when surgical fixes suffice
- ❌ **Unnecessary Files**: Create no test files, documentation files, or validation scripts
- ❌ **Unvalidated Claims**: Make no performance or quality assertions without direct evidence

## 🔧 IMPLEMENTATION GUIDANCE

### **Evidence-First Methodology**
1. **Before fixing anything**: Verify the problem exists through direct reproduction
2. **Before claiming success**: Test the fix through direct validation  
3. **Before asserting performance**: Measure actual behavior with proper methodology
4. **Before marking complete**: Ensure all evidence supports the completion claim

### **Minimal Intervention Principle**  
- Fix only confirmed issues with the smallest possible change
- Avoid refactoring or enhancement unless explicitly required
- Preserve existing functionality unless fixing breaks it
- Document evidence for every change made

### **Validation Standards**
- All assertions must be reproducible through provided commands
- Performance claims must be backed by actual measurements
- Success claims must be verified through direct testing
- Process compliance must be validated against specific CLAUDE.md requirements

**Repository Status**: Core functionality operational, specific violations identified, evidence-based tasks defined for systematic completion.
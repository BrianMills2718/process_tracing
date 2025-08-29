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

## 🎯 CURRENT STATUS: Phase 4B - Complete Batched Evaluation Integration (Updated 2025-01-29)

**System Status**: **Batched Infrastructure Built (60%)** - Integration incomplete
**Current Priority**: **COMPLETE INTEGRATION** - Make analyze.py actually use batched evaluation
**Critical Issue**: **Production code still uses old individual evaluation approach**

**PHASE 4A PARTIALLY COMPLETED (2025-01-29):**
- ✅ **Infrastructure Built**: BatchedHypothesisEvaluation schema and methods created
- ✅ **Test Validation**: Batched evaluation works perfectly in isolation
- ⚠️ **Integration Started**: Helper functions added but not used in main flow
- ❌ **Main Pipeline**: analyze.py still evaluates hypotheses individually
- ❌ **Dead Code**: Keyword matching code still exists (must be removed)

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

## 🚀 PHASE 4B: Complete Batched Evaluation Integration

### Critical Context
**What exists**: Batched evaluation infrastructure is built and tested
**What's missing**: Integration into main analyze.py pipeline
**Why it matters**: System still makes N individual LLM calls instead of 1 batched call

### Task 1: Discovery - Map All Evaluation Points

**Objective**: Find EVERY place where evidence evaluates against hypotheses

**Required Actions**:
1. Search analyze.py for these patterns:
```bash
grep -n "assess_probative_value\|get_comprehensive_analysis\|for.*hypothesis\|semantic_service\." core/analyze.py
```

2. Document each location in `evidence/current/Evidence_Phase4B_Discovery.md`:
```markdown
## Evaluation Points Found
- Line XXX: Function name, context, loop structure
- Line YYY: Error handling path, individual evaluation
```

3. Identify the MAIN evaluation loop (usually in a function like `execute_analysis()` or similar)

### Task 2: Remove Dead Code

**Objective**: Delete all keyword matching and unused code

**Required Actions**:
1. Delete the keyword matching function from semantic_analysis_service.py:
```python
# DELETE THIS ENTIRE FUNCTION (lines ~433-491):
def evaluate_relationship_lightweight(self, ...)
```

2. Search for and remove any calls to this function:
```bash
grep -r "evaluate_relationship_lightweight" core/
```

3. Document removal in `evidence/current/Evidence_Phase4B_Cleanup.md`

### Task 3: Implement Main Pipeline Integration

**Objective**: Replace individual evaluations with batched calls

**Step 3.1: Create Integration Function**
Add to analyze.py after imports:
```python
def batch_evaluate_evidence(evidence_node_data, hypothesis_nodes_data, G, semantic_service):
    """
    Evaluate one evidence against all hypotheses in a single LLM call.
    Updates graph edges with results.
    
    Args:
        evidence_node_data: Dict with evidence data including 'description'
        hypothesis_nodes_data: Dict of hypothesis_id -> hypothesis data
        G: NetworkX graph to update
        semantic_service: SemanticAnalysisService instance
    
    Returns:
        Dict of hypothesis_id -> evaluation results
    """
    # Format hypotheses for batch evaluation
    hypotheses = [
        {'id': hyp_id, 'text': hyp_data.get('description', '')}
        for hyp_id, hyp_data in hypothesis_nodes_data.items()
    ]
    
    # Get evidence description
    evidence_desc = evidence_node_data.get('description', '')
    evidence_id = evidence_node_data.get('id', 'unknown')
    
    # Call batched evaluation
    batch_result = semantic_service.evaluate_evidence_against_hypotheses_batch(
        evidence_id,
        evidence_desc,
        hypotheses,
        context="Main analysis pipeline"
    )
    
    # Process results and update graph
    results = {}
    for eval_result in batch_result.evaluations:
        hyp_id = eval_result.hypothesis_id
        
        # Create/update edge in graph
        G.add_edge(
            evidence_id,
            hyp_id,
            type='supports' if eval_result.relationship_type == 'supports' else 'challenges',
            properties={
                'probative_value': eval_result.confidence,
                'van_evera_diagnostic': eval_result.van_evera_diagnostic,
                'reasoning': eval_result.reasoning,
                'relationship_type': eval_result.relationship_type
            }
        )
        
        results[hyp_id] = eval_result
    
    return results
```

**Step 3.2: Find and Replace Main Loop**
Look for the main evidence-hypothesis evaluation loop (likely around lines 850-950):
```python
# REPLACE patterns like:
for evidence_id, evidence_node_data in evidence_nodes_data.items():
    for hypothesis_id in hypothesis_nodes_data:
        # Individual evaluation
        assessment = semantic_service.assess_probative_value(...)

# WITH:
for evidence_id, evidence_node_data in evidence_nodes_data.items():
    # Batch evaluation for all hypotheses
    evaluations = batch_evaluate_evidence(
        evidence_node_data, 
        hypothesis_nodes_data, 
        G, 
        semantic_service
    )
```

### Task 4: Update Error Handling Paths

**Objective**: Make error recovery use batched evaluation too

**Required Actions**:
1. Find all error handling sections (lines ~898-935, ~1139, etc.)
2. Replace individual calls with batch evaluation
3. Ensure fallback behavior is preserved

### Task 5: Validation

**Objective**: Prove the integration works correctly

**Test Script**: Create `validate_phase4b_integration.py`:
```python
#!/usr/bin/env python3
"""Validate Phase 4B integration is complete and working"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.analyze import run_analysis
from core.semantic_analysis_service import get_semantic_service

def validate_integration():
    # Load test data
    test_graph_path = Path("test_data/american_revolution_graph.json")
    if not test_graph_path.exists():
        print("[FAIL] Test data not found")
        return False
    
    # Clear cache and reset counters
    semantic_service = get_semantic_service()
    semantic_service.clear_cache()
    initial_calls = semantic_service._stats['llm_calls']
    
    # Run analysis
    results = run_analysis(str(test_graph_path))
    
    # Check call reduction
    total_calls = semantic_service._stats['llm_calls'] - initial_calls
    evidence_count = len([n for n in results['nodes'] if n['type'] == 'Evidence'])
    hypothesis_count = len([n for n in results['nodes'] if n['type'] == 'Hypothesis'])
    
    expected_max_calls = evidence_count  # One batch per evidence
    old_expected_calls = evidence_count * hypothesis_count  # Old approach
    
    print(f"Evidence nodes: {evidence_count}")
    print(f"Hypothesis nodes: {hypothesis_count}")
    print(f"LLM calls made: {total_calls}")
    print(f"Expected with batching: {expected_max_calls}")
    print(f"Expected without batching: {old_expected_calls}")
    
    if total_calls <= expected_max_calls * 1.5:  # Allow some overhead
        print("[OK] Batching is working!")
        return True
    else:
        print("[FAIL] Still making too many individual calls")
        return False

if __name__ == "__main__":
    success = validate_integration()
    sys.exit(0 if success else 1)
```

**Evidence Requirements**:
Document all results in `evidence/current/Evidence_Phase4B_Integration.md`:
- Before/after LLM call counts
- Performance metrics
- Quality comparison
- Any errors encountered

### Task 6: Final Cleanup

**Required Actions**:
1. Remove `get_comprehensive_analysis()` helper if no longer needed
2. Remove any TODO comments related to optimization
3. Update docstrings to reflect batched approach
4. Run lint and type checking

### Success Criteria

**Must demonstrate**:
- ✅ LLM calls reduced by 70%+ for multi-hypothesis scenarios
- ✅ All tests pass with identical or better quality
- ✅ No keyword matching code remains
- ✅ Main analyze.py uses batched evaluation throughout
- ✅ Error paths use batched evaluation

### Expected Outcomes

**Performance**: 
- 1 LLM call per evidence (instead of N calls for N hypotheses)
- 70-90% reduction in total LLM calls
- 50-70% faster execution

**Quality**:
- Better semantic coherence
- Inter-hypothesis relationship insights
- No degradation in accuracy

### Files to Modify

1. **core/analyze.py**: Main integration point
2. **core/semantic_analysis_service.py**: Remove dead code
3. **evidence/current/**: Create evidence files for each task

### Testing Commands

```bash
# Run validation
python validate_phase4b_integration.py

# Check for dead code
grep -r "evaluate_relationship_lightweight" core/

# Count LLM calls in test
python test_batched_evaluation.py

# Run main analysis with logging
python -m core.analyze test_data/american_revolution_graph.json
```

## Evidence Files Structure

Create these files in `evidence/current/`:
- `Evidence_Phase4B_Discovery.md` - Document all evaluation points found
- `Evidence_Phase4B_Cleanup.md` - Document dead code removal
- `Evidence_Phase4B_Integration.md` - Document integration results
- `Evidence_Phase4B_Validation.md` - Final validation results

Each evidence file must contain:
- Raw command outputs
- Before/after metrics
- Error logs if any
- Success/failure determination

## Next Phase Preview

After Phase 4B is complete, Phase 5 will focus on:
- Completing remaining 7 files migration to LLM-first
- Enhancing Van Evera test generation
- Adding counterfactual analysis
- Strengthening causal mechanism detection
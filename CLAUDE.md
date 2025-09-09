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
- ❌ Mixed LLM configurations (some calls to Gemini, others to different models)
- ❌ Direct API calls bypassing LiteLLM infrastructure

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation
- ✅ Generalist process tracing without dataset-specific hardcoding
- ✅ Raise LLMRequiredError on any LLM failure (fail-fast)
- ✅ Consistent LiteLLM routing for ALL LLM operations
- ✅ Single model configuration across entire system

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: Analysis Phase Performance Optimization Required (Updated 2025-01-09)

**System Status**: **EXTRACTION PIPELINE FULLY WORKING - ANALYSIS OPTIMIZATION NEEDED**  
**Latest Achievement**: **WSL migration successful + 100-150 LLM call bottleneck identified**  
**Current Priority**: **Implement batch processing optimization to reduce analysis time from 5-12 minutes to <3 minutes**

**PIPELINE STATUS**:
- ✅ **Extraction Phase**: **FULLY FUNCTIONAL** - French Revolution extracted in 132.93s (39 nodes, 31 edges)
- ✅ **WSL Migration**: **COMPLETE** - All Windows hangs eliminated, system stable
- ✅ **Analysis Phase**: **FUNCTIONAL BUT SLOW** - 100-150 sequential LLM calls = 5-12 minute runtime
- 🎯 **Performance Target**: **<3 minutes analysis time with 80% call reduction**

**OPTIMIZATION STRATEGY**:
- **Primary Bottleneck**: Evidence-hypothesis pair evaluation (80% of analysis time)
- **Solution**: Batch processing 5-10 pairs per LLM call (100 calls → 10-20 calls)
- **Quality Requirement**: Maintain LLM-first architecture with structured outputs
- **Risk Mitigation**: A/B testing and feature flags for rollback

## 📋 PHASE 21: LLM Call Batch Processing Optimization

### OBJECTIVE: Reduce analysis time from 5-12 minutes to <3 minutes via intelligent batch processing

**BACKGROUND**: WSL testing confirmed the system works but analysis phase makes 100-150 sequential LLM calls for evidence-hypothesis pair evaluation, creating a 5-12 minute bottleneck. Solution: batch 5-10 pairs per LLM call to achieve 80% call reduction.

### TASK 1: Create Baseline Performance Metrics (20 minutes)
**Purpose**: Establish quantitative baseline for optimization comparison

**File**: `core/analyze.py`
**Add at top after imports**:
```python
import time
import json
from datetime import datetime
from pathlib import Path

# PHASE 21: Performance measurement baseline
class PerformanceTracker:
    def __init__(self, output_dir=None):
        self.start_time = time.time()
        self.llm_calls = []
        self.total_calls = 0
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.metrics_file = self.output_dir / f"performance_baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    
    def log_llm_call(self, function_name, input_size, duration, success=True):
        self.total_calls += 1
        call_data = {
            "call_number": self.total_calls,
            "function": function_name,
            "input_size": input_size,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "elapsed_total": time.time() - self.start_time
        }
        self.llm_calls.append(call_data)
        print(f"[LLM-BASELINE-{self.total_calls}] {function_name} | {duration:.2f}s | {input_size} chars | Total: {self.total_calls}")
        
        # Save incrementally
        self.save_metrics()
    
    def save_metrics(self):
        metrics = {
            "baseline_run": True,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_runtime": time.time() - self.start_time,
            "total_llm_calls": self.total_calls,
            "average_call_time": sum(call["duration"] for call in self.llm_calls) / max(1, len(self.llm_calls)),
            "calls": self.llm_calls
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

# Initialize tracker globally
performance_tracker = None
```

**Instrument existing LLM calls**:
Find `refine_evidence_assessment_with_llm` calls and wrap them:
```python
# Before:
enhanced_assessment = refine_evidence_assessment_with_llm(evidence_text, hypothesis_text, ...)

# After:
start_time = time.time()
enhanced_assessment = refine_evidence_assessment_with_llm(evidence_text, hypothesis_text, ...)
if performance_tracker:
    performance_tracker.log_llm_call("refine_evidence_assessment", len(evidence_text), time.time() - start_time)
```

**Initialize in main function**:
```python
def analyze_graph(graph_path, output_dir=None, ...):
    global performance_tracker
    performance_tracker = PerformanceTracker(output_dir)
    # ... rest of function
```

### TASK 2: Design Batch Processing Schema (30 minutes)  
**Purpose**: Create Pydantic models for batch evidence-hypothesis evaluation

**File**: `core/batch_evaluation_models.py` (new file)
```python
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class DiagnosticType(str, Enum):
    SMOKING_GUN = "smoking_gun"
    HOOP_TEST = "hoop_test" 
    STRAW_IN_THE_WIND = "straw_in_the_wind"
    DOUBLY_DECISIVE = "doubly_decisive"

class EvidenceHypothesisPair(BaseModel):
    pair_id: str = Field(description="Unique identifier for this evidence-hypothesis pair")
    evidence_text: str = Field(description="Full text content of the evidence")
    evidence_id: str = Field(description="Evidence node ID")
    hypothesis_text: str = Field(description="Full text content of the hypothesis")  
    hypothesis_id: str = Field(description="Hypothesis node ID")

class ProbativeValueAssessment(BaseModel):
    pair_id: str = Field(description="Matches the pair_id from input")
    diagnostic_type: DiagnosticType = Field(description="Van Evera diagnostic test classification")
    probative_value: float = Field(ge=0.0, le=1.0, description="Probative strength (0.0-1.0)")
    confirmation_power: float = Field(ge=0.0, le=1.0, description="Power to confirm hypothesis if true")
    disconfirmation_power: float = Field(ge=0.0, le=1.0, description="Power to disconfirm hypothesis if false")
    reasoning: str = Field(description="Detailed reasoning for probative value assignment")
    relationship_strength: str = Field(description="Strength of evidence-hypothesis relationship")

class BatchEvaluationRequest(BaseModel):
    pairs: List[EvidenceHypothesisPair] = Field(description="List of evidence-hypothesis pairs to evaluate")
    context: str = Field(description="Broader context for the analysis")

class BatchEvaluationResponse(BaseModel):
    evaluations: List[ProbativeValueAssessment] = Field(description="Probative value assessments for each pair")
    batch_metadata: dict = Field(default_factory=dict, description="Metadata about batch processing")
```

### TASK 3: Implement Batch Processing Function (45 minutes)
**Purpose**: Create the core batch evaluation function

**File**: `core/batch_evidence_evaluator.py` (new file)
```python
from typing import List, Tuple
from .batch_evaluation_models import BatchEvaluationRequest, BatchEvaluationResponse, EvidenceHypothesisPair
from .llm_required import make_llm_call
import time

BATCH_EVALUATION_SYSTEM_PROMPT = """You are an expert in process tracing methodology and Van Evera diagnostic testing. You will evaluate evidence-hypothesis relationships in batches to determine probative values.

For each evidence-hypothesis pair, you must:
1. Classify the diagnostic test type (smoking gun, hoop test, straw in the wind, doubly decisive)
2. Calculate probative value (0.0-1.0) based on evidential strength
3. Assess confirmation and disconfirmation powers
4. Provide detailed reasoning

Maintain the same quality standards as individual evaluations while processing multiple pairs efficiently.

CRITICAL: Your response must be valid JSON matching the BatchEvaluationResponse schema exactly."""

def create_batch_evaluation_prompt(pairs: List[EvidenceHypothesisPair], context: str) -> str:
    prompt = f"""Context: {context}

Evaluate these evidence-hypothesis pairs for probative value:

"""
    for i, pair in enumerate(pairs, 1):
        prompt += f"""
PAIR {i} (ID: {pair.pair_id}):
Evidence ({pair.evidence_id}): {pair.evidence_text[:500]}{'...' if len(pair.evidence_text) > 500 else ''}
Hypothesis ({pair.hypothesis_id}): {pair.hypothesis_text[:300]}{'...' if len(pair.hypothesis_text) > 300 else ''}

---
"""
    
    prompt += """
Provide a JSON response with probative value assessments for all pairs following the BatchEvaluationResponse schema.
"""
    return prompt

def evaluate_evidence_hypothesis_batch(pairs: List[EvidenceHypothesisPair], context: str = "") -> BatchEvaluationResponse:
    """
    Evaluate multiple evidence-hypothesis pairs in a single LLM call
    
    Args:
        pairs: List of evidence-hypothesis pairs to evaluate
        context: Broader context for the analysis
        
    Returns:
        BatchEvaluationResponse with evaluations for each pair
    """
    if not pairs:
        return BatchEvaluationResponse(evaluations=[])
    
    # Create batch request
    batch_request = BatchEvaluationRequest(pairs=pairs, context=context)
    
    # Generate prompt
    prompt = create_batch_evaluation_prompt(pairs, context)
    
    # Make LLM call with structured output
    start_time = time.time()
    response = make_llm_call(
        system_instruction=BATCH_EVALUATION_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_schema=BatchEvaluationResponse,
        use_structured_output=True
    )
    duration = time.time() - start_time
    
    # Add batch metadata
    response.batch_metadata = {
        "batch_size": len(pairs),
        "processing_time": duration,
        "pairs_per_second": len(pairs) / duration if duration > 0 else 0
    }
    
    # Log performance
    if hasattr(__builtins__, 'performance_tracker') and performance_tracker:
        performance_tracker.log_llm_call(
            "batch_evidence_evaluation", 
            len(prompt), 
            duration
        )
    
    return response
```

### TASK 4: Integrate Batch Processing into Analysis Pipeline (30 minutes)
**Purpose**: Replace individual calls with batch processing in the main analysis

**File**: `core/analyze.py`
**Modify the evidence analysis section**:

```python
# Add import
from .batch_evidence_evaluator import evaluate_evidence_hypothesis_batch, EvidenceHypothesisPair

def analyze_evidence(evidence_edges, hypotheses, graph, output_dir):
    """Enhanced with batch processing for performance optimization"""
    print(f"[BATCH-ANALYSIS] Processing {len(evidence_edges)} evidence edges against {len(hypotheses)} hypotheses")
    
    # Create evidence-hypothesis pairs
    pairs = []
    for evidence_edge in evidence_edges:
        evidence_text = evidence_edge.get('properties', {}).get('description', '')
        evidence_id = evidence_edge.get('source', evidence_edge.get('id', ''))
        
        for hypothesis in hypotheses:
            hypothesis_text = hypothesis.get('properties', {}).get('description', '')
            hypothesis_id = hypothesis.get('id', '')
            
            pair = EvidenceHypothesisPair(
                pair_id=f"{evidence_id}_vs_{hypothesis_id}",
                evidence_text=evidence_text,
                evidence_id=evidence_id,
                hypothesis_text=hypothesis_text,
                hypothesis_id=hypothesis_id
            )
            pairs.append(pair)
    
    print(f"[BATCH-ANALYSIS] Created {len(pairs)} evidence-hypothesis pairs")
    
    # Process in batches of 8-10 pairs
    BATCH_SIZE = 8
    all_evaluations = []
    total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[batch_num:batch_num + BATCH_SIZE]
        current_batch = batch_num // BATCH_SIZE + 1
        
        print(f"[BATCH-ANALYSIS] Processing batch {current_batch}/{total_batches} ({len(batch_pairs)} pairs)")
        
        try:
            batch_response = evaluate_evidence_hypothesis_batch(
                pairs=batch_pairs,
                context="Process tracing analysis for historical case study"
            )
            all_evaluations.extend(batch_response.evaluations)
            
            print(f"[BATCH-SUCCESS] Batch {current_batch} completed in {batch_response.batch_metadata.get('processing_time', 0):.2f}s")
            
        except Exception as e:
            print(f"[BATCH-ERROR] Batch {current_batch} failed: {e}")
            # Fallback to individual processing for this batch
            print(f"[BATCH-FALLBACK] Processing batch {current_batch} individually")
            for pair in batch_pairs:
                # Individual processing fallback code here
                pass
    
    print(f"[BATCH-ANALYSIS] Completed {len(all_evaluations)} evaluations in {total_batches} batches")
    return all_evaluations
```

### TASK 5: Add A/B Testing Infrastructure (25 minutes)
**Purpose**: Compare batch vs. individual processing performance and quality

**File**: `core/analyze.py`
**Add feature flag system**:

```python
# Add at top of file
ENABLE_BATCH_PROCESSING = True  # Feature flag for batch processing
ENABLE_AB_TESTING = False       # Feature flag for A/B comparison

def analyze_evidence_with_comparison(evidence_edges, hypotheses, graph, output_dir):
    """Run both batch and individual processing for comparison"""
    
    if not ENABLE_AB_TESTING:
        if ENABLE_BATCH_PROCESSING:
            return analyze_evidence_batch(evidence_edges, hypotheses, graph, output_dir)
        else:
            return analyze_evidence_individual(evidence_edges, hypotheses, graph, output_dir)
    
    print("[A/B-TEST] Running both batch and individual processing for comparison")
    
    # Run individual processing (baseline)
    start_time = time.time()
    individual_results = analyze_evidence_individual(evidence_edges, hypotheses, graph, output_dir)
    individual_time = time.time() - start_time
    
    # Run batch processing
    start_time = time.time()
    batch_results = analyze_evidence_batch(evidence_edges, hypotheses, graph, output_dir)
    batch_time = time.time() - start_time
    
    # Compare results
    comparison = {
        "individual_processing": {
            "time": individual_time,
            "results_count": len(individual_results),
            "calls_made": getattr(performance_tracker, 'individual_calls', 0)
        },
        "batch_processing": {
            "time": batch_time,
            "results_count": len(batch_results),
            "calls_made": getattr(performance_tracker, 'batch_calls', 0)
        },
        "improvement": {
            "time_reduction": (individual_time - batch_time) / individual_time * 100,
            "call_reduction": (getattr(performance_tracker, 'individual_calls', 1) - getattr(performance_tracker, 'batch_calls', 1)) / getattr(performance_tracker, 'individual_calls', 1) * 100
        }
    }
    
    # Save comparison results
    if output_dir:
        comparison_file = Path(output_dir) / f"ab_test_comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"[A/B-TEST] Comparison saved to {comparison_file}")
    
    # Use batch results (assuming they pass quality validation)
    return batch_results
```

## 🧪 TESTING PROTOCOL

### Test Level 1: Baseline Measurement (5 minutes)
**Command**: `source test_env/bin/activate && echo -e "1\n1" | python3 process_trace_advanced.py`  
**Expected Output**: Performance baseline JSON with individual call timings
**Success Criteria**: Complete metrics file with 50+ LLM call measurements

### Test Level 2: Batch Processing Test (5 minutes)  
**Setup**: Enable batch processing flag
**Command**: Same as Level 1
**Expected Output**: 80% reduction in LLM calls, similar quality results
**Success Criteria**: <15 total LLM calls, analysis time <3 minutes

### Test Level 3: A/B Comparison (10 minutes)
**Setup**: Enable A/B testing flag  
**Command**: Same as Level 1
**Expected Output**: Side-by-side performance comparison
**Success Criteria**: Quantitative proof of improvement with quality validation

## 🎯 SUCCESS CRITERIA

1. **Performance**: Analysis time reduced from 5-12 minutes to <3 minutes
2. **Efficiency**: 80% reduction in LLM calls (100 calls → <20 calls)  
3. **Quality**: Probative values within 10% of individual processing baseline
4. **Reliability**: 100% Pydantic schema compliance for all batch responses
5. **Fallback**: Graceful degradation to individual processing on batch failures

## 📊 EXPECTED RESULTS

**Before Optimization**:
- 100-150 individual LLM calls
- 5-12 minute analysis time
- 3-5 seconds per call

**After Batch Processing**:
- 10-20 batch LLM calls  
- <3 minute analysis time
- 8-15 seconds per batch (5-10 pairs)
- 80% time reduction achieved

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`process_trace_advanced.py`**: Main orchestration script with project selection and pipeline management
- **`core/analyze.py`**: Analysis phase entry point - current performance bottleneck location
- **`core/extract.py`**: Extraction phase entry point - working perfectly (132.93s for 39 nodes)

### Module Organization  
- **`core/`**: Core processing modules (extraction, analysis, LLM interfaces)
- **`core/plugins/`**: Plugin system architecture with registry-based loading
- **`universal_llm_kit/`**: LLM abstraction layer with LiteLLM integration
- **`input_text/`**: Test cases (French Revolution verified working, American Revolution available)
- **`output_data/`**: Generated outputs (graphs, HTML reports, diagnostic files)

### Important Integration Points
- **LLM Interface**: `core/llm_required.py` with structured output support
- **Plugin Registry**: `core/plugins/register_plugins.py` (loads successfully in WSL)
- **Pydantic Models**: `core/structured_models.py` (validation working perfectly)

### WSL Environment Setup
- **Virtual Environment**: `test_env/` with all dependencies installed
- **Activation Command**: `source test_env/bin/activate`
- **Dependencies**: pandas, litellm, google-generativeai, networkx, pydantic

## 📋 Coding Philosophy

### NO LAZY IMPLEMENTATIONS
- No mocking, stubs, fallbacks, pseudo-code, or simplified implementations
- Every batch processing function must be fully functional with real LLM calls
- Test each implementation thoroughly - assume nothing works until proven

### FAIL-FAST PRINCIPLES  
- Surface LLM failures immediately with LLMRequiredError
- Don't hide batch processing failures - make them visible
- Use feature flags for safe rollback, not to hide problems

### EVIDENCE-BASED DEVELOPMENT
- All optimization claims require performance metrics JSON files
- Raw LLM call logs required for baseline vs. batch comparison
- No success declarations without quantitative proof of improvement

### VALIDATION + SELF-HEALING
- Every batch evaluator must validate Pydantic schema compliance
- Graceful degradation to individual processing on batch failures
- A/B testing infrastructure for quality assurance

## Evidence Structure

Evidence for this phase should be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase21_BatchOptimization.md
```

Include:
- Performance baseline JSON files with individual call timings
- Batch processing results with call reduction measurements  
- A/B comparison data showing quality preservation
- Raw console output demonstrating <3 minute analysis times
- Specific bottleneck elimination proof

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
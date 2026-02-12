# Enhanced LLM Reasoning Implementation Plan
# Date: 2025-09-14
# Status: APPROVED FOR AGGRESSIVE IMPLEMENTATION

## Executive Summary

**OBJECTIVE**: Implement "perfect" LLM academic expert extraction with reasoning chains, mathematical precision, and alternative interpretation generation within existing architecture.

**APPROACH**: Rollback-safe aggressive development with non-breaking schema extensions and feature flags for instant safety.

**TIMELINE**: 2-3 weeks with dual-path testing and quality gates

**ROLLBACK STRATEGY**: Git tagging, feature flags, and shadow testing for maximum safety

---

## Implementation Architecture

### Core Principles

1. **PRESERVE EXISTING SYSTEM**: All current functionality, tests, and architecture maintained
2. **NON-BREAKING EXTENSIONS**: Only additive schema changes
3. **ROLLBACK SAFETY**: Instant revert capability via feature flags
4. **FAIL-FAST VALIDATION**: Strict JSON validation with clear error handling
5. **LLM-FIRST REASONING**: All semantic decisions made by LLM with transparency

### Safety Architecture

#### Rollback Points
```bash
# Primary rollback point
git tag vA-rollback-baseline
git push origin vA-rollback-baseline

# Working branch  
git checkout -b feature/llm-reasoning-max
```

#### Feature Flags (Environment Variables)
```bash
ENHANCED_REASONING_REQUIRED=true    # Enable reasoning chain extraction
DS_ENABLED=true                     # Enable Dempster-Shafer analysis  
POSTHOC_MERGE_ENABLED=true         # Enable post-hoc merging
KILL_SWITCH_ENHANCEMENTS=false     # Emergency revert to baseline
```

#### Dual-Path Testing
- Run enhanced extraction alongside baseline
- Store both outputs for comparison
- Promote only after quality validation
- Instant rollback via KILL_SWITCH_ENHANCEMENTS=true

---

## Technical Implementation Plan

### Week 1: Core Schema Extensions & LLM Prompting

#### Task 1.1: Non-Breaking Schema Extensions (Days 1-2)

**File Modifications**:
- `config/ontology_config.json`: Add new optional fields
- `core/structured_extractor.py`: Extend Pydantic models
- No breaking changes to existing fields

**Schema Extensions**:

```json
{
  "edge_types": {
    "tests_hypothesis": {
      "properties": {
        // EXISTING FIELDS PRESERVED  
        "probative_value": {...},
        "diagnostic_type": {...},
        
        // NEW REQUIRED FIELDS
        "lr_point": {
          "type": "float", 
          "min": 0.001, 
          "required": true,
          "description": "Likelihood ratio P(E|H)/P(E|¬H) - must be > 0"
        },
        "reliability_r": {
          "type": "float",
          "min": 0.0,
          "max": 1.0, 
          "required": true,
          "description": "Source reliability for discounting in DS theory"
        },
        "lr_rationale": {
          "type": "string",
          "required": true,
          "description": "One sentence explaining LR assessment"
        },
        "reliability_rationale": {
          "type": "string", 
          "required": true,
          "description": "One sentence explaining reliability assessment"
        },
        
        // NEW OPTIONAL FIELDS
        "reasoning_chain": {
          "type": "object",
          "required": false,
          "properties": {
            "step_by_step": {
              "type": "array",
              "items": {"type": "string"},
              "description": "Sequential reasoning steps"
            },
            "alternatives_considered": {
              "type": "array", 
              "items": {
                "type": "object",
                "properties": {
                  "interpretation": {"type": "string"},
                  "rejection_reason": {"type": "string"},
                  "probability_assessed": {"type": "float", "min": 0.0, "max": 1.0}
                }
              }
            },
            "confidence_reasoning": {
              "type": "string",
              "description": "Explanation of confidence assessment"
            }
          }
        },
        "probative_assessment": {
          "type": "object",
          "required": false,
          "properties": {
            "p_e_given_h": {"type": "float", "min": 0.0, "max": 1.0},
            "p_e_given_not_h": {"type": "float", "min": 0.0, "max": 1.0},
            "bayes_factor": {"type": "float", "min": 0.001},
            "mathematical_reasoning": {"type": "string"}
          }
        },
        "ds_bpa": {
          "type": "object",
          "required": false,
          "properties": {
            "H": {"type": "float", "min": 0.0, "max": 1.0},
            "not_H": {"type": "float", "min": 0.0, "max": 1.0}, 
            "Theta": {"type": "float", "min": 0.0, "max": 1.0}
          },
          "validation": "sum must equal 1.0"
        },
        "ds_rationale": {
          "type": "string",
          "required": false,
          "description": "One sentence explaining DS mass assignment"
        }
      }
    }
  }
}
```

#### Task 1.2: Enhanced LLM Prompting (Days 2-3)

**New Prompt Templates**:

```python
# core/prompts/enhanced_reasoning_prompts.py

REASONING_ENHANCEMENT_PROMPT = """
SYSTEM: You are an expert academic historian performing process tracing analysis. 
Provide precise numerical assessments and step-by-step reasoning. 
Output ONLY valid JSON per schema. All required fields must be present.

USER: 
Text: {text}
Basic Graph: {basic_graph_json}

For EACH Evidence→Hypothesis relationship in the graph, provide:

1. DIAGNOSTIC ASSESSMENT:
   - diagnostic_type: hoop|smoking_gun|straw_in_the_wind|doubly_decisive
   - Reasoning for diagnostic classification

2. MATHEMATICAL PRECISION:
   - lr_point: Likelihood ratio P(E|H)/P(E|¬H) as single value > 0
   - lr_rationale: One sentence explaining this LR value
   - reliability_r: Source reliability 0.0-1.0  
   - reliability_rationale: One sentence explaining reliability

3. PROBATIVE ASSESSMENT:
   - p_e_given_h: Precise probability (e.g., 0.85, not range)
   - p_e_given_not_h: Precise probability for alternative
   - bayes_factor: Calculated BF = lr_point
   - mathematical_reasoning: Brief justification

4. REASONING CHAIN:
   - step_by_step: 3-5 reasoning steps taken
   - alternatives_considered: 2-3 alternative interpretations with rejection reasons
   - confidence_reasoning: Explanation of certainty assessment

5. DEMPSTER-SHAFER (optional):
   - ds_bpa: Mass assignments {H, not_H, Theta} summing to 1.0
   - ds_rationale: Why these mass values

REQUIREMENTS:
- All probabilities as precise values (0.85, not "high" or "0.8-0.9")
- Mathematical calculations must be consistent
- Reasoning must be academically rigorous
- Alternative interpretations must be plausible and specific

Schema: {schema_json}
"""

TEMPORAL_REASONING_PROMPT = """
SYSTEM: Analyze temporal relationships between events in causal claims.
Provide only LLM assessment, no programmatic date parsing.

USER:
Graph with causal relationships: {graph_json}

For each Event→Event causal edge, assess:

1. temporal_relation: before|after|overlaps|unknown
2. temporal_confidence: 0.0-1.0 confidence in temporal assessment  
3. temporal_reasoning: One sentence explaining temporal judgment
4. time_evidence: Verbatim text snippets supporting temporal assessment

Base assessment only on text evidence, not date calculations.
If temporal order is unclear from text, use "unknown" with explanation.

Schema: {schema_json}
"""
```

#### Task 1.3: Validation Framework (Day 3)

**Strict JSON Validation**:

```python
# core/validation/enhanced_validation.py

def validate_enhanced_reasoning(reasoning_data):
    """Strict validation with fail-fast behavior"""
    
    errors = []
    
    # Required field validation
    required_fields = ['lr_point', 'reliability_r', 'lr_rationale', 'reliability_rationale']
    for field in required_fields:
        if field not in reasoning_data:
            errors.append(f"Missing required field: {field}")
    
    # Numerical validation
    if 'lr_point' in reasoning_data:
        if reasoning_data['lr_point'] <= 0:
            errors.append(f"lr_point must be > 0, got: {reasoning_data['lr_point']}")
    
    if 'reliability_r' in reasoning_data:
        r = reasoning_data['reliability_r']
        if not (0.0 <= r <= 1.0):
            errors.append(f"reliability_r must be in [0,1], got: {r}")
    
    # DS BPA validation (if present)
    if 'ds_bpa' in reasoning_data:
        bpa = reasoning_data['ds_bpa'] 
        mass_sum = bpa.get('H', 0) + bpa.get('not_H', 0) + bpa.get('Theta', 0)
        if abs(mass_sum - 1.0) > 0.001:
            errors.append(f"DS BPA masses must sum to 1.0, got: {mass_sum}")
    
    # Probative assessment consistency
    if 'probative_assessment' in reasoning_data:
        pa = reasoning_data['probative_assessment']
        if 'p_e_given_h' in pa and 'p_e_given_not_h' in pa:
            calculated_bf = pa['p_e_given_h'] / pa['p_e_given_not_h'] if pa['p_e_given_not_h'] > 0 else float('inf')
            stated_bf = reasoning_data.get('lr_point', 0)
            if abs(calculated_bf - stated_bf) > 0.1:
                errors.append(f"Bayes factor inconsistency: calculated {calculated_bf}, stated {stated_bf}")
    
    if errors:
        raise ValidationError(f"Enhanced reasoning validation failed: {errors}")
    
    return True

class ValidationError(Exception):
    pass

def retry_with_validation(llm_call, max_retries=2):
    """Retry logic for LLM calls with strict validation"""
    
    for attempt in range(max_retries + 1):
        try:
            result = llm_call()
            validate_enhanced_reasoning(result)
            return result
        except ValidationError as e:
            if attempt == max_retries:
                # Final failure
                return {
                    "status": "FAIL",
                    "stage": "enhanced_reasoning", 
                    "attempt": attempt + 1,
                    "reason": str(e),
                    "action": "Human review required or system rollback"
                }
            # Retry with same prompt
            continue
```

### Week 2: DS Integration & Temporal Enhancement

#### Task 2.1: Dempster-Shafer Module (Days 4-5)

**Modular DS Implementation**:

```python
# core/analysis/dempster_shafer.py

import math
from typing import Dict, List, Tuple

class DempsterShaferAnalyzer:
    """Modular Dempster-Shafer analysis for hypothesis evaluation"""
    
    def __init__(self):
        self.frame = ["H", "not_H", "Theta"]  # Frame of discernment
    
    def discount_mass(self, bpa: Dict[str, float], reliability: float) -> Dict[str, float]:
        """Discount mass assignment by source reliability"""
        
        m_H = reliability * bpa.get("H", 0.0)
        m_not_H = reliability * bpa.get("not_H", 0.0) 
        m_Theta = 1.0 - reliability * (1.0 - bpa.get("Theta", 0.0))
        
        # Normalize
        total = m_H + m_not_H + m_Theta
        return {
            "H": m_H / total,
            "not_H": m_not_H / total,
            "Theta": m_Theta / total
        }
    
    def combine_masses(self, m1: Dict[str, float], m2: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """Combine two mass functions using Dempster's rule"""
        
        # Calculate conflict
        conflict = m1["H"] * m2["not_H"] + m1["not_H"] * m2["H"]
        
        if conflict >= 1.0:
            # Total conflict - cannot combine
            return None, 1.0
        
        denom = 1.0 - conflict
        
        # Combined masses
        m_H = (m1["H"] * m2["H"] + m1["H"] * m2["Theta"] + m1["Theta"] * m2["H"]) / denom
        m_not_H = (m1["not_H"] * m2["not_H"] + m1["not_H"] * m2["Theta"] + m1["Theta"] * m2["not_H"]) / denom
        m_Theta = (m1["Theta"] * m2["Theta"]) / denom
        
        # Normalize
        total = m_H + m_not_H + m_Theta
        combined = {
            "H": m_H / total,
            "not_H": m_not_H / total, 
            "Theta": m_Theta / total
        }
        
        return combined, conflict
    
    def fold_evidence(self, evidence_list: List[Dict]) -> Dict:
        """Fold all evidence for a hypothesis using DS combination"""
        
        if not evidence_list:
            return {"belief": 0.0, "plausibility": 1.0, "conflict": 0.0, "pignistic": 0.5}
        
        # Start with vacuous mass
        combined_mass = {"H": 0.0, "not_H": 0.0, "Theta": 1.0}
        max_conflict = 0.0
        combination_log = []
        
        for i, evidence in enumerate(evidence_list):
            # Discount by reliability
            discounted = self.discount_mass(evidence["ds_bpa"], evidence["reliability_r"])
            
            # Combine with running total
            new_combined, conflict = self.combine_masses(combined_mass, discounted)
            
            if new_combined is None:
                # Total conflict encountered
                return {
                    "status": "CONFLICT_ERROR",
                    "message": f"Total conflict at evidence {i}: {evidence['evidence_id']}",
                    "conflict": 1.0
                }
            
            combined_mass = new_combined
            max_conflict = max(max_conflict, conflict)
            
            combination_log.append({
                "evidence_id": evidence["evidence_id"],
                "reliability": evidence["reliability_r"],
                "conflict_introduced": conflict,
                "running_mass": combined_mass.copy()
            })
        
        # Calculate final metrics
        belief_H = combined_mass["H"]
        plausibility_H = combined_mass["H"] + combined_mass["Theta"] 
        pignistic_H = combined_mass["H"] + 0.5 * combined_mass["Theta"]
        
        return {
            "belief": belief_H,
            "plausibility": plausibility_H,
            "conflict": max_conflict,
            "pignistic": pignistic_H,
            "final_mass": combined_mass,
            "combination_log": combination_log,
            "evidence_count": len(evidence_list)
        }
    
    def analyze_hypothesis(self, hypothesis_id: str, evidence_list: List[Dict]) -> Dict:
        """Complete DS analysis for a hypothesis"""
        
        # Filter evidence for this hypothesis
        relevant_evidence = [
            ev for ev in evidence_list 
            if ev.get("hypothesis_id") == hypothesis_id and ev.get("ds_bpa")
        ]
        
        if not relevant_evidence:
            return {
                "hypothesis_id": hypothesis_id,
                "status": "NO_DS_DATA",
                "message": "No evidence with DS BPA found"
            }
        
        # Perform DS combination
        ds_result = self.fold_evidence(relevant_evidence)
        
        # Add hypothesis context
        ds_result["hypothesis_id"] = hypothesis_id
        ds_result["evidence_analyzed"] = len(relevant_evidence)
        
        return ds_result

# Integration with existing analysis
def add_ds_analysis(graph_data: Dict) -> Dict:
    """Add DS analysis to existing graph analysis"""
    
    if not os.getenv('DS_ENABLED', 'false').lower() == 'true':
        return graph_data  # DS disabled
    
    ds_analyzer = DempsterShaferAnalyzer()
    
    # Extract evidence for DS analysis
    evidence_list = []
    for edge in graph_data.get("edges", []):
        if edge.get("type") == "tests_hypothesis" and edge.get("properties", {}).get("ds_bpa"):
            evidence_list.append({
                "evidence_id": edge["source_id"],
                "hypothesis_id": edge["target_id"], 
                "ds_bpa": edge["properties"]["ds_bpa"],
                "reliability_r": edge["properties"]["reliability_r"],
                "lr_point": edge["properties"]["lr_point"]
            })
    
    # Analyze each hypothesis
    hypotheses = [node for node in graph_data.get("nodes", []) if node.get("type") == "Hypothesis"]
    ds_results = {}
    
    for hypothesis in hypotheses:
        h_id = hypothesis["id"]
        ds_results[h_id] = ds_analyzer.analyze_hypothesis(h_id, evidence_list)
    
    # Add DS results to graph data
    if "analysis_results" not in graph_data:
        graph_data["analysis_results"] = {}
    
    graph_data["analysis_results"]["dempster_shafer"] = ds_results
    
    return graph_data
```

#### Task 2.2: Temporal Reasoning Enhancement (Day 6)

**LLM-Only Temporal Assessment**:

```python
# core/analysis/temporal_reasoning.py

def enhance_temporal_relationships(graph_data: Dict) -> Dict:
    """Add LLM-based temporal reasoning to causal relationships"""
    
    # Find causal edges (Event → Event)
    causal_edges = [
        edge for edge in graph_data.get("edges", [])
        if edge.get("type") == "causes"
    ]
    
    if not causal_edges:
        return graph_data
    
    # Get LLM temporal assessment
    temporal_prompt = build_temporal_reasoning_prompt(graph_data, causal_edges)
    
    try:
        temporal_assessment = retry_with_validation(
            lambda: call_llm_with_prompt(temporal_prompt),
            max_retries=2
        )
        
        if temporal_assessment.get("status") == "FAIL":
            # Log failure but don't break pipeline
            logging.warning(f"Temporal reasoning failed: {temporal_assessment}")
            return graph_data
        
        # Add temporal enhancements to edges
        for enhancement in temporal_assessment.get("temporal_enhancements", []):
            edge_id = enhancement.get("edge_id")
            
            # Find corresponding edge
            for edge in causal_edges:
                if edge["id"] == edge_id:
                    if "temporal_analysis" not in edge["properties"]:
                        edge["properties"]["temporal_analysis"] = {}
                    
                    edge["properties"]["temporal_analysis"].update({
                        "relation": enhancement.get("temporal_relation", "unknown"),
                        "confidence": enhancement.get("temporal_confidence", 0.0),
                        "reasoning": enhancement.get("temporal_reasoning", ""),
                        "evidence_snippets": enhancement.get("time_evidence", [])
                    })
                    break
        
        return graph_data
        
    except Exception as e:
        logging.warning(f"Temporal reasoning enhancement failed: {e}")
        return graph_data  # Graceful degradation

def build_temporal_reasoning_prompt(graph_data: Dict, causal_edges: List[Dict]) -> str:
    """Build prompt for temporal relationship assessment"""
    
    # Extract relevant events and their descriptions
    event_descriptions = {}
    for node in graph_data.get("nodes", []):
        if node.get("type") == "Event":
            event_descriptions[node["id"]] = {
                "description": node["properties"]["description"],
                "time_raw": node["properties"].get("time_raw", ""),
                "date": node["properties"].get("date", "")
            }
    
    # Build context for each causal relationship
    causal_contexts = []
    for edge in causal_edges:
        source_event = event_descriptions.get(edge["source_id"], {})
        target_event = event_descriptions.get(edge["target_id"], {})
        
        causal_contexts.append({
            "edge_id": edge["id"],
            "source_description": source_event.get("description", ""),
            "target_description": target_event.get("description", ""),
            "source_time_info": source_event.get("time_raw", "") + " " + source_event.get("date", ""),
            "target_time_info": target_event.get("time_raw", "") + " " + target_event.get("date", ""),
            "causal_rationale": edge["properties"].get("rationale", "")
        })
    
    return TEMPORAL_REASONING_PROMPT.format(
        causal_contexts=json.dumps(causal_contexts, indent=2),
        schema_json=get_temporal_schema()
    )
```

### Week 3: Quality Validation & Production Rollout

#### Task 3.1: Dual-Path Testing Implementation (Days 7-8)

**Shadow Testing Architecture**:

```python
# core/testing/dual_path_testing.py

class DualPathTester:
    """Run enhanced and baseline extraction in parallel for comparison"""
    
    def __init__(self, baseline_extractor, enhanced_extractor):
        self.baseline_extractor = baseline_extractor
        self.enhanced_extractor = enhanced_extractor
        self.comparison_results = []
    
    def run_dual_extraction(self, text: str, doc_id: str) -> Dict:
        """Run both extractors and compare results"""
        
        start_time = time.time()
        
        # Run baseline extraction
        try:
            baseline_result = self.baseline_extractor.extract(text, doc_id)
            baseline_success = True
            baseline_error = None
        except Exception as e:
            baseline_result = None
            baseline_success = False
            baseline_error = str(e)
        
        # Run enhanced extraction
        try:
            enhanced_result = self.enhanced_extractor.extract(text, doc_id)
            enhanced_success = True
            enhanced_error = None
        except Exception as e:
            enhanced_result = None
            enhanced_success = False 
            enhanced_error = str(e)
        
        extraction_time = time.time() - start_time
        
        # Compare results
        comparison = self.compare_extractions(baseline_result, enhanced_result)
        
        result = {
            "doc_id": doc_id,
            "timestamp": datetime.utcnow().isoformat(),
            "extraction_time_seconds": extraction_time,
            "baseline": {
                "success": baseline_success,
                "error": baseline_error,
                "node_count": len(baseline_result.get("nodes", [])) if baseline_result else 0,
                "edge_count": len(baseline_result.get("edges", [])) if baseline_result else 0
            },
            "enhanced": {
                "success": enhanced_success,
                "error": enhanced_error,
                "node_count": len(enhanced_result.get("nodes", [])) if enhanced_result else 0,
                "edge_count": len(enhanced_result.get("edges", [])) if enhanced_result else 0,
                "reasoning_chains": self.count_reasoning_chains(enhanced_result) if enhanced_result else 0,
                "ds_analyses": self.count_ds_analyses(enhanced_result) if enhanced_result else 0
            },
            "comparison": comparison
        }
        
        self.comparison_results.append(result)
        return result
    
    def compare_extractions(self, baseline: Dict, enhanced: Dict) -> Dict:
        """Compare baseline vs enhanced extraction results"""
        
        if not baseline or not enhanced:
            return {"status": "INCOMPLETE", "reason": "One extraction failed"}
        
        # Compare structural elements
        baseline_nodes = set(node["id"] for node in baseline.get("nodes", []))
        enhanced_nodes = set(node["id"] for node in enhanced.get("nodes", []))
        
        baseline_edges = set(edge["id"] for edge in baseline.get("edges", []))
        enhanced_edges = set(edge["id"] for edge in enhanced.get("edges", []))
        
        # Check for major structural differences
        node_overlap = len(baseline_nodes & enhanced_nodes) / max(len(baseline_nodes | enhanced_nodes), 1)
        edge_overlap = len(baseline_edges & enhanced_edges) / max(len(baseline_edges | enhanced_edges), 1)
        
        # Quality metrics for enhanced version
        enhanced_quality = self.assess_enhancement_quality(enhanced)
        
        return {
            "node_overlap_ratio": node_overlap,
            "edge_overlap_ratio": edge_overlap,
            "structural_consistency": "HIGH" if node_overlap > 0.8 and edge_overlap > 0.8 else "MEDIUM" if node_overlap > 0.6 else "LOW",
            "enhancement_quality": enhanced_quality,
            "recommendation": self.get_rollout_recommendation(node_overlap, edge_overlap, enhanced_quality)
        }
    
    def assess_enhancement_quality(self, enhanced_result: Dict) -> Dict:
        """Assess quality of enhanced features"""
        
        reasoning_chains = 0
        mathematical_precision = 0
        alternatives_generated = 0
        ds_analyses = 0
        
        for edge in enhanced_result.get("edges", []):
            if edge.get("type") == "tests_hypothesis":
                props = edge.get("properties", {})
                
                # Check for reasoning chains
                if props.get("reasoning_chain"):
                    reasoning_chains += 1
                    
                    # Check for alternatives
                    alternatives = props["reasoning_chain"].get("alternatives_considered", [])
                    if alternatives:
                        alternatives_generated += len(alternatives)
                
                # Check for mathematical precision
                if props.get("probative_assessment"):
                    pa = props["probative_assessment"]
                    if pa.get("p_e_given_h") is not None and pa.get("bayes_factor") is not None:
                        mathematical_precision += 1
                
                # Check for DS analysis
                if props.get("ds_bpa"):
                    ds_analyses += 1
        
        return {
            "reasoning_chains_count": reasoning_chains,
            "mathematical_precision_count": mathematical_precision,
            "alternatives_generated": alternatives_generated,
            "ds_analyses_count": ds_analyses,
            "quality_score": self.calculate_quality_score(
                reasoning_chains, mathematical_precision, alternatives_generated, ds_analyses
            )
        }
    
    def get_rollout_recommendation(self, node_overlap: float, edge_overlap: float, quality: Dict) -> str:
        """Recommend whether to proceed with rollout"""
        
        if node_overlap < 0.6 or edge_overlap < 0.6:
            return "DO_NOT_ROLLOUT - Low structural consistency"
        
        if quality["quality_score"] < 0.5:
            return "INVESTIGATE - Low enhancement quality"
        
        if node_overlap > 0.8 and edge_overlap > 0.8 and quality["quality_score"] > 0.7:
            return "PROCEED_WITH_ROLLOUT - High quality and consistency"
        
        return "CAUTIOUS_ROLLOUT - Moderate quality, monitor closely"
    
    def calculate_quality_score(self, reasoning_chains: int, math_precision: int, alternatives: int, ds_analyses: int) -> float:
        """Calculate overall quality score for enhancements"""
        
        # Weight different quality factors
        weights = {
            "reasoning_chains": 0.4,
            "mathematical_precision": 0.3,
            "alternatives": 0.2,
            "ds_analyses": 0.1
        }
        
        # Normalize by expected counts (assume 10 evidence-hypothesis relationships)
        expected_count = 10
        
        scores = {
            "reasoning_chains": min(reasoning_chains / expected_count, 1.0),
            "mathematical_precision": min(math_precision / expected_count, 1.0), 
            "alternatives": min(alternatives / (expected_count * 2), 1.0),  # 2 alternatives per evidence
            "ds_analyses": min(ds_analyses / expected_count, 1.0)
        }
        
        quality_score = sum(weights[key] * scores[key] for key in weights)
        return quality_score
```

#### Task 3.2: Quality Gates & Rollout Decision (Days 8-9)

**Automated Quality Assessment**:

```python
# core/quality/quality_gates.py

class QualityGateManager:
    """Manage quality gates for production rollout decision"""
    
    def __init__(self, test_documents: List[str]):
        self.test_documents = test_documents
        self.quality_thresholds = {
            "structural_consistency_min": 0.8,
            "enhancement_quality_min": 0.7,
            "error_rate_max": 0.1,
            "performance_degradation_max": 2.0  # 2x slower is acceptable
        }
    
    def run_quality_assessment(self, dual_path_results: List[Dict]) -> Dict:
        """Run comprehensive quality assessment"""
        
        # Aggregate results
        total_tests = len(dual_path_results)
        successful_tests = sum(1 for r in dual_path_results if r["enhanced"]["success"])
        
        structural_consistencies = [
            r["comparison"]["node_overlap_ratio"] 
            for r in dual_path_results 
            if r["comparison"].get("node_overlap_ratio")
        ]
        
        enhancement_qualities = [
            r["comparison"]["enhancement_quality"]["quality_score"]
            for r in dual_path_results
            if r["comparison"].get("enhancement_quality", {}).get("quality_score")
        ]
        
        performance_ratios = [
            r["extraction_time_seconds"] / 60.0  # Assume baseline takes ~60s
            for r in dual_path_results
            if r["extraction_time_seconds"]
        ]
        
        # Calculate metrics
        error_rate = 1.0 - (successful_tests / total_tests) if total_tests > 0 else 1.0
        avg_structural_consistency = sum(structural_consistencies) / len(structural_consistencies) if structural_consistencies else 0.0
        avg_enhancement_quality = sum(enhancement_qualities) / len(enhancement_qualities) if enhancement_qualities else 0.0
        avg_performance_ratio = sum(performance_ratios) / len(performance_ratios) if performance_ratios else float('inf')
        
        # Assess against thresholds
        quality_checks = {
            "error_rate_acceptable": error_rate <= self.quality_thresholds["error_rate_max"],
            "structural_consistency_acceptable": avg_structural_consistency >= self.quality_thresholds["structural_consistency_min"],
            "enhancement_quality_acceptable": avg_enhancement_quality >= self.quality_thresholds["enhancement_quality_min"],
            "performance_acceptable": avg_performance_ratio <= self.quality_thresholds["performance_degradation_max"]
        }
        
        all_checks_passed = all(quality_checks.values())
        
        # Generate recommendation
        if all_checks_passed:
            recommendation = "APPROVE_ROLLOUT"
            reason = "All quality gates passed"
        else:
            failed_checks = [check for check, passed in quality_checks.items() if not passed]
            recommendation = "BLOCK_ROLLOUT" 
            reason = f"Quality gates failed: {failed_checks}"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "metrics": {
                "error_rate": error_rate,
                "avg_structural_consistency": avg_structural_consistency,
                "avg_enhancement_quality": avg_enhancement_quality,
                "avg_performance_ratio": avg_performance_ratio
            },
            "quality_checks": quality_checks,
            "recommendation": recommendation,
            "reason": reason,
            "detailed_results": dual_path_results
        }
    
    def generate_quality_report(self, assessment: Dict) -> str:
        """Generate human-readable quality report"""
        
        report = f"""
# Enhanced LLM Reasoning Quality Assessment Report
Generated: {assessment['timestamp']}

## Summary
- **Tests Run**: {assessment['total_tests']}
- **Success Rate**: {assessment['successful_tests']}/{assessment['total_tests']} ({assessment['successful_tests']/assessment['total_tests']*100:.1f}%)
- **Recommendation**: {assessment['recommendation']}
- **Reason**: {assessment['reason']}

## Quality Metrics
- **Error Rate**: {assessment['metrics']['error_rate']:.3f} (threshold: ≤{self.quality_thresholds['error_rate_max']})
- **Structural Consistency**: {assessment['metrics']['avg_structural_consistency']:.3f} (threshold: ≥{self.quality_thresholds['structural_consistency_min']})  
- **Enhancement Quality**: {assessment['metrics']['avg_enhancement_quality']:.3f} (threshold: ≥{self.quality_thresholds['enhancement_quality_min']})
- **Performance Impact**: {assessment['metrics']['avg_performance_ratio']:.1f}x (threshold: ≤{self.quality_thresholds['performance_degradation_max']}x)

## Quality Gates Status
"""
        
        for check, passed in assessment['quality_checks'].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report += f"- **{check}**: {status}\n"
        
        if assessment['recommendation'] == "APPROVE_ROLLOUT":
            report += """
## Rollout Authorization
**STATUS**: ✅ APPROVED FOR PRODUCTION ROLLOUT

The enhanced LLM reasoning system has passed all quality gates and is ready for production deployment.

### Next Steps:
1. Set ENHANCED_REASONING_REQUIRED=true in production
2. Monitor initial production usage
3. Keep rollback capability ready via KILL_SWITCH_ENHANCEMENTS flag
"""
        else:
            report += """
## Rollout Block
**STATUS**: ❌ BLOCKED FROM PRODUCTION

Quality issues detected that require resolution before production deployment.

### Required Actions:
1. Investigate failed quality checks
2. Address identified issues
3. Re-run quality assessment
4. Consider using KILL_SWITCH_ENHANCEMENTS if already deployed
"""
        
        return report
```

---

## Risk Analysis & Mitigation

### Implementation Risks

#### 1. **LLM Response Quality Risk**
- **Risk**: LLM fails to provide consistent reasoning chains or mathematical precision
- **Mitigation**: Strict validation with retry logic; fallback to baseline on failure
- **Monitoring**: Track validation failure rates and reasoning chain quality

#### 2. **Performance Degradation Risk**  
- **Risk**: Enhanced extraction significantly slower than baseline
- **Mitigation**: Performance monitoring in dual-path testing; rollback if >2x slower
- **Monitoring**: Track extraction times per document size

#### 3. **Schema Validation Risk**
- **Risk**: New required fields break existing functionality
- **Mitigation**: Non-breaking schema design; all new fields optional or with defaults
- **Monitoring**: Validate all existing tests continue passing

#### 4. **Feature Flag Risk**
- **Risk**: Feature flags not properly implemented or tested
- **Mitigation**: Comprehensive feature flag testing; manual verification of rollback
- **Monitoring**: Test all flag combinations before production

### Technical Uncertainties

#### 1. **LLM Mathematical Consistency**
- **Uncertainty**: Can LLM provide consistent P(E|H) values and Bayes factor calculations?
- **Investigation**: Test mathematical consistency across multiple runs with same evidence
- **Timeline**: Week 1, Day 2-3

#### 2. **Reasoning Chain Quality**  
- **Uncertainty**: Will LLM reasoning chains meet academic standards?
- **Investigation**: Expert review of reasoning chain samples; academic validation
- **Timeline**: Week 3, Day 8

#### 3. **Dempster-Shafer Integration Complexity**
- **Uncertainty**: How complex will DS theory integration be with existing Van Evera logic?
- **Investigation**: Prototype DS module separately; test integration points
- **Timeline**: Week 2, Day 4-5

#### 4. **Alternative Interpretation Generation**
- **Uncertainty**: Can LLM consistently generate plausible alternative interpretations?
- **Investigation**: Test alternative generation quality; assess academic rigor
- **Timeline**: Week 1, Day 3

### Implementation Unknowns

#### 1. **Current System Integration Points**
- **Unknown**: Which files will require modification for schema extensions?
- **Investigation**: Comprehensive grep analysis of ontology field usage
- **Action**: Complete before Week 1 implementation

#### 2. **Test Suite Impact**
- **Unknown**: How many tests will need updates for new optional fields?  
- **Investigation**: Run existing test suite against extended schemas
- **Action**: Validate during Week 1, Task 1.1

#### 3. **Prompt Engineering Optimization**
- **Unknown**: What prompt structure yields best reasoning chain quality?
- **Investigation**: A/B test different prompt templates; optimize for consistency
- **Action**: Iterative improvement during Week 1

#### 4. **Production Load Impact**
- **Unknown**: How will enhanced extraction perform under production load?
- **Investigation**: Load testing with enhanced extraction; measure resource usage
- **Action**: Week 3 quality testing phase

---

## Success Criteria & Quality Metrics

### Primary Success Criteria

#### 1. **Reasoning Chain Quality**
- ✅ 90%+ of evidence-hypothesis relationships have complete reasoning chains
- ✅ Academic reviewer rates reasoning quality as "graduate level" or higher
- ✅ Reasoning steps are logically coherent and factually accurate

#### 2. **Mathematical Precision**  
- ✅ All probative assessments use precise format (0.85, not ranges)
- ✅ Bayes factor calculations are mathematically consistent
- ✅ P(E|H) values are academically justified and logically coherent

#### 3. **Alternative Interpretation Generation**
- ✅ Average 2+ alternative interpretations per evidence item
- ✅ Alternatives are plausible and academically rigorous
- ✅ Rejection reasoning is clear and well-justified

#### 4. **System Reliability**
- ✅ Enhanced extraction success rate >90%
- ✅ All existing functionality preserved (22/22 tests passing)
- ✅ Performance degradation <2x baseline extraction time

### Quality Gates

#### Gate 1: Technical Implementation (End of Week 1)
- [ ] Schema extensions implemented without breaking changes
- [ ] Enhanced LLM prompts generate valid JSON responses  
- [ ] Validation framework properly rejects invalid responses
- [ ] Feature flags functional and tested

#### Gate 2: Enhanced Analysis Integration (End of Week 2)  
- [ ] Dempster-Shafer module properly integrated
- [ ] Temporal reasoning enhancement functional
- [ ] Dual-path testing architecture operational
- [ ] No regression in existing functionality

#### Gate 3: Production Readiness (End of Week 3)
- [ ] Quality assessment shows >80% structural consistency
- [ ] Enhancement quality score >70%
- [ ] Academic validation of reasoning chain quality
- [ ] Rollback procedures tested and verified

### Measurement & Monitoring

#### Real-Time Metrics
```python
# Metrics to track during implementation
IMPLEMENTATION_METRICS = {
    "extraction_success_rate": "Percentage of successful enhanced extractions",
    "reasoning_chain_completeness": "Percentage of evidence with full reasoning chains",
    "mathematical_precision_rate": "Percentage of evidence with precise P(E|H) values",
    "alternative_generation_rate": "Average alternatives generated per evidence item",
    "validation_failure_rate": "Percentage of LLM responses requiring retry",
    "performance_degradation": "Enhanced extraction time / baseline extraction time",
    "structural_consistency": "Overlap ratio between enhanced and baseline extractions",
    "ds_analysis_success": "Percentage of hypotheses successfully analyzed with DS theory"
}
```

#### Quality Validation Process
```python
# Quality validation workflow
QUALITY_VALIDATION_STEPS = [
    "1. Technical validation - Schema and JSON compliance",
    "2. Mathematical validation - P(E|H) and Bayes factor consistency", 
    "3. Academic validation - Expert review of reasoning quality",
    "4. Performance validation - Extraction time and resource usage",
    "5. Integration validation - Compatibility with existing system",
    "6. Rollback validation - Feature flag and safety mechanism testing"
]
```

---

## Rollback & Safety Procedures

### Immediate Rollback (Emergency)
```bash
# Emergency rollback via environment variable
export KILL_SWITCH_ENHANCEMENTS=true

# Restart application with baseline extraction only
sudo systemctl restart process_tracing_service

# Verify rollback successful
python -c "from core.structured_extractor import get_extraction_mode; print(get_extraction_mode())"
# Expected output: "baseline"
```

### Planned Rollback (Quality Issues)
```bash
# Disable enhanced features gradually
export ENHANCED_REASONING_REQUIRED=false
export DS_ENABLED=false  
export POSTHOC_MERGE_ENABLED=false

# Revert to tagged baseline
git checkout vA-rollback-baseline

# Verify system functionality
python analyze_direct.py input_text/russia_ukraine_debate/westminster_pirchner_v_bryan.txt --validation-mode strict
```

### Rollback Verification Checklist
- [ ] All 22 existing tests passing
- [ ] Westminster debate test case processes successfully  
- [ ] Extraction output matches baseline format
- [ ] Performance returns to baseline levels
- [ ] No enhanced fields present in output

---

## Communication & Stakeholder Management

### Weekly Status Reports

#### Week 1 Status Template
```markdown
# Week 1 Status: Core Schema Extensions & LLM Prompting

## Completed
- [ ] Non-breaking schema extensions implemented
- [ ] Enhanced LLM prompts developed and tested
- [ ] Validation framework with fail-fast behavior
- [ ] Feature flags and safety architecture

## Quality Metrics  
- Validation success rate: X%
- Reasoning chain completeness: X%
- Mathematical precision rate: X%

## Risks/Issues
- List any technical challenges or blockers

## Next Week Focus
- Dempster-Shafer integration
- Temporal reasoning enhancement
```

### Rollout Communication Plan

#### Pre-Rollout (Week 3)
- **Internal**: Technical team briefing on enhanced features
- **Documentation**: Updated API documentation with new fields
- **Training**: User guide for interpreting reasoning chains

#### Post-Rollout (Week 4+)
- **Monitoring**: Daily quality metrics review for first week
- **Feedback**: Academic user feedback collection
- **Optimization**: Performance and quality improvements based on usage

---

## Investigation Requirements

Before proceeding with implementation, the following uncertainties require investigation:

### Critical Investigations (Must Complete Before Week 1)

#### 1. **Current System Integration Analysis**
```bash
# Find all files referencing ontology structure
grep -r "tests_hypothesis" --include="*.py" .
grep -r "Evidence.*properties" --include="*.py" .  
grep -r "diagnostic_type" --include="*.py" .

# Document exact modification points
find . -name "*.py" -exec grep -l "ontology_config" {} \;
```
**Timeline**: 1 day before implementation start
**Owner**: Technical lead

#### 2. **LLM Mathematical Precision Testing**
```python
# Test LLM consistency for mathematical calculations
test_cases = [
    {"evidence": "Napoleon invaded Russia in winter", "hypothesis": "Napoleon was overconfident"},
    # ... more test cases
]

# Run same evidence-hypothesis pair 10 times
# Measure consistency of P(E|H) values and Bayes factors
# Acceptable variance: <10% standard deviation
```
**Timeline**: Week 1, Day 1
**Owner**: LLM integration specialist

#### 3. **Feature Flag Architecture Validation**  
```python
# Test all feature flag combinations
flag_combinations = [
    {"ENHANCED_REASONING_REQUIRED": True, "DS_ENABLED": True, "KILL_SWITCH": False},
    {"ENHANCED_REASONING_REQUIRED": True, "DS_ENABLED": False, "KILL_SWITCH": False}, 
    {"ENHANCED_REASONING_REQUIRED": False, "DS_ENABLED": False, "KILL_SWITCH": True},
    # ... all combinations
]

# Verify system behavior for each combination
# Ensure no undefined states or crashes
```
**Timeline**: Week 1, Day 2  
**Owner**: DevOps engineer

### Secondary Investigations (Can Be Done During Implementation)

#### 4. **Academic Quality Standards Definition**
- Define specific criteria for "graduate level" reasoning chains
- Establish rubric for alternative interpretation evaluation
- Create benchmark dataset for academic validation

#### 5. **Performance Baseline Establishment**
- Measure current extraction times across document types
- Establish acceptable performance degradation thresholds
- Identify optimization opportunities

#### 6. **Dempster-Shafer Theory Integration Research**
- Review academic literature on DS theory in process tracing
- Validate mathematical implementation against academic standards  
- Test DS conflict detection with real-world examples

---

## Conclusion

This implementation plan provides a **comprehensive roadmap** for enhancing the process tracing system with advanced LLM reasoning capabilities while maintaining **maximum safety** and **rollback capability**.

### Key Strengths of This Plan:
1. **Non-breaking architecture** preserves all existing functionality
2. **Rollback safety** provides multiple safety nets and instant revert capability  
3. **Quality-driven approach** with comprehensive testing and validation
4. **Risk mitigation** addresses all identified technical and implementation risks
5. **Clear success criteria** with measurable quality gates

### Implementation Readiness:
- ✅ Technical architecture defined
- ✅ Rollback procedures established  
- ✅ Quality gates specified
- ✅ Risk mitigation planned
- ⚠️ Critical investigations required before start

**RECOMMENDATION**: Complete critical investigations, then proceed with aggressive implementation using rollback-safe approach.
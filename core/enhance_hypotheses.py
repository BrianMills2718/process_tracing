"""
Enhanced Hypothesis Evaluation with LLM Integration
Implements sophisticated hypothesis confidence scoring using Van Evera methodology
"""

import logging
from typing import Dict, List, Optional, Any
from .plugins.van_evera_llm_interface import VanEveraLLMInterface
from .plugins.van_evera_llm_schemas import VanEveraPredictionEvaluation

logger = logging.getLogger(__name__)


def enhance_hypothesis_with_llm(hypothesis_node: Dict[str, Any], 
                               supporting_evidence: List[Dict[str, Any]], 
                               refuting_evidence: List[Dict[str, Any]], 
                               van_evera_tests: Optional[Dict[str, Any]] = None,
                               graph_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Use VanEveraLLMInterface for sophisticated hypothesis evaluation with structured output.
    
    Args:
        hypothesis_node: The hypothesis node from the graph
        supporting_evidence: List of supporting evidence nodes
        refuting_evidence: List of refuting evidence nodes
        van_evera_tests: Results from Van Evera testing engine
        graph_context: Additional graph context for evaluation
        
    Returns:
        Dict containing enhanced hypothesis assessment with LLM-generated confidence and reasoning
    """
    try:
        # Initialize LLM interface
        llm_interface = VanEveraLLMInterface()
        
        # Extract hypothesis information
        hypothesis_id = hypothesis_node.get('id', 'unknown')
        hypothesis_description = hypothesis_node.get('properties', {}).get('description', '')
        
        # Determine primary diagnostic type from Van Evera tests
        diagnostic_type = _determine_primary_diagnostic_type(van_evera_tests) if van_evera_tests else "hoop"
        
        # Extract theoretical mechanism
        theoretical_mechanism = _extract_theoretical_mechanism(hypothesis_node, graph_context)
        
        # Compile evidence context
        evidence_context = _compile_evidence_context(supporting_evidence, refuting_evidence, graph_context)
        
        logger.info(f"Enhancing hypothesis {hypothesis_id} with LLM evaluation", 
                   extra={'hypothesis_id': hypothesis_id, 'diagnostic_type': diagnostic_type})
        
        # Get structured LLM evaluation
        structured_result = llm_interface.evaluate_prediction_structured(
            prediction_description=hypothesis_description,
            diagnostic_type=diagnostic_type,
            theoretical_mechanism=theoretical_mechanism,
            evidence_context=evidence_context
        )
        
        # Combine Van Evera test results with LLM assessment
        enhanced_assessment = {
            'hypothesis_id': hypothesis_id,
            'description': hypothesis_description,
            
            # LLM-generated confidence and reasoning
            'llm_confidence_score': structured_result.confidence_score,
            'llm_reasoning': structured_result.diagnostic_reasoning,
            'evidence_assessment': structured_result.evidence_assessment,
            'theoretical_mechanism_evaluation': structured_result.theoretical_mechanism_evaluation,
            
            # Van Evera academic analysis
            'diagnostic_type': diagnostic_type,
            'test_result': structured_result.test_result.value,
            'necessity_analysis': structured_result.necessity_analysis,
            'sufficiency_analysis': structured_result.sufficiency_analysis,
            'elimination_implications': structured_result.elimination_implications,
            
            # Evidence quality metrics
            'evidence_quality': structured_result.evidence_quality.value,
            'evidence_coverage': structured_result.evidence_coverage,
            'indicator_matches': structured_result.indicator_matches,
            
            # Academic quality assessment
            'academic_quality': structured_result.publication_quality_assessment,
            'methodological_soundness': structured_result.methodological_soundness,
            
            # Evidence counts and context
            'supporting_evidence_count': len(supporting_evidence),
            'refuting_evidence_count': len(refuting_evidence),
            'evidence_balance': _calculate_evidence_balance(supporting_evidence, refuting_evidence)
        }
        
        logger.info(f"Successfully enhanced hypothesis {hypothesis_id}", 
                   extra={'confidence_score': structured_result.confidence_score,
                          'test_result': structured_result.test_result.value,
                          'evidence_quality': structured_result.evidence_quality.value})
        
        return enhanced_assessment
        
    except Exception as e:
        logger.error(f"Failed to enhance hypothesis {hypothesis_id} with LLM - FAILING FAST", 
                    exc_info=True, extra={'hypothesis_id': hypothesis_id})
        raise  # FAIL FAST - no fallbacks


def _determine_primary_diagnostic_type(van_evera_tests: Dict[str, Any]) -> str:
    """
    Determine the primary diagnostic type from Van Evera test results.
    Prioritizes based on test strength and academic significance.
    """
    if not van_evera_tests:
        return "hoop"  # Default to most restrictive test
    
    test_results = getattr(van_evera_tests, 'test_results', [])
    if not test_results:
        return "hoop"
    
    # Count test types
    test_types = [getattr(test, 'prediction_id', '').split('_')[-1] if hasattr(test, 'prediction_id') else 'hoop' 
                  for test in test_results]
    
    # Priority order: doubly_decisive > hoop > smoking_gun > straw_in_wind
    if 'doubly_decisive' in test_types:
        return "doubly_decisive"
    elif 'hoop' in test_types:
        return "hoop"
    elif 'smoking_gun' in test_types:
        return "smoking_gun"
    else:
        return "straw_in_wind"


def _extract_theoretical_mechanism(hypothesis_node: Dict[str, Any], 
                                 graph_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Extract the theoretical causal mechanism from hypothesis and graph context.
    """
    mechanism_description = hypothesis_node.get('properties', {}).get('description', '')
    
    # Look for connected mechanism nodes in graph context
    if graph_context and 'nodes' in graph_context:
        mechanism_nodes = [node for node in graph_context['nodes'] 
                          if node.get('type') == 'Causal_Mechanism']
        
        if mechanism_nodes:
            mechanisms = [node.get('properties', {}).get('description', '') 
                         for node in mechanism_nodes]
            mechanism_context = " | ".join(mechanisms)
            return f"{mechanism_description} (Mechanisms: {mechanism_context})"
    
    return mechanism_description


def _compile_evidence_context(supporting_evidence: List[Dict[str, Any]], 
                            refuting_evidence: List[Dict[str, Any]],
                            graph_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Compile comprehensive evidence context for LLM evaluation.
    """
    context_parts = []
    
    # Supporting evidence
    if supporting_evidence:
        supporting_desc = []
        for ev in supporting_evidence:
            desc = ev.get('description', ev.get('properties', {}).get('description', 'No description'))
            ev_type = ev.get('type', 'unknown')
            supporting_desc.append(f"- [{ev_type}] {desc}")
        
        context_parts.append("SUPPORTING EVIDENCE:")
        context_parts.extend(supporting_desc)
    
    # Refuting evidence
    if refuting_evidence:
        refuting_desc = []
        for ev in refuting_evidence:
            desc = ev.get('description', ev.get('properties', {}).get('description', 'No description'))
            ev_type = ev.get('type', 'unknown')
            refuting_desc.append(f"- [{ev_type}] {desc}")
        
        context_parts.append("\nREFUTING EVIDENCE:")
        context_parts.extend(refuting_desc)
    
    # Additional context from graph
    if graph_context and 'nodes' in graph_context:
        context_nodes = [node for node in graph_context['nodes'] 
                        if node.get('type') in ['Context', 'Actor', 'Condition']]
        
        if context_nodes:
            context_desc = []
            for node in context_nodes[:5]:  # Limit to avoid token overflow
                desc = node.get('properties', {}).get('description', 'No description')
                node_type = node.get('type', 'unknown')
                context_desc.append(f"- [{node_type}] {desc}")
            
            context_parts.append("\nADDITIONAL CONTEXT:")
            context_parts.extend(context_desc)
    
    if not context_parts:
        return "No evidence context available."
    
    return "\n".join(context_parts)


def _calculate_evidence_balance(supporting_evidence: List[Dict[str, Any]], 
                              refuting_evidence: List[Dict[str, Any]]) -> float:
    """
    Calculate evidence balance ratio (academic standard: 0.6-0.8 support ratio).
    """
    total_evidence = len(supporting_evidence) + len(refuting_evidence)
    if total_evidence == 0:
        return 0.0
    
    support_ratio = len(supporting_evidence) / total_evidence
    return support_ratio
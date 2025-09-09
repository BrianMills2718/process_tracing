"""
Diagnostic Rebalancing Integration Module
Simple integration functions for adding diagnostic rebalancing to existing analysis pipelines
"""

import json
from typing import Dict, Any, Optional, Callable
from .diagnostic_rebalancer import DiagnosticRebalancerPlugin, create_diagnostic_rebalancer_plugin
from .base import PluginContext


def rebalance_graph_diagnostics(
    graph_data: Dict[str, Any], 
    llm_query_func: Optional[Callable] = None,
    target_distribution: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Simple function to rebalance diagnostic distribution in graph data.
    
    Args:
        graph_data: Graph data with nodes and edges
        llm_query_func: Optional LLM query function for enhanced assessment
        target_distribution: Optional custom target distribution (defaults to Van Evera standards)
        
    Returns:
        Dictionary containing:
        - 'updated_graph_data': Graph data with rebalanced diagnostic types
        - 'rebalancing_summary': Summary of changes made
        - 'academic_compliance': Compliance metrics
    """
    
    # Create minimal plugin context
    config: Dict[str, Any] = {
        'diagnostic_rebalancing.enabled': True,
        'van_evera.academic_standards': True
    }
    
    if target_distribution:
        config['diagnostic_rebalancing.target_distribution'] = target_distribution
    
    context = PluginContext(config)
    
    # Create and execute plugin
    plugin = create_diagnostic_rebalancer_plugin(context, llm_query_func)
    plugin.initialize()
    
    try:
        # Execute rebalancing
        result = plugin.execute({'graph_data': graph_data})
        
        return {
            'updated_graph_data': result['updated_graph_data'],
            'rebalancing_summary': {
                'rebalanced_count': result['rebalanced_count'],
                'enhanced_count': result['enhanced_count'], 
                'error_count': result['error_count'],
                'compliance_improvement': result['compliance_improvement']
            },
            'academic_compliance': {
                'original_score': result['original_distribution']['academic_compliance_score'],
                'rebalanced_score': result['rebalanced_distribution']['academic_compliance_score'],
                'van_evera_compliant': result['rebalanced_distribution']['academic_compliance_score'] >= 75
            },
            'distribution_analysis': {
                'original': result['original_distribution']['percentages'],
                'rebalanced': result['rebalanced_distribution']['percentages']
            }
        }
        
    finally:
        plugin.cleanup()


def validate_diagnostic_distribution(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze current diagnostic distribution without making changes.
    
    Args:
        graph_data: Graph data to analyze
        
    Returns:
        Analysis of current diagnostic distribution
    """
    from .diagnostic_rebalancer import DiagnosticDistribution, VAN_EVERA_TARGET_DISTRIBUTION
    
    # Find evidence edges
    evidence_edges = []
    node_lookup = {n['id']: n for n in graph_data['nodes']}
    
    for edge in graph_data['edges']:
        source_id = edge.get('source_id')
        target_id = edge.get('target_id')
        
        if source_id in node_lookup and target_id in node_lookup:
            source_node = node_lookup[source_id]
            target_node = node_lookup[target_id]
            
            if (source_node.get('type') == 'Evidence' and 
                target_node.get('type') in ['Hypothesis', 'Alternative_Explanation']):
                evidence_edges.append(edge)
    
    # Count distribution
    counts = {'hoop': 0, 'smoking_gun': 0, 'doubly_decisive': 0, 'straw_in_wind': 0, 'general': 0}
    
    for edge in evidence_edges:
        diagnostic_type = edge.get('properties', {}).get('diagnostic_type', 'general')
        
        # Handle legacy naming: straw_in_the_wind -> straw_in_wind
        if diagnostic_type == 'straw_in_the_wind':
            diagnostic_type = 'straw_in_wind'
            
        if diagnostic_type in counts:
            counts[diagnostic_type] += 1
        else:
            counts['general'] += 1
    
    distribution = DiagnosticDistribution(**counts)
    
    return {
        'total_evidence_relationships': distribution.total,
        'current_distribution': distribution.percentages,
        'target_distribution': VAN_EVERA_TARGET_DISTRIBUTION,
        'academic_compliance_score': distribution.academic_compliance_score,
        'van_evera_compliant': distribution.academic_compliance_score >= 75,
        'needs_rebalancing': distribution.academic_compliance_score < 80,
        'distribution_gaps': {
            test_type: {
                'current': distribution.percentages.get(test_type, 0),
                'target': target_pct,
                'gap': target_pct - distribution.percentages.get(test_type, 0),
                'needs_adjustment': abs(target_pct - distribution.percentages.get(test_type, 0)) > 0.05
            }
            for test_type, target_pct in VAN_EVERA_TARGET_DISTRIBUTION.items()
        }
    }


def apply_van_evera_rebalancing_to_analysis(
    analysis_results: Dict[str, Any],
    llm_query_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Apply Van Evera diagnostic rebalancing to existing analysis results.
    
    Args:
        analysis_results: Existing analysis results with graph_data
        llm_query_func: Optional LLM query function
        
    Returns:
        Updated analysis results with rebalanced diagnostics
    """
    
    if 'graph_data' not in analysis_results:
        raise ValueError("Analysis results must contain 'graph_data' for rebalancing")
    
    # Perform rebalancing
    rebalancing_result = rebalance_graph_diagnostics(
        analysis_results['graph_data'], 
        llm_query_func
    )
    
    # Update analysis results
    updated_results = analysis_results.copy()
    updated_results['graph_data'] = rebalancing_result['updated_graph_data']
    
    # Add rebalancing metadata
    updated_results['diagnostic_rebalancing'] = {
        'performed': True,
        'summary': rebalancing_result['rebalancing_summary'],
        'compliance': rebalancing_result['academic_compliance'],
        'distribution_analysis': rebalancing_result['distribution_analysis'],
        'methodology': 'Van Evera Diagnostic Rebalancing'
    }
    
    # Update academic quality if present
    if 'academic_quality' in updated_results:
        updated_results['academic_quality']['diagnostic_rebalancing_applied'] = True
        updated_results['academic_quality']['diagnostic_compliance_score'] = rebalancing_result['academic_compliance']['rebalanced_score']
    
    return updated_results


def create_van_evera_rebalancing_report(rebalancing_result: Dict[str, Any]) -> str:
    """
    Generate a human-readable report of Van Evera diagnostic rebalancing.
    
    Args:
        rebalancing_result: Result from rebalance_graph_diagnostics()
        
    Returns:
        Formatted text report
    """
    
    summary = rebalancing_result['rebalancing_summary']
    compliance = rebalancing_result['academic_compliance']
    distribution = rebalancing_result['distribution_analysis']
    
    report = f"""VAN EVERA DIAGNOSTIC REBALANCING REPORT
{'='*50}

REBALANCING SUMMARY:
- Evidence relationships processed: {summary['rebalanced_count'] + summary['error_count']}
- Successfully rebalanced: {summary['rebalanced_count']}
- LLM enhancements applied: {summary['enhanced_count']}
- Processing errors: {summary['error_count']}

ACADEMIC COMPLIANCE:
- Original compliance: {compliance['original_score']:.1f}%
- Rebalanced compliance: {compliance['rebalanced_score']:.1f}%
- Improvement: +{summary['compliance_improvement']:.1f}%
- Van Evera compliant: {'YES' if compliance['van_evera_compliant'] else 'NO'}

DISTRIBUTION ANALYSIS:
                   Original    Rebalanced    Target
Hoop Tests         {distribution['original'].get('hoop', 0):.1%}       {distribution['rebalanced'].get('hoop', 0):.1%}        25.0%
Smoking Gun        {distribution['original'].get('smoking_gun', 0):.1%}       {distribution['rebalanced'].get('smoking_gun', 0):.1%}        25.0%
Doubly Decisive    {distribution['original'].get('doubly_decisive', 0):.1%}       {distribution['rebalanced'].get('doubly_decisive', 0):.1%}        15.0%
Straw-in-Wind      {distribution['original'].get('straw_in_wind', 0):.1%}       {distribution['rebalanced'].get('straw_in_wind', 0):.1%}        35.0%

METHODOLOGY COMPLIANCE:
{'[YES]' if compliance['rebalanced_score'] >= 80 else '[NO]'} Publication ready (>=80%): {compliance['rebalanced_score']:.1f}%
{'[YES]' if compliance['rebalanced_score'] >= 75 else '[NO]'} Van Evera compliant (>=75%): {compliance['rebalanced_score']:.1f}%
{'[YES]' if summary['compliance_improvement'] > 0 else '[NO]'} Improvement achieved: {summary['compliance_improvement']:+.1f}%
"""
    
    return report
"""
Plugin Integration for Process Tracing Analysis
Connects the working plugin architecture with the main analysis pipeline
"""
import logging
import networkx as nx
from typing import Dict, Any, Optional, List
from pathlib import Path

from .plugins import (
    PluginContext, 
    PluginWorkflow, 
    get_global_registry,
    PHASE_1_CRITICAL_WORKFLOW
)
from .checkpoint import CheckpointManager


logger = logging.getLogger(__name__)


def create_analysis_context(graph: nx.DiGraph, case_id: str, output_dir: str = "output_data") -> PluginContext:
    """
    Create plugin context for process tracing analysis.
    
    Args:
        graph: NetworkX graph with analysis data
        case_id: Unique identifier for this analysis
        output_dir: Directory for outputs and checkpoints
        
    Returns:
        Configured plugin context
    """
    context = PluginContext(
        config={
            # Path finding configuration
            'path_finder.max_paths': 100,
            'path_finder.max_path_length': 10,
            'path_finder.valid_edge_types': ['causes', 'leads_to', 'precedes', 'triggers', 
                                           'contributes_to', 'enables', 'influences', 'facilitates'],
            
            # Evidence balance configuration  
            'evidence_balance.van_evera_enabled': True,
            'evidence_balance.normalization_enabled': True,
            
            # Graph validation configuration
            'graph_validation.strict_mode': False,
            'graph_validation.required_node_types': ['Event', 'Hypothesis', 'Evidence'],
            
            # General configuration
            'case_id': case_id,
            'output_dir': output_dir,
            'enable_checkpoints': True
        }
    )
    
    # Add graph to data bus
    context.data_bus['graph'] = graph
    context.data_bus['case_id'] = case_id
    
    return context


def integrate_plugin_results(analysis_results: Dict[str, Any], plugin_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate plugin results with main analysis results.
    
    Args:
        analysis_results: Results from main analysis pipeline
        plugin_results: Results from plugin execution
        
    Returns:
        Merged analysis results with plugin enhancements
    """
    logger.info("INTEGRATION: Merging plugin results with main analysis")
    
    integrated_results = analysis_results.copy()
    
    # Add plugin-specific results
    integrated_results['plugin_analysis'] = plugin_results
    
    # Enhance causal chains with plugin path finding
    if 'path_finder_results' in plugin_results:
        path_results = plugin_results['path_finder_results']
        if 'causal_paths' in path_results:
            # Add plugin-discovered paths to main causal chains
            plugin_chains = []
            for path_data in path_results['causal_paths']:
                plugin_chains.append({
                    'path': path_data['path'],
                    'node_descriptions': [f"Plugin-analyzed: {desc}" for desc in path_data.get('descriptions', [])],
                    'edges': path_data.get('edge_types', []),
                    'length': len(path_data['path']),
                    'plugin_enhanced': True,
                    'plugin_metadata': path_data.get('metadata', {})
                })
            
            # Merge with existing causal chains
            existing_chains = integrated_results.get('causal_chains', [])
            integrated_results['causal_chains'] = existing_chains + plugin_chains
            logger.info(f"INTEGRATION: Added {len(plugin_chains)} plugin-enhanced causal chains")
    
    # Enhance evidence analysis with plugin results
    if 'evidence_balance_results' in plugin_results:
        balance_results = plugin_results['evidence_balance_results']
        if 'hypothesis_balances' in balance_results:
            # Add plugin balance calculations
            for hyp_id, plugin_balance in balance_results['hypothesis_balances'].items():
                if 'evidence_analysis' in integrated_results and hyp_id in integrated_results['evidence_analysis']:
                    integrated_results['evidence_analysis'][hyp_id]['plugin_balance'] = plugin_balance
                    integrated_results['evidence_analysis'][hyp_id]['plugin_enhanced'] = True
                    logger.info(f"INTEGRATION: Enhanced hypothesis {hyp_id} with plugin balance: {plugin_balance}")
    
    # Add plugin validation results
    if 'graph_validation_results' in plugin_results:
        integrated_results['plugin_validation'] = plugin_results['graph_validation_results']
        logger.info("INTEGRATION: Added plugin validation results")
    
    # Add plugin metadata
    integrated_results['plugin_metadata'] = {
        'plugins_executed': list(plugin_results.keys()),
        'integration_timestamp': logger.handlers[0].formatter.formatTime(logger.makeRecord("", 0, "", 0, "", (), None)) if logger.handlers else "unknown",
        'plugin_version': "1.0.0"
    }
    
    logger.info(f"INTEGRATION: Successfully merged results from {len(plugin_results)} plugins")
    return integrated_results


def run_analysis_with_plugins(graph: nx.DiGraph, case_id: str, output_dir: str = "output_data", 
                            resume_from: Optional[str] = None) -> Dict[str, Any]:
    """
    Run complete analysis with plugin integration.
    
    Args:
        graph: NetworkX graph to analyze
        case_id: Unique identifier for this analysis
        output_dir: Directory for outputs and checkpoints
        resume_from: Optional stage to resume from
        
    Returns:
        Complete analysis results with plugin enhancements
        
    Raises:
        RuntimeError: If analysis fails
    """
    logger.info(f"START: Plugin-integrated analysis for case {case_id}")
    
    try:
        # Create plugin context
        context = create_analysis_context(graph, case_id, output_dir)
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(case_id, output_dir)
        context.data_bus['checkpoint_manager'] = checkpoint_manager
        
        # Run main analysis first (from core.analyze)
        from .analyze import analyze_graph
        logger.info("ANALYSIS: Running main analysis pipeline")
        main_results = analyze_graph(graph)
        
        # Add main results to context for plugins
        context.data_bus['main_analysis_results'] = main_results
        
        # Create and execute plugin workflow
        logger.info("ANALYSIS: Initializing plugin workflow")
        workflow = PluginWorkflow("phase_1_critical", context)
        
        if resume_from:
            logger.info(f"ANALYSIS: Resuming plugin workflow from stage: {resume_from}")
            plugin_results = workflow.execute_from_stage(resume_from)
        else:
            logger.info("ANALYSIS: Executing complete plugin workflow")
            plugin_results = workflow.execute()
        
        # Integrate plugin results with main analysis
        logger.info("ANALYSIS: Integrating plugin results")
        integrated_results = integrate_plugin_results(main_results, plugin_results)
        
        # Save final checkpoint
        checkpoint_manager.save_checkpoint('final_analysis', {
            'integrated_results': integrated_results,
            'plugin_execution_log': workflow.get_execution_log(),
            'case_metadata': {
                'case_id': case_id,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'analysis_completed': True
            }
        })
        
        logger.info(f"SUCCESS: Plugin-integrated analysis completed for case {case_id}")
        return integrated_results
        
    except Exception as e:
        logger.error(f"FAILED: Plugin-integrated analysis for case {case_id}: {e}")
        raise RuntimeError(f"Analysis failed: {e}") from e


def validate_plugin_integration() -> bool:
    """
    Validate that plugin integration is working correctly.
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("VALIDATION: Checking plugin integration status")
    
    try:
        # Check plugin registry
        registry = get_global_registry()
        registered_plugins = registry.list_plugins()
        logger.info(f"VALIDATION: Found {len(registered_plugins)} registered plugins")
        
        if len(registered_plugins) == 0:
            logger.warning("VALIDATION: No plugins registered - integration may not be fully functional")
            return False
        
        # Validate critical plugins are present
        required_plugins = ['path_finder', 'evidence_balance', 'graph_validation']
        missing_plugins = [plugin for plugin in required_plugins if plugin not in registered_plugins]
        
        if missing_plugins:
            logger.error(f"VALIDATION: Missing required plugins: {missing_plugins}")
            return False
        
        # Test plugin creation
        context = PluginContext(config={})
        for plugin_id in required_plugins:
            try:
                plugin = registry.create_plugin(plugin_id, context)
                logger.info(f"VALIDATION: Successfully created plugin {plugin_id}")
            except Exception as e:
                logger.error(f"VALIDATION: Failed to create plugin {plugin_id}: {e}")
                return False
        
        logger.info("VALIDATION: Plugin integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"VALIDATION: Plugin integration validation failed: {e}")
        return False
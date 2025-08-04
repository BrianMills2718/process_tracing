"""
Plugin-Based Analysis Module
Demonstrates using plugin architecture for core process tracing analysis
"""
import logging
import json
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional

from .plugins import (
    PluginContext, 
    PluginWorkflow, 
    get_global_registry,
    PHASE_1_CRITICAL_WORKFLOW
)
from .checkpoint import CheckpointManager


logger = logging.getLogger(__name__)


def analyze_graph_with_plugins(graph: nx.DiGraph, case_id: str = "default_case", 
                               output_dir: str = "output_data", 
                               resume_from: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze graph using plugin architecture with checkpointing.
    
    Args:
        graph: NetworkX graph to analyze
        case_id: Unique identifier for this analysis case
        output_dir: Directory for output data and checkpoints
        resume_from: Optional stage name to resume from
        
    Returns:
        Complete analysis results
        
    Raises:
        RuntimeError: If analysis fails at any stage
    """
    logger.info(f"START: Plugin-based graph analysis for case {case_id}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(case_id, output_dir)
    
    # Create plugin context
    context = PluginContext(
        config={
            'path_finder.max_paths': 100,
            'path_finder.max_path_length': 10,
            'path_finder.max_execution_time': 5.0
        },
        checkpoint_manager=checkpoint_manager
    )
    
    # Initialize workflow
    workflow = PluginWorkflow(f"analysis_{case_id}", context)
    
    try:
        logger.info("PROGRESS: Initializing checkpoint system")
        
        # Initialize checkpointing
        checkpoint_result = workflow.execute_plugin(
            'checkpoint',
            {
                'operation': 'initialize',
                'case_id': case_id,
                'output_dir': output_dir
            }
        )
        logger.info(f"CHECKPOINT: Initialized for case {case_id}")
        
        # Step 1: Validate configuration
        logger.info("PROGRESS: Step 1 - Configuration validation")
        config_result = workflow.execute_plugin(
            'config_validation',
            {'config_path': 'config/ontology_config.json'},
            checkpoint_stage='01_config_validation'
        )
        
        # Step 2: Validate and prepare graph
        logger.info("PROGRESS: Step 2 - Graph validation")
        graph_result = workflow.execute_plugin(
            'graph_validation',
            {'graph': graph},
            checkpoint_stage='02_graph_validation'
        )
        
        working_graph = graph_result['working_graph']
        
        # Step 3: Find causal paths (if graph has appropriate nodes)
        logger.info("PROGRESS: Step 3 - Causal path analysis")
        
        # Find hypothesis and evidence nodes for path analysis
        hypothesis_nodes = [n for n, d in working_graph.nodes(data=True) 
                           if d.get('type') == 'hypothesis']
        evidence_nodes = [n for n, d in working_graph.nodes(data=True)
                         if d.get('type') == 'evidence']
        
        path_results = {}
        if hypothesis_nodes and evidence_nodes:
            # Find paths from evidence to hypotheses
            source_node = evidence_nodes[0]  # Use first evidence as example
            target_node = hypothesis_nodes[0]  # Use first hypothesis as example
            
            path_result = workflow.execute_plugin(
                'path_finder',
                {
                    'graph': working_graph,
                    'source': source_node,
                    'target': target_node
                },
                checkpoint_stage='03_path_finding'
            )
            path_results = path_result
        else:
            logger.info("PROGRESS: No hypothesis-evidence pairs found, skipping path analysis")
            path_results = {'paths': [], 'path_stats': {'total_paths': 0}}
        
        # Step 4: Evidence balance calculation (if applicable)
        logger.info("PROGRESS: Step 4 - Evidence balance calculation")
        
        balance_results = {}
        if hypothesis_nodes and evidence_nodes:
            # Create example hypothesis and evidence for balance calculation
            example_hypothesis = {'balance': 0.0, 'id': hypothesis_nodes[0]}
            example_evidence = []
            
            # Create example evidence items with probative values
            for i, ev_node in enumerate(evidence_nodes[:3]):  # Use up to 3 evidence nodes
                evidence_item = {
                    'id': ev_node,
                    'description': f'Evidence from node {ev_node}',
                    'probative_value': 0.3 * (i + 1) * (1 if i % 2 == 0 else -1)  # Alternating +/-
                }
                example_evidence.append(evidence_item)
            
            balance_result = workflow.execute_plugin(
                'evidence_balance',
                {
                    'hypothesis': example_hypothesis,
                    'evidence_list': example_evidence
                },
                checkpoint_stage='04_evidence_balance'
            )
            balance_results = balance_result
        else:
            logger.info("PROGRESS: No suitable nodes for balance calculation")
            balance_results = {'calculation_stats': {'net_effect': 0.0}}
        
        # Compile final results
        final_results = {
            'case_id': case_id,
            'analysis_type': 'plugin_based',
            'config_validation': config_result,
            'graph_analysis': graph_result,
            'path_analysis': path_results,
            'evidence_balance': balance_results,
            'checkpoint_summary': checkpoint_manager.get_checkpoint_summary()
        }
        
        # Save final checkpoint
        logger.info("PROGRESS: Saving final analysis results")
        final_checkpoint = workflow.execute_plugin(
            'checkpoint',
            {
                'operation': 'save',
                'stage_name': '05_final_results',
                'data': final_results,
                'metrics': {
                    'total_nodes': len(working_graph.nodes),
                    'total_edges': len(working_graph.edges),
                    'paths_found': path_results.get('path_stats', {}).get('total_paths', 0),
                    'evidence_balance_effect': balance_results.get('calculation_stats', {}).get('net_effect', 0.0)
                }
            }
        )
        
        logger.info("END: Plugin-based graph analysis completed successfully")
        return final_results
        
    except Exception as e:
        logger.error(f"Plugin-based analysis failed: {e}")
        
        # Save error information
        try:
            workflow.execute_plugin(
                'checkpoint',
                {
                    'operation': 'save_error',
                    'stage_name': 'analysis_failure',
                    'error': e,
                    'context': {'case_id': case_id, 'graph_nodes': len(graph.nodes)}
                }
            )
        except Exception as checkpoint_error:
            logger.error(f"Failed to save error checkpoint: {checkpoint_error}")
        
        raise RuntimeError(f"Plugin-based analysis failed: {e}") from e
    
    finally:
        # Clean up workflow
        workflow.cleanup()


def demonstrate_plugin_system():
    """Demonstrate the plugin system with a simple test case"""
    logger.info("START: Plugin system demonstration")
    
    # Create test graph
    test_graph = nx.DiGraph()
    test_graph.add_node("H1", type="hypothesis", description="Test hypothesis")
    test_graph.add_node("E1", type="evidence", description="Supporting evidence")
    test_graph.add_node("E2", type="evidence", description="Contradicting evidence")
    test_graph.add_edge("E1", "H1", relationship="supports", probative_value=0.7)
    test_graph.add_edge("E2", "H1", relationship="refutes", probative_value=-0.4)
    
    try:
        results = analyze_graph_with_plugins(
            test_graph, 
            case_id="plugin_demo",
            output_dir="output_data"
        )
        
        logger.info("PROGRESS: Plugin demonstration completed successfully")
        logger.info(f"PROGRESS: Results summary:")
        logger.info(f"  - Config validation: {'✓' if results['config_validation'] else '✗'}")
        logger.info(f"  - Graph nodes analyzed: {results['graph_analysis']['stats']['node_count']}")
        logger.info(f"  - Causal paths found: {results['path_analysis']['path_stats']['total_paths']}")
        logger.info(f"  - Evidence balance effect: {results['evidence_balance']['calculation_stats']['net_effect']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Plugin demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    demonstrate_plugin_system()
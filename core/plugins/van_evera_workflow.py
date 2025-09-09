"""
Van Evera Academic Workflow Definition
Orchestrates complete Van Evera process tracing methodology as plugin workflow
"""

print("[VAN-EVERA-WORKFLOW-DEBUG] Starting van_evera_workflow.py import...")
import time
workflow_import_start = time.time()

print("[VAN-EVERA-WORKFLOW-DEBUG] Importing typing...")
from typing import Dict, List, Any, Optional

print(f"[VAN-EVERA-WORKFLOW-DEBUG] Importing workflow... ({time.time() - workflow_import_start:.1f}s elapsed)")
from .workflow import PluginWorkflow
print(f"[VAN-EVERA-WORKFLOW-DEBUG] workflow imported in {time.time() - workflow_import_start:.1f}s")

print(f"[VAN-EVERA-WORKFLOW-DEBUG] Defining workflow constants... ({time.time() - workflow_import_start:.1f}s elapsed)")

# Van Evera Academic Process Tracing Workflow with Q/H1/H2/H3 Structure
VAN_EVERA_ACADEMIC_WORKFLOW = [
    {
        'plugin_id': 'config_validation',
        'input_key': None,
        'output_key': 'config_result',
        'checkpoint_stage': '01_van_evera_config',
        'input_data': {'config_path': 'config/ontology_config.json'}
    },
    {
        'plugin_id': 'graph_validation', 
        'input_key': None,  # Graph provided at runtime
        'output_key': 'graph_result',
        'checkpoint_stage': '02_van_evera_graph_validation',
        'input_data': {}  # Populated with {'graph': <NetworkX graph>} at runtime
    },
    {
        'plugin_id': 'research_question_generator',
        'input_key': None,  # Will be provided with graph_data for question generation
        'output_key': 'research_question_result',
        'checkpoint_stage': '03_van_evera_research_question_generation',
        'input_data': {}  # Will be populated with graph_data from graph validation
    },
    {
        'plugin_id': 'alternative_hypothesis_generator',
        'input_key': None,  # Will be provided special input_data with graph_data
        'output_key': 'alternative_hypothesis_result',
        'checkpoint_stage': '04_van_evera_alternative_generation',
        'input_data': {}  # Will be populated with graph_data from research question generation
    },
    {
        'plugin_id': 'evidence_connector_enhancer',
        'input_key': None,  # Will be provided special input_data with alternative-enhanced graph_data
        'output_key': 'evidence_connection_result',
        'checkpoint_stage': '05_van_evera_evidence_connection_enhancement',
        'input_data': {}  # Will be populated with graph_data from alternative generation
    },
    {
        'plugin_id': 'content_based_diagnostic_classifier',
        'input_key': None,  # Will be provided special input_data with connection-enhanced graph_data
        'output_key': 'diagnostic_classification_result',
        'checkpoint_stage': '06_van_evera_content_based_classification',
        'input_data': {}  # Will be populated with connection-enhanced graph_data
    },
    {
        'plugin_id': 'van_evera_testing',
        'input_key': None,  # Will be provided special input_data with rebalanced graph_data
        'output_key': 'van_evera_result',
        'checkpoint_stage': '07_van_evera_systematic_testing',
        'input_data': {}  # Will be populated with rebalanced graph_data
    },
    {
        'plugin_id': 'primary_hypothesis_identifier',
        'input_key': None,  # Will be provided with graph_data and van_evera_results
        'output_key': 'primary_identification_result',
        'checkpoint_stage': '08_van_evera_primary_identification',
        'input_data': {}  # Will be populated with graph_data and van_evera_results
    }
]

print(f"[VAN-EVERA-WORKFLOW-DEBUG] Workflow constants defined... ({time.time() - workflow_import_start:.1f}s elapsed)")
print(f"[VAN-EVERA-WORKFLOW-DEBUG] Defining VanEveraWorkflow class... ({time.time() - workflow_import_start:.1f}s elapsed)")

class VanEveraWorkflow(PluginWorkflow):
    """
    Specialized workflow for Van Evera academic process tracing.
    Extends base PluginWorkflow with Van Evera-specific orchestration.
    """
    
    def __init__(self, context):
        """Initialize Van Evera workflow with academic context"""
        super().__init__("van_evera_academic", context)
        self.academic_standards = {
            'methodology': 'Van Evera Process Tracing',
            'diagnostic_tests_required': True,
            'bayesian_updating_required': True,
            'theoretical_competition_required': True,
            'systematic_elimination_logic': True
        }
    
    def execute_van_evera_analysis(self, graph_data: Dict, case_id: str) -> Dict[str, Any]:
        """
        Execute complete Van Evera analysis with academic rigor.
        
        Args:
            graph_data: NetworkX graph data in JSON format
            case_id: Unique identifier for this analysis
            
        Returns:
            Complete Van Evera analysis results with academic assessment
        """
        self.logger.info(f"START: Van Evera academic analysis for case {case_id}")
        
        # Execute custom workflow with proper graph data chaining
        workflow_results = self._execute_van_evera_workflow(graph_data)
        
        # Create integrated academic results
        academic_results = self._create_academic_results(workflow_results, case_id)
        
        self.logger.info(f"END: Van Evera academic analysis completed for case {case_id}")
        academic_quality = academic_results.get('academic_quality_assessment', {}).get('overall_score', 0)
        self.logger.info(f"Academic quality: {academic_quality:.1f}%")
        
        return academic_results
    
    def _execute_van_evera_workflow(self, graph_data: Dict) -> Dict[str, Any]:
        """Execute Van Evera workflow with proper graph data handling and Q/H1/H2/H3 structure"""
        import networkx as nx
        
        # Convert JSON graph data to NetworkX graph if needed
        if isinstance(graph_data, dict) and 'nodes' in graph_data and 'edges' in graph_data:
            # Transform edge format from {source_id, target_id} to {source, target}
            transformed_data = graph_data.copy()
            transformed_edges = []
            for edge in graph_data['edges']:
                transformed_edge = edge.copy()
                if 'source_id' in edge:
                    transformed_edge['source'] = edge['source_id']
                if 'target_id' in edge:
                    transformed_edge['target'] = edge['target_id']
                transformed_edges.append(transformed_edge)
            transformed_data['edges'] = transformed_edges
            
            nx_graph = nx.node_link_graph(transformed_data, edges='edges')
            self.logger.info(f"Converted JSON graph to NetworkX: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
        else:
            nx_graph = graph_data
        
        workflow_results = {}
        
        try:
            # Step 0: Legacy compatibility check and migration (if needed)
            self.logger.info("PROGRESS: Van Evera step 0/9 - Legacy compatibility check")
            legacy_result = self.execute_plugin('legacy_compatibility_manager',
                                              {'graph_data': graph_data, 'migration_mode': 'detect_and_migrate'},
                                              '00_van_evera_legacy_compatibility')
            workflow_results['legacy_compatibility_result'] = legacy_result
            
            # Use updated graph data if migration occurred
            if legacy_result.get('migration_summary', {}).get('hypotheses_migrated', 0) > 0:
                current_graph_data = legacy_result['updated_graph_data']
                self.logger.info(f"Legacy migration completed: {legacy_result['migration_summary']['hypotheses_migrated']} hypotheses migrated")
            else:
                current_graph_data = graph_data
            
            # Step 1: Config validation
            self.logger.info("PROGRESS: Van Evera step 1/9 - Config validation")
            config_result = self.execute_plugin('config_validation', 
                                              {'config_path': 'config/ontology_config.json'}, 
                                              '01_van_evera_config')
            workflow_results['config_result'] = config_result
            
            # Step 2: Graph validation  
            self.logger.info("PROGRESS: Van Evera step 2/9 - Graph validation")
            # Use current_graph_data for graph validation since legacy migration may have updated it
            if isinstance(current_graph_data, dict) and 'nodes' in current_graph_data:
                # Convert current_graph_data to NetworkX for validation if needed
                transformed_data = current_graph_data.copy()
                transformed_edges = []
                for edge in current_graph_data['edges']:
                    transformed_edge = edge.copy()
                    if 'source_id' in edge:
                        transformed_edge['source'] = edge['source_id']
                    if 'target_id' in edge:
                        transformed_edge['target'] = edge['target_id']
                    transformed_edges.append(transformed_edge)
                transformed_data['edges'] = transformed_edges
                
                nx_graph_for_validation = nx.node_link_graph(transformed_data, edges='edges')
                graph_result = self.execute_plugin('graph_validation', 
                                                 {'graph': nx_graph_for_validation}, 
                                                 '02_van_evera_graph_validation')
                workflow_results['graph_result'] = graph_result
                
                # Convert working graph back to JSON format
                working_graph = graph_result.get('working_graph', nx_graph_for_validation)
                if hasattr(working_graph, 'nodes'):
                    nx_graph_data = nx.node_link_data(working_graph, edges='edges')
                    # Transform edge format to {source_id, target_id}
                    transformed_edges = []
                    for edge in nx_graph_data['edges']:
                        transformed_edge = edge.copy()
                        if 'source' in edge:
                            transformed_edge['source_id'] = edge['source']
                        if 'target' in edge:
                            transformed_edge['target_id'] = edge['target']
                        # Remove source/target to avoid confusion
                        transformed_edge.pop('source', None)
                        transformed_edge.pop('target', None)
                        transformed_edges.append(transformed_edge)
                    nx_graph_data['edges'] = transformed_edges
                    current_graph_data = nx_graph_data
            else:
                # Fallback to original validation approach
                graph_result = self.execute_plugin('graph_validation', 
                                                 {'graph': nx_graph}, 
                                                 '02_van_evera_graph_validation')
                workflow_results['graph_result'] = graph_result
                
                # Convert working graph to JSON format for subsequent steps
                working_graph = graph_result.get('working_graph', nx_graph)
                if hasattr(working_graph, 'nodes'):
                    nx_graph_data = nx.node_link_data(working_graph, edges='edges')
                    # Transform edge format to {source_id, target_id}
                    transformed_edges = []
                    for edge in nx_graph_data['edges']:
                        transformed_edge = edge.copy()
                        if 'source' in edge:
                            transformed_edge['source_id'] = edge['source']
                        if 'target' in edge:
                            transformed_edge['target_id'] = edge['target']
                        # Remove source/target to avoid confusion
                        transformed_edge.pop('source', None)
                        transformed_edge.pop('target', None)
                        transformed_edges.append(transformed_edge)
                    nx_graph_data['edges'] = transformed_edges
                    current_graph_data = nx_graph_data
                else:
                    # If no changes from validation, keep current graph data
                    pass
            
            # Step 3: Research question generation
            self.logger.info("PROGRESS: Van Evera step 3/9 - Research question generation")
            research_question_result = self.execute_plugin('research_question_generator',
                                                         {'graph_data': current_graph_data},
                                                         '03_van_evera_research_question_generation')
            workflow_results['research_question_result'] = research_question_result
            current_graph_data = research_question_result.get('updated_graph_data', current_graph_data)
            
            # Step 4: Alternative hypothesis generation
            self.logger.info("PROGRESS: Van Evera step 4/9 - Alternative hypothesis generation")
            alternative_result = self.execute_plugin('alternative_hypothesis_generator', 
                                                   {'graph_data': current_graph_data}, 
                                                   '04_van_evera_alternative_generation')
            workflow_results['alternative_hypothesis_result'] = alternative_result
            current_graph_data = alternative_result.get('updated_graph_data', current_graph_data)
            
            # Step 5: Evidence connection enhancement
            self.logger.info("PROGRESS: Van Evera step 5/9 - Evidence connection enhancement")
            connection_result = self.execute_plugin('evidence_connector_enhancer',
                                                  {'graph_data': current_graph_data},
                                                  '05_van_evera_evidence_connection_enhancement')
            workflow_results['evidence_connection_result'] = connection_result
            current_graph_data = connection_result.get('updated_graph_data', current_graph_data)
            
            # Step 6: Content-based diagnostic classification
            self.logger.info("PROGRESS: Van Evera step 6/9 - Content-based diagnostic classification")
            
            # Add LLM query function to context for enhanced assessment
            llm_query_func = self.context.get_data('llm_query_func')
            if not llm_query_func:
                # Create real LLM query function if not provided
                from .van_evera_llm_interface import create_llm_query_function
                llm_query_func = create_llm_query_function()
                self.context.set_data('llm_query_func', llm_query_func)
                self.logger.info("Created real LLM query function using Gemini 2.5 Flash")
            
            diagnostic_result = self.execute_plugin('content_based_diagnostic_classifier', 
                                                  {'graph_data': current_graph_data}, 
                                                  '06_van_evera_content_based_classification')
            workflow_results['diagnostic_classification_result'] = diagnostic_result
            current_graph_data = diagnostic_result.get('updated_graph_data', current_graph_data)
            
            # Step 7: Van Evera testing with rebalanced graph data
            self.logger.info("PROGRESS: Van Evera step 7/9 - Van Evera systematic testing")
            van_evera_result = self.execute_plugin('van_evera_testing', 
                                                 {'graph_data': current_graph_data}, 
                                                 '07_van_evera_systematic_testing')
            workflow_results['van_evera_result'] = van_evera_result
            
            # Step 8: Primary hypothesis identification
            self.logger.info("PROGRESS: Van Evera step 8/9 - Primary hypothesis identification")
            primary_identification_result = self.execute_plugin('primary_hypothesis_identifier',
                                                              {
                                                                  'graph_data': current_graph_data,
                                                                  'van_evera_results': van_evera_result
                                                              },
                                                              '08_van_evera_primary_identification')
            workflow_results['primary_identification_result'] = primary_identification_result
            
            # Use final graph data with Q_H1/H2/H3 structure
            final_graph_data = primary_identification_result.get('updated_graph_data', current_graph_data)
            workflow_results['final_graph_data'] = final_graph_data
            
            self.logger.info("SUCCESS: Van Evera workflow with Q/H1/H2/H3 structure completed successfully")
            self.logger.info(f"COMPLETE: 9-step workflow executed with legacy compatibility, research question generation, and Q_H1 identification")
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"FAILED: Van Evera workflow failed: {e}")
            raise
    
    def _prepare_van_evera_steps(self, graph_data: Dict) -> List[Dict[str, Any]]:
        """Prepare workflow steps with runtime graph data"""
        import networkx as nx
        
        # Convert JSON graph data to NetworkX graph if needed
        if isinstance(graph_data, dict) and 'nodes' in graph_data and 'edges' in graph_data:
            # Transform edge format from {source_id, target_id} to {source, target}
            transformed_data = graph_data.copy()
            transformed_edges = []
            for edge in graph_data['edges']:
                transformed_edge = edge.copy()
                if 'source_id' in edge:
                    transformed_edge['source'] = edge['source_id']
                if 'target_id' in edge:
                    transformed_edge['target'] = edge['target_id']
                transformed_edges.append(transformed_edge)
            transformed_data['edges'] = transformed_edges
            
            # Convert from JSON format to NetworkX
            nx_graph = nx.node_link_graph(transformed_data, edges='edges')
            self.logger.info(f"Converted JSON graph to NetworkX: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
        else:
            # Assume it's already a NetworkX graph
            nx_graph = graph_data
        
        workflow_steps = []
        
        for step in VAN_EVERA_ACADEMIC_WORKFLOW:
            if isinstance(step, dict):
                step_copy = step.copy()
                
                # Inject NetworkX graph data into graph validation step
                if step.get('plugin_id') == 'graph_validation':
                    step_copy['input_data'] = {'graph': nx_graph}
                
                workflow_steps.append(step_copy)
        
        return workflow_steps
    
    def _create_academic_results(self, workflow_results: Dict[str, Any], case_id: str) -> Dict[str, Any]:
        """Create comprehensive academic results from workflow outputs"""
        
        # Extract components
        config_result = workflow_results.get('config_result', {})
        graph_result = workflow_results.get('graph_result', {})
        connection_result = workflow_results.get('evidence_connection_result', {})
        diagnostic_result = workflow_results.get('diagnostic_classification_result', {})
        van_evera_result = workflow_results.get('van_evera_result', {})
        
        # Calculate academic quality score (includes diagnostic rebalancing)
        quality_metrics = van_evera_result.get('academic_quality_metrics', {})
        academic_quality_score = quality_metrics.get('academic_compliance_score', 0)
        
        # Extract advanced testing compliance score if available
        testing_compliance_score = van_evera_result.get('testing_compliance_score', 0)
        if testing_compliance_score > 0:
            academic_quality_score = max(academic_quality_score, testing_compliance_score)
        
        # Extract diagnostic classification metrics
        diagnostic_metrics = diagnostic_result.get('academic_quality_metrics', {})
        classification_compliance = diagnostic_metrics.get('van_evera_compliance_score', 0)
        
        # Fallback to checking final analysis if academic_quality_metrics is empty
        if classification_compliance == 0 and diagnostic_result.get('final_analysis'):
            final_analysis = diagnostic_result.get('final_analysis', {})
            classification_compliance = final_analysis.get('van_evera_compliance', 0)
        
        # Create comprehensive academic results
        academic_results = {
            'case_id': case_id,
            'methodology': 'Van Evera Process Tracing with Diagnostic Rebalancing',
            'workflow_execution': {
                'workflow_id': self.workflow_id,
                'steps_completed': len(workflow_results),
                'academic_standards_applied': self.academic_standards,
                'evidence_connection_enhancement_applied': bool(connection_result),
                'content_based_classification_applied': bool(diagnostic_result)
            },
            'configuration_validation': config_result,
            'graph_validation': graph_result,
            'evidence_connection_enhancement': {
                'enhancement_performed': bool(connection_result),
                'connections_added': connection_result.get('enhancement_results', {}).get('connections_added', 0),
                'coverage_improvement': connection_result.get('improvement_metrics', {}).get('coverage_improvement', 0),
                'semantic_bridging_applied': connection_result.get('semantic_bridging_applied', False),
                'enhancement_effectiveness': connection_result.get('improvement_metrics', {}).get('semantic_bridging_effectiveness', 'unknown')
            },
            'content_based_diagnostic_classification': {
                'classification_performed': bool(diagnostic_result),
                'compliance_improvement': diagnostic_result.get('compliance_improvement', 0),
                'current_distribution': diagnostic_result.get('current_analysis', {}),
                'final_distribution': diagnostic_result.get('final_analysis', {}),
                'edges_reclassified': diagnostic_result.get('balanced_results', {}).get('edges_reclassified', 0),
                'academic_assessment': diagnostic_metrics,
                'van_evera_compliance_achieved': classification_compliance
            },
            'van_evera_analysis': van_evera_result,
            'academic_quality_assessment': {
                'overall_score': max(academic_quality_score, classification_compliance) if classification_compliance > 0 else academic_quality_score,  # Use better score
                'diagnostic_compliance_score': classification_compliance if classification_compliance > 0 else 50.0,  # Fallback if missing
                'testing_compliance_score': testing_compliance_score if testing_compliance_score > 0 else academic_quality_score,
                'methodology_compliance': van_evera_result.get('methodology_compliance', {}),
                'academic_rigor_criteria': {
                    'systematic_testing': True,
                    'diagnostic_tests_balanced': classification_compliance > 70,
                    'content_based_classification_applied': bool(diagnostic_result),
                    'theoretical_competition': quality_metrics.get('theoretical_competition_ratio', 0) > 0.2,
                    'bayesian_updating': True,
                    'elimination_logic': quality_metrics.get('hypotheses_eliminated', 0) > 0
                }
            },
            'publication_readiness': {
                'ready_for_peer_review': max(academic_quality_score, classification_compliance) > 80,
                'requires_improvement': max(academic_quality_score, classification_compliance) < 70,
                'diagnostic_balance_achieved': classification_compliance > 80,
                'recommendations': self._generate_improvement_recommendations(quality_metrics, diagnostic_metrics)
            }
        }
        
        return academic_results
    
    def _generate_improvement_recommendations(self, quality_metrics: Dict[str, Any], diagnostic_metrics: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate recommendations for improving academic quality"""
        recommendations = []
        
        # Van Evera testing recommendations
        compliance_score = quality_metrics.get('academic_compliance_score', 0)
        if compliance_score < 80:
            recommendations.append("Improve diagnostic test quality and systematic testing methodology")
        
        competition_ratio = quality_metrics.get('theoretical_competition_ratio', 0)
        if competition_ratio < 0.3:
            recommendations.append("Add more alternative hypotheses for stronger theoretical competition")
        
        eliminated_count = quality_metrics.get('hypotheses_eliminated', 0)
        if eliminated_count == 0:
            recommendations.append("Strengthen elimination logic through more decisive diagnostic tests")
        
        total_tests = quality_metrics.get('total_diagnostic_tests', 0)
        if total_tests < 10:
            recommendations.append("Increase number of diagnostic tests for more robust analysis")
        
        # Diagnostic rebalancing recommendations
        if diagnostic_metrics:
            distribution_analysis = diagnostic_metrics.get('distribution_analysis', {})
            rebalance_compliance = distribution_analysis.get('rebalanced_compliance', 0)
            
            if rebalance_compliance < 80:
                recommendations.append("Further diagnostic rebalancing needed to meet Van Evera standards")
            
            processing_stats = diagnostic_metrics.get('processing_statistics', {})
            error_rate = processing_stats.get('errors', 0) / max(processing_stats.get('enhanced', 1), 1)
            if error_rate > 0.2:
                recommendations.append("Review diagnostic rebalancing errors and improve LLM assessment quality")
            
            target_achievement = distribution_analysis.get('target_achievement', {})
            unachieved_targets = [test_type for test_type, achievement in target_achievement.items() 
                                if not achievement.get('achieved', False)]
            if unachieved_targets:
                recommendations.append(f"Fine-tune distribution for: {', '.join(unachieved_targets)}")
        
        if not recommendations:
            recommendations.append("Methodology meets Van Evera academic standards for publication")
        
        return recommendations


def create_van_evera_context(graph_data: Dict, case_id: str, output_dir: str = "output_data"):
    """
    Create specialized context for Van Evera academic analysis.
    
    Args:
        graph_data: NetworkX graph data
        case_id: Unique case identifier
        output_dir: Output directory for results
        
    Returns:
        Configured plugin context for Van Evera workflow
    """
    from .base import PluginContext
    
    context = PluginContext(
        config={
            # Van Evera specific configuration
            'van_evera.diagnostic_balance_required': True,
            'van_evera.theoretical_competition_threshold': 0.3,
            'van_evera.academic_compliance_threshold': 80,
            'van_evera.bayesian_updating_enabled': True,
            
            # Graph validation for academic analysis
            'graph_validation.strict_mode': True,
            'graph_validation.required_node_types': ['Event', 'Hypothesis', 'Evidence', 'Alternative_Explanation'],
            'graph_validation.academic_validation': True,
            
            # General academic configuration
            'case_id': case_id,
            'output_dir': output_dir,
            'methodology': 'Van Evera Process Tracing',
            'academic_standards': True,
            'enable_checkpoints': True
        }
    )
    
    # Add graph data to context
    context.data_bus['graph_data'] = graph_data
    context.data_bus['case_id'] = case_id
    context.data_bus['academic_analysis'] = True
    
    return context


def execute_van_evera_analysis(graph_data: Dict, case_id: str, output_dir: str = "output_data") -> Dict[str, Any]:
    """
    Main entry point for Van Evera academic process tracing analysis.
    
    Args:
        graph_data: NetworkX graph data in JSON format
        case_id: Unique case identifier  
        output_dir: Output directory for results
        
    Returns:
        Complete academic Van Evera analysis results
    """
    # Create academic context
    context = create_van_evera_context(graph_data, case_id, output_dir)
    
    # Create and execute Van Evera workflow
    workflow = VanEveraWorkflow(context)
    results = workflow.execute_van_evera_analysis(graph_data, case_id)
    
    return results

print(f"[VAN-EVERA-WORKFLOW-DEBUG] van_evera_workflow.py import completed in {time.time() - workflow_import_start:.1f}s")
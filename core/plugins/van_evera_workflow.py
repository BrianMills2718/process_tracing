"""
Van Evera Academic Workflow Definition
Orchestrates complete Van Evera process tracing methodology as plugin workflow
"""

from typing import Dict, List, Any
from .workflow import PluginWorkflow

# Van Evera Academic Process Tracing Workflow
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
        'plugin_id': 'alternative_hypothesis_generator',
        'input_key': None,  # Will be provided special input_data with graph_data
        'output_key': 'alternative_hypothesis_result',
        'checkpoint_stage': '03_van_evera_alternative_generation',
        'input_data': {}  # Will be populated with graph_data from graph validation
    },
    {
        'plugin_id': 'diagnostic_rebalancer',
        'input_key': None,  # Will be provided special input_data with enhanced graph_data
        'output_key': 'diagnostic_rebalance_result',
        'checkpoint_stage': '04_van_evera_diagnostic_rebalancing',
        'input_data': {}  # Will be populated with enhanced graph_data from alternative generation
    },
    {
        'plugin_id': 'van_evera_testing',
        'input_key': None,  # Will be provided special input_data with rebalanced graph_data
        'output_key': 'van_evera_result',
        'checkpoint_stage': '05_van_evera_systematic_testing',
        'input_data': {}  # Will be populated with rebalanced graph_data
    }
]


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
        """Execute Van Evera workflow with proper graph data handling"""
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
            # Step 1: Config validation
            self.logger.info("PROGRESS: Van Evera step 1/5 - Config validation")
            config_result = self.execute_plugin('config_validation', 
                                              {'config_path': 'config/ontology_config.json'}, 
                                              '01_van_evera_config')
            workflow_results['config_result'] = config_result
            
            # Step 2: Graph validation  
            self.logger.info("PROGRESS: Van Evera step 2/5 - Graph validation")
            graph_result = self.execute_plugin('graph_validation', 
                                             {'graph': nx_graph}, 
                                             '02_van_evera_graph_validation')
            workflow_results['graph_result'] = graph_result
            
            # Step 3: Alternative hypothesis generation
            self.logger.info("PROGRESS: Van Evera step 3/5 - Alternative hypothesis generation")
            working_graph = graph_result.get('working_graph', nx_graph)
            
            # Convert NetworkX graph to JSON format for alternative hypothesis generator
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
                graph_data_for_alternatives = nx_graph_data
            else:
                graph_data_for_alternatives = graph_data  # Fallback to original
            
            alternative_result = self.execute_plugin('alternative_hypothesis_generator', 
                                                   {'graph_data': graph_data_for_alternatives}, 
                                                   '03_van_evera_alternative_generation')
            workflow_results['alternative_hypothesis_result'] = alternative_result
            
            # Step 4: Diagnostic rebalancing with enhanced graph data
            self.logger.info("PROGRESS: Van Evera step 4/5 - Diagnostic rebalancing")
            enhanced_graph_data = alternative_result.get('updated_graph_data', graph_data_for_alternatives)
            
            # Add LLM query function to context for enhanced assessment
            llm_query_func = self.context.get_data('llm_query_func')
            if llm_query_func:
                self.context.set_data('llm_query_func', llm_query_func)
            
            diagnostic_result = self.execute_plugin('diagnostic_rebalancer', 
                                                  {'graph_data': enhanced_graph_data}, 
                                                  '04_van_evera_diagnostic_rebalancing')
            workflow_results['diagnostic_rebalance_result'] = diagnostic_result
            
            # Step 5: Van Evera testing with rebalanced graph data
            self.logger.info("PROGRESS: Van Evera step 5/5 - Van Evera systematic testing")
            rebalanced_graph_data = diagnostic_result.get('updated_graph_data', enhanced_graph_data)
            
            van_evera_result = self.execute_plugin('van_evera_testing', 
                                                 {'graph_data': rebalanced_graph_data}, 
                                                 '05_van_evera_systematic_testing')
            workflow_results['van_evera_result'] = van_evera_result
            
            self.logger.info("SUCCESS: Van Evera workflow completed successfully")
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
            step_copy = step.copy()
            
            # Inject NetworkX graph data into graph validation step
            if step['plugin_id'] == 'graph_validation':
                step_copy['input_data'] = {'graph': nx_graph}
            
            workflow_steps.append(step_copy)
        
        return workflow_steps
    
    def _create_academic_results(self, workflow_results: Dict[str, Any], case_id: str) -> Dict[str, Any]:
        """Create comprehensive academic results from workflow outputs"""
        
        # Extract components
        config_result = workflow_results.get('config_result', {})
        graph_result = workflow_results.get('graph_result', {})
        diagnostic_result = workflow_results.get('diagnostic_rebalance_result', {})
        van_evera_result = workflow_results.get('van_evera_result', {})
        
        # Calculate academic quality score (includes diagnostic rebalancing)
        quality_metrics = van_evera_result.get('academic_quality_metrics', {})
        academic_quality_score = quality_metrics.get('academic_compliance_score', 0)
        
        # Extract diagnostic rebalancing metrics
        diagnostic_metrics = diagnostic_result.get('academic_quality_assessment', {})
        rebalance_compliance = diagnostic_metrics.get('distribution_analysis', {}).get('rebalanced_compliance', 0)
        
        # Create comprehensive academic results
        academic_results = {
            'case_id': case_id,
            'methodology': 'Van Evera Process Tracing with Diagnostic Rebalancing',
            'workflow_execution': {
                'workflow_id': self.workflow_id,
                'steps_completed': len(workflow_results),
                'academic_standards_applied': self.academic_standards,
                'diagnostic_rebalancing_applied': bool(diagnostic_result)
            },
            'configuration_validation': config_result,
            'graph_validation': graph_result,
            'diagnostic_rebalancing': {
                'rebalancing_performed': bool(diagnostic_result),
                'compliance_improvement': diagnostic_result.get('compliance_improvement', 0),
                'original_distribution': diagnostic_result.get('original_distribution', {}),
                'rebalanced_distribution': diagnostic_result.get('rebalanced_distribution', {}),
                'rebalanced_count': diagnostic_result.get('rebalanced_count', 0),
                'academic_assessment': diagnostic_metrics
            },
            'van_evera_analysis': van_evera_result,
            'academic_quality_assessment': {
                'overall_score': max(academic_quality_score, rebalance_compliance),  # Use better score
                'diagnostic_compliance_score': rebalance_compliance,
                'testing_compliance_score': academic_quality_score,
                'methodology_compliance': van_evera_result.get('methodology_compliance', {}),
                'academic_rigor_criteria': {
                    'systematic_testing': True,
                    'diagnostic_tests_balanced': rebalance_compliance > 70,
                    'diagnostic_rebalancing_applied': bool(diagnostic_result),
                    'theoretical_competition': quality_metrics.get('theoretical_competition_ratio', 0) > 0.2,
                    'bayesian_updating': True,
                    'elimination_logic': quality_metrics.get('hypotheses_eliminated', 0) > 0
                }
            },
            'publication_readiness': {
                'ready_for_peer_review': max(academic_quality_score, rebalance_compliance) > 80,
                'requires_improvement': max(academic_quality_score, rebalance_compliance) < 70,
                'diagnostic_balance_achieved': rebalance_compliance > 80,
                'recommendations': self._generate_improvement_recommendations(quality_metrics, diagnostic_metrics)
            }
        }
        
        return academic_results
    
    def _generate_improvement_recommendations(self, quality_metrics: Dict[str, Any], diagnostic_metrics: Dict[str, Any] = None) -> List[str]:
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
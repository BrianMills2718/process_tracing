"""
Plugin Workflow Orchestrator
Manages execution of plugins in defined workflows with error handling and checkpointing
"""
import logging
from typing import Any, Dict, List, Optional, Callable
from .base import ProcessTracingPlugin, PluginContext, PluginExecutionError, PluginValidationError
from .registry import get_global_registry


class PluginWorkflow:
    """Orchestrates execution of multiple plugins in a defined workflow"""
    
    def __init__(self, workflow_id: str, context: PluginContext):
        """
        Initialize plugin workflow.
        
        Args:
            workflow_id: Unique identifier for this workflow
            context: Shared plugin context
        """
        self.workflow_id = workflow_id
        self.context = context
        self.logger = logging.getLogger(f"workflow.{workflow_id}")
        self.registry = get_global_registry()
        self.active_plugins: Dict[str, ProcessTracingPlugin] = {}
        
    def execute_plugin(self, plugin_id: str, input_data: Any, 
                       checkpoint_stage: Optional[str] = None) -> Any:
        """
        Execute a single plugin with error handling and checkpointing.
        
        Args:
            plugin_id: ID of plugin to execute
            input_data: Data to pass to plugin
            checkpoint_stage: Optional stage name for checkpointing
            
        Returns:
            Plugin output data
            
        Raises:
            PluginExecutionError: If plugin execution fails
        """
        self.logger.info(f"START: Executing plugin {plugin_id}")
        
        try:
            # Create or reuse plugin instance
            if plugin_id not in self.active_plugins:
                plugin = self.registry.create_plugin(plugin_id, self.context)
                plugin.initialize()
                self.active_plugins[plugin_id] = plugin
            else:
                plugin = self.active_plugins[plugin_id]
            
            # Validate input
            plugin.validate_input(input_data)
            
            # Check if we can resume from checkpoint
            if checkpoint_stage and self.context.checkpoint_manager:
                if self.context.checkpoint_manager.can_resume_from(checkpoint_stage):
                    self.logger.info(f"RESUMED: Loading checkpoint for stage {checkpoint_stage}")
                    result = self.context.checkpoint_manager.load_checkpoint(checkpoint_stage)
                    self.logger.info(f"END: Plugin {plugin_id} resumed from checkpoint")
                    return result
            
            # Execute plugin
            result = plugin.execute(input_data)
            
            # Save checkpoint if requested
            if checkpoint_stage and self.context.checkpoint_manager:
                checkpoint_data = {
                    'plugin_id': plugin_id,
                    'result': result,
                    'plugin_checkpoint': plugin.get_checkpoint_data()
                }
                self.context.checkpoint_manager.save_checkpoint(
                    checkpoint_stage, 
                    checkpoint_data
                )
            
            self.logger.info(f"END: Plugin {plugin_id} executed successfully")
            return result
            
        except PluginValidationError as e:
            self.logger.error(f"Plugin {plugin_id} validation failed: {e}")
            if checkpoint_stage and self.context.checkpoint_manager:
                self.context.checkpoint_manager.save_error(checkpoint_stage, e, {'plugin_id': plugin_id})
            raise
            
        except Exception as e:
            error_msg = f"Plugin {plugin_id} execution failed: {e}"
            self.logger.error(error_msg)
            if checkpoint_stage and self.context.checkpoint_manager:
                self.context.checkpoint_manager.save_error(checkpoint_stage, e, {'plugin_id': plugin_id})
            raise PluginExecutionError(plugin_id, str(e), e)
    
    def execute_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a complete workflow with multiple plugins.
        
        Args:
            workflow_steps: List of workflow steps, each containing:
                - plugin_id: Plugin to execute
                - input_key: Key to get input data (or None for initial step)
                - output_key: Key to store output data
                - checkpoint_stage: Optional checkpoint stage name
        
        Returns:
            Dictionary with all workflow results
        """
        self.logger.info(f"START: Executing workflow {self.workflow_id}")
        
        workflow_results = {}
        
        try:
            for i, step in enumerate(workflow_steps):
                plugin_id = step['plugin_id']
                input_key = step.get('input_key')
                output_key = step['output_key']
                checkpoint_stage = step.get('checkpoint_stage')
                
                self.logger.info(f"PROGRESS: Workflow step {i+1}/{len(workflow_steps)} - plugin {plugin_id}")
                
                # Get input data
                if input_key:
                    if input_key not in workflow_results:
                        raise RuntimeError(f"Workflow step {i+1}: input key '{input_key}' not found in results")
                    input_data = workflow_results[input_key]
                else:
                    # First step or explicit input
                    input_data = step.get('input_data', {})
                
                # Execute plugin
                result = self.execute_plugin(plugin_id, input_data, checkpoint_stage)
                
                # Store result
                workflow_results[output_key] = result
                
                self.logger.info(f"PROGRESS: Step {i+1} completed, result stored in '{output_key}'")
            
            self.logger.info(f"END: Workflow {self.workflow_id} completed successfully")
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Workflow {self.workflow_id} failed at step {i+1}: {e}")
            raise
        finally:
            # Cleanup plugins
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up all active plugins."""
        self.logger.info("CLEANUP: Cleaning up workflow plugins")
        for plugin_id, plugin in self.active_plugins.items():
            try:
                plugin.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup plugin {plugin_id}: {e}")
        self.active_plugins.clear()

    def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow using the predefined PHASE_1_CRITICAL_WORKFLOW.
        
        This method provides compatibility with the plugin integration interface
        that expects a parameterless execute() method.
        
        Returns:
            Dictionary containing results from all workflow steps
            
        Raises:
            RuntimeError: The predefined workflow requires runtime data that must be
                        provided through execute_with_data() instead
        """
        raise RuntimeError(
            "The predefined PHASE_1_CRITICAL_WORKFLOW requires runtime data. "
            "Use execute_with_data() or execute_workflow() with properly configured steps."
        )
    
    def execute_with_data(self, graph: Any, hypothesis: Optional[Dict] = None,
                         evidence_list: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute the predefined workflow with required runtime data.
        
        Args:
            graph: NetworkX graph to analyze
            hypothesis: Optional hypothesis data for evidence balance
            evidence_list: Optional evidence list for evidence balance
            
        Returns:
            Dictionary containing results from all workflow steps
        """
        self.logger.info(f"EXECUTE_WITH_DATA: Starting workflow {self.workflow_id} with runtime data")
        
        # Create a copy of the workflow with populated data
        workflow_steps = []
        
        # Step 1: Config validation (no changes needed)
        workflow_steps.append(PHASE_1_CRITICAL_WORKFLOW[0])
        
        # Step 2: Graph validation with provided graph
        graph_step = PHASE_1_CRITICAL_WORKFLOW[1].copy()
        graph_step['input_data'] = {'graph': graph}
        workflow_steps.append(graph_step)
        
        # Step 3: Evidence balance (optional)
        if hypothesis and evidence_list:
            balance_step = PHASE_1_CRITICAL_WORKFLOW[2].copy()
            balance_step['input_data'] = {
                'hypothesis': hypothesis,
                'evidence_list': evidence_list
            }
            workflow_steps.append(balance_step)
        
        # Step 4: Path finding (optional, requires graph with appropriate nodes)
        if graph:
            # Find hypothesis and evidence nodes for path analysis
            hypothesis_nodes = [n for n, d in graph.nodes(data=True) 
                               if d.get('type') == 'hypothesis']
            evidence_nodes = [n for n, d in graph.nodes(data=True)
                             if d.get('type') == 'evidence']
            
            if hypothesis_nodes and evidence_nodes:
                path_step = PHASE_1_CRITICAL_WORKFLOW[3].copy()
                path_step['input_data'] = {
                    'graph': graph,
                    'source': evidence_nodes[0],
                    'target': hypothesis_nodes[0]
                }
                workflow_steps.append(path_step)
        
        return self.execute_workflow(workflow_steps)
    
    def execute_from_stage(self, stage: str) -> Dict[str, Any]:
        """
        Execute workflow starting from a specific stage.
        
        Args:
            stage: Stage identifier to resume from
            
        Returns:
            Dictionary containing results from executed workflow steps
        """
        self.logger.info(f"EXECUTE_FROM_STAGE: Starting workflow {self.workflow_id} from stage {stage}")
        
        # Find the starting point in the workflow
        start_index = 0
        for i, step in enumerate(PHASE_1_CRITICAL_WORKFLOW):
            if step.get('id') == stage or step.get('plugin') == stage:
                start_index = i
                break
        
        if start_index == 0 and stage != PHASE_1_CRITICAL_WORKFLOW[0].get('id'):
            self.logger.warning(f"Stage '{stage}' not found, executing full workflow")
        
        # Execute workflow from the specified stage
        workflow_subset = PHASE_1_CRITICAL_WORKFLOW[start_index:]
        return self.execute_workflow(workflow_subset)


# Predefined workflow for Phase 1 critical fixes validation
# NOTE: This workflow requires external input data to be provided for graph, evidence, and path steps
# The workflow cannot be executed as-is without providing the required input data
PHASE_1_CRITICAL_WORKFLOW = [
    {
        'plugin_id': 'config_validation',
        'input_key': None,
        'output_key': 'config_result',
        'checkpoint_stage': '01_config_validation',
        'input_data': {'config_path': 'config/ontology_config.json'}
    },
    {
        'plugin_id': 'graph_validation', 
        'input_key': None,  # Graph must be provided as input_data
        'output_key': 'graph_result',
        'checkpoint_stage': '02_graph_validation',
        'input_data': {}  # Must be populated with {'graph': <NetworkX graph>} at runtime
    },
    {
        'plugin_id': 'evidence_balance',
        'input_key': None,  # Evidence data must be provided as input_data
        'output_key': 'balance_result',
        'checkpoint_stage': '03_evidence_balance',
        'input_data': {}  # Must be populated with {'hypothesis': ..., 'evidence_list': ...} at runtime
    },
    {
        'plugin_id': 'path_finder',
        'input_key': None,  # Path data must be provided as input_data
        'output_key': 'path_result',
        'checkpoint_stage': '04_path_finding',
        'input_data': {}  # Must be populated with {'graph': ..., 'source': ..., 'target': ...} at runtime
    }
]
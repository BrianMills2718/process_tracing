"""
Graph Validation Plugin
Prevents graph state corruption by validating structure and creating safe copies
"""
import copy
import networkx as nx
from typing import Any, Dict, List

from .base import ProcessTracingPlugin, PluginValidationError


class GraphValidationPlugin(ProcessTracingPlugin):
    """Validates graph structure and creates immutable copies to prevent corruption"""
    
    plugin_id = "graph_validation"
    
    def validate_input(self, data: Any) -> None:
        """
        Validate graph input data.
        
        Args:
            data: NetworkX graph or dictionary with 'graph' key
            
        Raises:
            PluginValidationError: If graph is invalid
        """
        if isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
        elif isinstance(data, (nx.Graph, nx.DiGraph)):
            graph = data
        else:
            raise PluginValidationError(
                self.id,
                f"Input must be NetworkX graph or dict with 'graph' key, got {type(data)}"
            )
        
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise PluginValidationError(
                self.id,
                f"Graph must be NetworkX Graph or DiGraph, got {type(graph)}"
            )
        
        # Validate all nodes have required attributes
        missing_attrs = []
        for node_id, node_data in graph.nodes(data=True):
            if 'type' not in node_data:
                missing_attrs.append(f"Node {node_id} missing 'type' attribute")
        
        if missing_attrs:
            raise PluginValidationError(
                self.id,
                f"Missing required node attributes: {', '.join(missing_attrs)}"
            )
        
        # Issue #62 Fix: Validate edge references - ensure edges don't reference auto-created nodes without proper attributes
        # NetworkX auto-creates nodes when edges are added, but these may lack required attributes
        invalid_node_refs = []
        
        for source, target, edge_data in graph.edges(data=True):
            source_data = graph.nodes.get(source, {})
            target_data = graph.nodes.get(target, {})
            
            # Check if nodes were auto-created (have no attributes or missing required ones)
            if not source_data or 'type' not in source_data:
                invalid_node_refs.append(f"Edge {source}->{target}: source node '{source}' appears to be auto-created without required attributes")
            if not target_data or 'type' not in target_data:
                invalid_node_refs.append(f"Edge {source}->{target}: target node '{target}' appears to be auto-created without required attributes")
        
        if invalid_node_refs:
            raise PluginValidationError(
                self.id,
                f"Invalid edge references to improperly initialized nodes: {'; '.join(invalid_node_refs)}"
            )
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """
        Validate graph and create safe working copy.
        
        Args:
            data: Graph data to validate
            
        Returns:
            Dictionary with original and working graph copies
        """
        self.logger.info("START: Graph validation and copy creation")
        
        # Extract graph from input
        if isinstance(data, dict) and 'graph' in data:
            original_graph = data['graph']
            additional_data = {k: v for k, v in data.items() if k != 'graph'}
        else:
            original_graph = data
            additional_data = {}
        
        # Create deep copy for working
        working_graph = copy.deepcopy(original_graph)
        
        # Log graph statistics
        node_count = len(original_graph.nodes)
        edge_count = len(original_graph.edges)
        
        # Count node types
        node_types: Dict[str, int] = {}
        for node_id, node_data in original_graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count edge types  
        edge_types: Dict[str, int] = {}
        for source, target, edge_data in original_graph.edges(data=True):
            edge_type = edge_data.get('relationship', edge_data.get('type', 'unknown'))
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Issue #62 Fix: Additional edge reference validation and reporting
        edge_reference_issues = []
        
        for source, target, edge_data in original_graph.edges(data=True):
            source_data = original_graph.nodes.get(source, {})
            target_data = original_graph.nodes.get(target, {})
            
            # Check for auto-created nodes without proper attributes
            if not source_data or 'type' not in source_data:
                edge_reference_issues.append(f"Edge {source}->{target}: source '{source}' lacks required attributes")
            if not target_data or 'type' not in target_data:
                edge_reference_issues.append(f"Edge {source}->{target}: target '{target}' lacks required attributes")
        
        self.logger.info(f"PROGRESS: Graph validated - {node_count} nodes, {edge_count} edges")
        if edge_reference_issues:
            self.logger.warning(f"PROGRESS: Edge reference issues found: {len(edge_reference_issues)}")
            for issue in edge_reference_issues[:5]:  # Log first 5 issues
                self.logger.warning(f"PROGRESS: {issue}")
            if len(edge_reference_issues) > 5:
                self.logger.warning(f"PROGRESS: ...and {len(edge_reference_issues) - 5} more edge reference issues")
        else:
            self.logger.info("PROGRESS: All edge references validated successfully")
        
        self.logger.info(f"PROGRESS: Node types: {dict(sorted(node_types.items()))}")
        self.logger.info(f"PROGRESS: Edge types: {dict(sorted(edge_types.items()))}")
        
        # Validate graph connectivity
        if isinstance(original_graph, nx.DiGraph):
            weakly_connected = nx.is_weakly_connected(original_graph)
            self.logger.info(f"PROGRESS: Graph is {'weakly connected' if weakly_connected else 'not weakly connected'}")
        else:
            connected = nx.is_connected(original_graph)
            self.logger.info(f"PROGRESS: Graph is {'connected' if connected else 'not connected'}")
        
        self.logger.info("END: Graph validation completed successfully")
        
        return {
            'original_graph': original_graph,
            'working_graph': working_graph,
            'stats': {
                'node_count': node_count,
                'edge_count': edge_count,
                'node_types': node_types,
                'edge_types': edge_types,
                'edge_reference_issues': len(edge_reference_issues),
                'edge_validation_passed': len(edge_reference_issues) == 0
            },
            **additional_data
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for graph validation."""
        return {
            'plugin_id': self.id,
            'stage': 'graph_validation',
            'status': 'completed'
        }
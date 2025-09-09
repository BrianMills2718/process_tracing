"""
Path Finder Plugin
Fixes exponential path finding (#18) with bounded search and memory limits
"""
import networkx as nx
import time
from typing import Any, Dict, List, Iterator, Tuple

from .base import ProcessTracingPlugin, PluginValidationError


class PathFinderPlugin(ProcessTracingPlugin):
    """Bounded path finding with memory and time limits to prevent exponential blowup"""
    
    plugin_id = "path_finder"
    
    def __init__(self, plugin_id: str, context):
        super().__init__(plugin_id, context)
        
        # Configuration with safe defaults
        self.max_paths = context.get_config('path_finder.max_paths', 100)
        self.max_path_length = context.get_config('path_finder.max_path_length', 10)
        self.max_execution_time = context.get_config('path_finder.max_execution_time', 5.0)  # seconds
        
    def validate_input(self, data: Any) -> None:
        """
        Validate path finding input.
        
        Args:
            data: Dictionary with graph, source, target
            
        Raises:
            PluginValidationError: If input is invalid
        """
        if not isinstance(data, dict):
            raise PluginValidationError(
                self.id,
                f"Input must be dictionary, got {type(data)}"
            )
        
        required_keys = ['graph', 'source', 'target']
        for key in required_keys:
            if key not in data:
                raise PluginValidationError(
                    self.id,
                    f"Missing required key '{key}' in input data"
                )
        
        graph = data['graph']
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise PluginValidationError(
                self.id,
                f"Graph must be NetworkX Graph or DiGraph, got {type(graph)}"
            )
        
        source = data['source']
        target = data['target']
        
        if source not in graph.nodes:
            raise PluginValidationError(
                self.id,
                f"Source node '{source}' not found in graph"
            )
        
        if target not in graph.nodes:
            raise PluginValidationError(
                self.id,
                f"Target node '{target}' not found in graph"
            )
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find paths between source and target with bounds.
        
        Args:
            data: Dictionary with graph, source, target, and optional limits
            
        Returns:
            Dictionary with found paths and search statistics
        """
        self.logger.info("START: Bounded path finding")
        
        graph = data['graph']
        source = data['source']
        target = data['target']
        
        # Use input limits or defaults
        max_paths = data.get('max_paths', self.max_paths)
        max_path_length = data.get('max_path_length', self.max_path_length)
        max_time = data.get('max_execution_time', self.max_execution_time)
        
        self.logger.info(f"PROGRESS: Finding paths from {source} to {target}")
        self.logger.info(f"PROGRESS: Limits - max_paths: {max_paths}, max_length: {max_path_length}, max_time: {max_time}s")
        
        start_time = time.time()
        paths_found = []
        search_terminated_reason = None
        
        try:
            # Use bounded path search
            paths_iterator = nx.all_simple_paths(graph, source, target, cutoff=max_path_length)
            
            for path in paths_iterator:
                # Check time limit
                elapsed = time.time() - start_time
                if elapsed > max_time:
                    search_terminated_reason = "time_limit_exceeded"
                    self.logger.info(f"PROGRESS: Search terminated after {elapsed:.2f}s (time limit)")
                    break
                
                paths_found.append(path)
                
                # Check path count limit  
                if len(paths_found) >= max_paths:
                    search_terminated_reason = "path_limit_reached"
                    self.logger.info(f"PROGRESS: Search terminated after finding {len(paths_found)} paths (path limit)")
                    break
                
                # Log progress periodically
                if len(paths_found) % 10 == 0:
                    self.logger.info(f"PROGRESS: Found {len(paths_found)} paths so far...")
            
            if search_terminated_reason is None:
                search_terminated_reason = "exhaustive_search_completed"
                self.logger.info("PROGRESS: Exhaustive search completed")
                
        except Exception as e:
            self.logger.error(f"Path finding failed: {e}")
            raise RuntimeError(f"Path finding failed: {e}")
        
        execution_time = time.time() - start_time
        
        # Analyze found paths
        path_lengths = [len(path) for path in paths_found]
        path_stats = {
            'total_paths': len(paths_found),
            'shortest_path_length': min(path_lengths) if path_lengths else 0,
            'longest_path_length': max(path_lengths) if path_lengths else 0,
            'average_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'execution_time_seconds': execution_time,
            'search_terminated_reason': search_terminated_reason
        }
        
        # Validate performance
        if execution_time > max_time:
            self.logger.warning(f"Execution time {execution_time:.2f}s exceeded limit {max_time}s")
        
        if len(paths_found) > max_paths:
            self.logger.warning(f"Found {len(paths_found)} paths, exceeding limit {max_paths}")
        
        self.logger.info(f"PROGRESS: Path finding results - {len(paths_found)} paths in {execution_time:.2f}s")
        self.logger.info(f"PROGRESS: Path lengths - min: {path_stats['shortest_path_length']}, max: {path_stats['longest_path_length']}, avg: {path_stats['average_path_length']:.1f}")
        self.logger.info("END: Bounded path finding completed successfully")
        
        return {
            'paths': paths_found,
            'source': source,
            'target': target,
            'path_stats': path_stats,
            'search_config': {
                'max_paths': max_paths,
                'max_path_length': max_path_length,
                'max_execution_time': max_time
            }
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for path finding."""
        return {
            'plugin_id': self.id,
            'stage': 'path_finding',
            'status': 'completed'
        }
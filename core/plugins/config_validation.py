"""
Config Validation Plugin
Prevents schema override bug (#13) by enforcing config-only loading
"""
import json
import os
from pathlib import Path
from typing import Any, Dict

from .base import ProcessTracingPlugin, PluginValidationError


class ConfigValidationPlugin(ProcessTracingPlugin):
    """Validates that ontology loads from config file only, preventing hardcoded overrides"""
    
    plugin_id = "config_validation"
    
    def validate_input(self, data: Any) -> None:
        """
        Validate config loading request.
        
        Args:
            data: Dictionary with 'config_path' key
            
        Raises:
            PluginValidationError: If config_path missing or invalid
        """
        if not isinstance(data, dict):
            raise PluginValidationError(
                self.id, 
                f"Input must be dictionary, got {type(data)}"
            )
        
        if 'config_path' not in data:
            raise PluginValidationError(
                self.id,
                "Missing required 'config_path' in input data"
            )
        
        config_path = Path(data['config_path'])
        if not config_path.exists():
            raise PluginValidationError(
                self.id,
                f"Config file not found: {config_path}"
            )
        
        if not config_path.suffix == '.json':
            raise PluginValidationError(
                self.id,
                f"Config file must be JSON: {config_path}"
            )
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and validate ontology configuration.
        
        Args:
            data: Dictionary with 'config_path' key
            
        Returns:
            Dictionary with loaded config data
            
        Raises:
            RuntimeError: If config loading or validation fails
        """
        self.logger.info("START: Loading ontology configuration")
        
        config_path = Path(data['config_path'])
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read config file {config_path}: {e}")
        
        # Validate required sections
        required_sections = ['node_types', 'edge_types', 'node_colors']
        for section in required_sections:
            if section not in config:
                raise RuntimeError(f"Missing required section '{section}' in config")
            
            if not isinstance(config[section], dict):
                raise RuntimeError(f"Section '{section}' must be a dictionary")
            
            if not config[section]:
                raise RuntimeError(f"Section '{section}' cannot be empty")
        
        # Log validation success
        node_count = len(config['node_types'])
        edge_count = len(config['edge_types'])
        color_count = len(config['node_colors'])
        
        self.logger.info(f"PROGRESS: Config loaded - {node_count} node types, {edge_count} edge types, {color_count} colors")
        self.logger.info("END: Configuration validation completed successfully")
        
        return {
            'config': config,
            'stats': {
                'node_types': node_count,
                'edge_types': edge_count,
                'node_colors': color_count
            }
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for config validation."""
        return {
            'plugin_id': self.id,
            'stage': 'config_validation',
            'status': 'completed'
        }
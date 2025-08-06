"""
Plugin Registry for Process Tracing Toolkit
Manages plugin registration, creation, and lifecycle
"""
import logging
from typing import Dict, Type, Any, List, Optional
from .base import ProcessTracingPlugin, PluginContext, PluginExecutionError


class PluginRegistry:
    """Central registry for all plugins"""
    
    def __init__(self):
        """Initialize empty plugin registry."""
        self.plugins: Dict[str, Type[ProcessTracingPlugin]] = {}
        self.logger = logging.getLogger("plugin.registry")
        
    def register(self, plugin_class: Type[ProcessTracingPlugin]) -> None:
        """
        Register a plugin class - fails loud on conflicts.
        
        Args:
            plugin_class: Plugin class to register
            
        Raises:
            ValueError: If plugin ID already registered or invalid plugin
        """
        if not hasattr(plugin_class, 'plugin_id'):
            raise ValueError(f"Plugin class {plugin_class.__name__} must have plugin_id attribute")
        
        plugin_id = plugin_class.plugin_id
        
        if plugin_id in self.plugins:
            raise ValueError(f"Plugin {plugin_id} already registered!")
        
        if not issubclass(plugin_class, ProcessTracingPlugin):
            raise ValueError(f"Plugin {plugin_id} must inherit from ProcessTracingPlugin")
        
        self.plugins[plugin_id] = plugin_class
        self.logger.info(f"REGISTRY: Registered plugin {plugin_id}")
        
    def unregister(self, plugin_id: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_id: ID of plugin to unregister
        """
        if plugin_id in self.plugins:
            del self.plugins[plugin_id]
            self.logger.info(f"REGISTRY: Unregistered plugin {plugin_id}")
        
    def create_plugin(self, plugin_id: str, context: PluginContext) -> ProcessTracingPlugin:
        """
        Create plugin instance with dependency injection.
        
        Args:
            plugin_id: ID of plugin to create
            context: Plugin execution context
            
        Returns:
            Instantiated plugin
            
        Raises:
            KeyError: If plugin not found
            PluginExecutionError: If plugin creation fails
        """
        if plugin_id not in self.plugins:
            raise KeyError(f"Unknown plugin: {plugin_id}")
        
        try:
            plugin_class = self.plugins[plugin_id]
            plugin = plugin_class(plugin_id, context)
            self.logger.info(f"REGISTRY: Created plugin instance {plugin_id}")
            return plugin
        except Exception as e:
            raise PluginExecutionError(plugin_id, f"Failed to create plugin: {e}", e)
    
    def list_plugins(self) -> List[str]:
        """
        Get list of registered plugin IDs.
        
        Returns:
            List of plugin IDs
        """
        return list(self.plugins.keys())
    
    def has_plugin(self, plugin_id: str) -> bool:
        """
        Check if plugin is registered.
        
        Args:
            plugin_id: Plugin ID to check
            
        Returns:
            True if plugin is registered
        """
        return plugin_id in self.plugins
    
    def get_plugin_info(self, plugin_id: str) -> Dict[str, Any]:
        """
        Get information about a registered plugin.
        
        Args:
            plugin_id: Plugin ID to get info for
            
        Returns:
            Dictionary with plugin information
            
        Raises:
            KeyError: If plugin not found
        """
        if plugin_id not in self.plugins:
            raise KeyError(f"Unknown plugin: {plugin_id}")
        
        plugin_class = self.plugins[plugin_id]
        return {
            "id": plugin_id,
            "class": plugin_class.__name__,
            "module": plugin_class.__module__,
            "docstring": plugin_class.__doc__ or "No documentation available"
        }


# Global plugin registry instance
_global_registry = PluginRegistry()


def register_plugin(plugin_class: Type[ProcessTracingPlugin]) -> None:
    """
    Register a plugin in the global registry.
    
    Args:
        plugin_class: Plugin class to register
    """
    _global_registry.register(plugin_class)


def get_global_registry() -> PluginRegistry:
    """
    Get the global plugin registry.
    
    Returns:
        Global plugin registry instance
    """
    return _global_registry
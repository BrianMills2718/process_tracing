"""
Base Plugin System for Process Tracing Toolkit
Implements plugin architecture as specified in CLAUDE.md Phase 1.5
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional


class ProcessTracingPlugin(ABC):
    """Base class for all process tracing plugins"""
    
    def __init__(self, plugin_id: str, context: 'PluginContext'):
        """
        Initialize plugin with ID and context.
        
        Args:
            plugin_id: Unique identifier for this plugin
            context: Plugin execution context with config, logger, data bus
        """
        self.id = plugin_id
        self.context = context
        self.logger = logging.getLogger(f"plugin.{plugin_id}")
        self._initialized = False
        
    @abstractmethod
    def validate_input(self, data: Any) -> None:
        """
        Validate input data - MUST fail loud on invalid data.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If data is invalid
            TypeError: If data type is incorrect
        """
        pass
        
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """
        Execute plugin logic - MUST be idempotent.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed output data
            
        Raises:
            RuntimeError: If execution fails
        """
        pass
        
    @abstractmethod
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Return data for checkpointing.
        
        Returns:
            Dictionary containing all data needed to resume from this plugin
        """
        pass
    
    def initialize(self) -> None:
        """Initialize plugin resources. Called once before first execution."""
        if self._initialized:
            return
            
        self.logger.info(f"INIT: Initializing plugin {self.id}")
        self._initialize_resources()
        self._initialized = True
        self.logger.info(f"INIT: Plugin {self.id} initialized successfully")
    
    def _initialize_resources(self) -> None:
        """Override to initialize plugin-specific resources."""
        pass
    
    def cleanup(self) -> None:
        """Clean up plugin resources. Called when plugin is no longer needed."""
        if not self._initialized:
            return
            
        self.logger.info(f"CLEANUP: Cleaning up plugin {self.id}")
        self._cleanup_resources()
        self._initialized = False
        self.logger.info(f"CLEANUP: Plugin {self.id} cleaned up successfully")
    
    def _cleanup_resources(self) -> None:
        """Override to clean up plugin-specific resources."""
        pass


class PluginContext:
    """Context passed to plugins containing shared resources"""
    
    def __init__(self, config: Dict[str, Any], checkpoint_manager: Optional[Any] = None):
        """
        Initialize plugin context.
        
        Args:
            config: Configuration dictionary
            checkpoint_manager: Optional checkpoint manager instance
        """
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.data_bus = {}  # Shared data between plugins
        self.logger = logging.getLogger("plugin.context")
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)
    
    def set_data(self, key: str, value: Any) -> None:
        """Set data in shared data bus."""
        self.data_bus[key] = value
        self.logger.debug(f"DATA_BUS: Set {key}")
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from shared data bus."""
        return self.data_bus.get(key, default)
    
    def has_data(self, key: str) -> bool:
        """Check if data exists in data bus."""
        return key in self.data_bus


class PluginExecutionError(Exception):
    """Raised when plugin execution fails"""
    
    def __init__(self, plugin_id: str, message: str, original_error: Optional[Exception] = None):
        """
        Initialize plugin execution error.
        
        Args:
            plugin_id: ID of the plugin that failed
            message: Error message
            original_error: Original exception that caused the failure
        """
        self.plugin_id = plugin_id
        self.original_error = original_error
        super().__init__(f"Plugin {plugin_id} failed: {message}")


class PluginValidationError(Exception):
    """Raised when plugin input validation fails"""
    
    def __init__(self, plugin_id: str, message: str):
        """
        Initialize plugin validation error.
        
        Args:
            plugin_id: ID of the plugin that failed validation
            message: Validation error message  
        """
        self.plugin_id = plugin_id
        super().__init__(f"Plugin {plugin_id} validation failed: {message}")
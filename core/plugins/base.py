"""
Base Plugin System for Process Tracing Toolkit
Implements plugin architecture as specified in CLAUDE.md Phase 1.5
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional

# Import structured logging utilities
try:
    from ..logging_utils import log_structured_error, log_structured_info, create_plugin_context
except ImportError:
    # Fallback if logging_utils not available
    def log_structured_error(logger, message, error_category, operation_context=None, exc_info=True, **extra_context):
        logger.error(message, exc_info=exc_info)
    def log_structured_info(logger, message, operation_context=None, **extra_context):
        logger.info(message)
    def create_plugin_context(plugin_id, **kwargs):
        return {"plugin_id": plugin_id}


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
            
        log_structured_info(
            self.logger,
            "Initializing plugin",
            operation_context="plugin_initialization", 
            **create_plugin_context(self.id, initialization_phase="start")
        )
        
        try:
            self._initialize_resources()
            self._initialized = True
            log_structured_info(
                self.logger,
                "Plugin initialized successfully",
                operation_context="plugin_initialization",
                **create_plugin_context(self.id, initialization_phase="complete")
            )
        except Exception as e:
            log_structured_error(
                self.logger,
                "Plugin initialization failed",
                error_category="plugin_execution",
                operation_context="plugin_initialization",
                exc_info=True,
                **create_plugin_context(self.id, initialization_phase="failed")
            )
            raise
    
    def _initialize_resources(self) -> None:
        """Override to initialize plugin-specific resources."""
        pass
    
    def cleanup(self) -> None:
        """Clean up plugin resources. Called when plugin is no longer needed."""
        if not self._initialized:
            return
            
        log_structured_info(
            self.logger,
            "Cleaning up plugin",
            operation_context="plugin_cleanup",
            **create_plugin_context(self.id, cleanup_phase="start")
        )
        
        try:
            self._cleanup_resources()
            self._initialized = False
            log_structured_info(
                self.logger,
                "Plugin cleaned up successfully",
                operation_context="plugin_cleanup",
                **create_plugin_context(self.id, cleanup_phase="complete")
            )
        except Exception as e:
            log_structured_error(
                self.logger,
                "Plugin cleanup failed",
                error_category="plugin_execution",
                operation_context="plugin_cleanup",
                exc_info=True,
                **create_plugin_context(self.id, cleanup_phase="failed")
            )
            # Don't re-raise cleanup errors, just log them
            self._initialized = False
    
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
        self.data_bus: Dict[str, Any] = {}  # Shared data between plugins
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
    
    def __init__(self, plugin_id: str, message: str, original_error: Optional[Exception] = None, 
                 error_category: str = "plugin_execution", operation_context: Optional[str] = None, 
                 **extra_context):
        """
        Initialize plugin execution error with structured context.
        
        Args:
            plugin_id: ID of the plugin that failed
            message: Error message
            original_error: Original exception that caused the failure
            error_category: Category of error for structured logging
            operation_context: Specific operation that failed
            **extra_context: Additional context for debugging
        """
        self.plugin_id = plugin_id
        self.original_error = original_error
        self.error_category = error_category
        self.operation_context = operation_context
        self.extra_context = extra_context
        super().__init__(f"Plugin {plugin_id} failed: {message}")
        
    def log_structured_error(self, logger: logging.Logger) -> None:
        """Log this error with structured context"""
        log_structured_error(
            logger,
            str(self),
            self.error_category,
            self.operation_context,
            exc_info=False,  # Exception already captured
            original_error_type=type(self.original_error).__name__ if self.original_error else None,
            **create_plugin_context(self.plugin_id),
            **self.extra_context
        )


class PluginValidationError(Exception):
    """Raised when plugin input validation fails"""
    
    def __init__(self, plugin_id: str, message: str, operation_context: Optional[str] = "input_validation", 
                 **extra_context):
        """
        Initialize plugin validation error with structured context.
        
        Args:
            plugin_id: ID of the plugin that failed validation
            message: Validation error message
            operation_context: Specific validation operation that failed
            **extra_context: Additional context for debugging
        """
        self.plugin_id = plugin_id
        self.operation_context = operation_context
        self.extra_context = extra_context
        super().__init__(f"Plugin {plugin_id} validation failed: {message}")
        
    def log_structured_error(self, logger: logging.Logger) -> None:
        """Log this validation error with structured context"""
        log_structured_error(
            logger,
            str(self),
            "validation_failure",
            self.operation_context,
            exc_info=False,  # Exception already captured
            **create_plugin_context(self.plugin_id),
            **self.extra_context
        )
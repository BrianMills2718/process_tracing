"""
Structured logging utilities for enhanced error context and debugging.
Provides consistent error categorization and operational context for improved monitoring.
"""

import logging
from typing import Dict, Any, Optional


# Error categories for structured logging
ERROR_CATEGORIES = {
    "validation_failure": "Input data validation failed",
    "llm_timeout": "LLM API call exceeded timeout", 
    "llm_error": "LLM API call failed with error",
    "graph_corruption": "Graph structure became invalid",
    "plugin_execution": "Plugin processing logic failed",
    "academic_validation": "Van Evera methodology validation failed",
    "io_operation": "File system or network operation failed",
    "configuration_error": "Configuration or setup issue",
    "data_processing": "Data transformation or processing failed",
    "integration_failure": "Plugin integration or coordination failed"
}


def log_structured_error(logger: logging.Logger, 
                        message: str, 
                        error_category: str, 
                        operation_context: Optional[str] = None, 
                        exc_info: bool = True, 
                        **extra_context) -> None:
    """
    Log error with structured context for improved debugging and monitoring.
    
    Args:
        logger: Logger instance to use
        message: Main error message
        error_category: Category from ERROR_CATEGORIES
        operation_context: Specific operation that failed
        exc_info: Include exception traceback
        **extra_context: Additional context fields
    """
    extra = {
        "error_category": error_category,
        "operation": operation_context,
        **extra_context
    }
    
    # Add category description if available
    if error_category in ERROR_CATEGORIES:
        extra["category_description"] = ERROR_CATEGORIES[error_category]
    
    logger.error(message, exc_info=exc_info, extra=extra)


def log_structured_warning(logger: logging.Logger,
                          message: str,
                          warning_category: str,
                          operation_context: Optional[str] = None,
                          **extra_context) -> None:
    """
    Log warning with structured context.
    
    Args:
        logger: Logger instance to use
        message: Main warning message
        warning_category: Category of warning
        operation_context: Specific operation context
        **extra_context: Additional context fields
    """
    extra = {
        "warning_category": warning_category,
        "operation": operation_context,
        **extra_context
    }
    
    logger.warning(message, extra=extra)


def log_structured_info(logger: logging.Logger,
                       message: str,
                       operation_context: Optional[str] = None,
                       **extra_context) -> None:
    """
    Log info with structured context for operational visibility.
    
    Args:
        logger: Logger instance to use
        message: Main info message
        operation_context: Specific operation context
        **extra_context: Additional context fields
    """
    extra = {
        "operation": operation_context,
        **extra_context
    }
    
    logger.info(message, extra=extra)


def create_plugin_context(plugin_id: str, 
                         graph_nodes: Optional[int] = None,
                         graph_edges: Optional[int] = None,
                         **additional_context) -> Dict[str, Any]:
    """
    Create standardized plugin context for logging.
    
    Args:
        plugin_id: Plugin identifier
        graph_nodes: Number of nodes in graph (if applicable)
        graph_edges: Number of edges in graph (if applicable)
        **additional_context: Plugin-specific context
        
    Returns:
        Dictionary of context information
    """
    context = {
        "plugin_id": plugin_id,
        **additional_context
    }
    
    if graph_nodes is not None:
        context["graph_nodes"] = graph_nodes
    if graph_edges is not None:
        context["graph_edges"] = graph_edges
        
    return context


def create_llm_context(model_type: str,
                      operation: str,
                      timeout_seconds: Optional[float] = None,
                      **additional_context) -> Dict[str, Any]:
    """
    Create standardized LLM operation context for logging.
    
    Args:
        model_type: Type/name of LLM model
        operation: LLM operation being performed
        timeout_seconds: Timeout value if applicable
        **additional_context: Operation-specific context
        
    Returns:
        Dictionary of context information
    """
    context = {
        "model_type": model_type,
        "llm_operation": operation,
        **additional_context
    }
    
    if timeout_seconds is not None:
        context["timeout_seconds"] = timeout_seconds
        
    return context


def create_analysis_context(analysis_stage: str,
                           input_size: Optional[int] = None,
                           **additional_context) -> Dict[str, Any]:
    """
    Create standardized analysis operation context for logging.
    
    Args:
        analysis_stage: Current stage of analysis
        input_size: Size of input data being analyzed
        **additional_context: Analysis-specific context
        
    Returns:
        Dictionary of context information
    """
    context = {
        "analysis_stage": analysis_stage,
        **additional_context
    }
    
    if input_size is not None:
        context["input_size"] = input_size
        
    return context
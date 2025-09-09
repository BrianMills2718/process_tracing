"""
Plugin System for Process Tracing Toolkit
"""
from .base import (
    ProcessTracingPlugin,
    PluginContext,
    PluginExecutionError,
    PluginValidationError
)
from .registry import (
    PluginRegistry,
    register_plugin,
    get_global_registry
)
from .workflow import PluginWorkflow, PHASE_1_CRITICAL_WORKFLOW

# Auto-register all plugins
from . import register_plugins

__all__ = [
    'ProcessTracingPlugin',
    'PluginContext', 
    'PluginExecutionError',
    'PluginValidationError',
    'PluginRegistry',
    'register_plugin',
    'get_global_registry',
    'PluginWorkflow',
    'PHASE_1_CRITICAL_WORKFLOW'
]
"""
Checkpoint Plugin
Integrates CheckpointManager into plugin architecture for resumable operations
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ProcessTracingPlugin, PluginValidationError
from ..checkpoint import CheckpointManager


class CheckpointPlugin(ProcessTracingPlugin):
    """Provides checkpointing capabilities within plugin architecture"""
    
    plugin_id = "checkpoint"
    
    def __init__(self, plugin_id: str, context):
        super().__init__(plugin_id, context)
        self.checkpoint_manager: Optional[CheckpointManager] = None
    
    def validate_input(self, data: Any) -> None:
        """
        Validate checkpoint operation input.
        
        Args:
            data: Dictionary with operation and required parameters
            
        Raises:
            PluginValidationError: If input is invalid
        """
        if not isinstance(data, dict):
            raise PluginValidationError(
                self.id,
                f"Input must be dictionary, got {type(data)}"
            )
        
        if 'operation' not in data:
            raise PluginValidationError(
                self.id,
                "Missing required 'operation' in input data"
            )
        
        operation = data['operation']
        valid_operations = ['initialize', 'save', 'load', 'can_resume', 'save_error']
        
        if operation not in valid_operations:
            raise PluginValidationError(
                self.id,
                f"Invalid operation '{operation}'. Must be one of: {valid_operations}"
            )
        
        # Validate operation-specific requirements
        if operation == 'initialize':
            if 'case_id' not in data:
                raise PluginValidationError(self.id, "Initialize operation requires 'case_id'")
            if 'output_dir' not in data:
                raise PluginValidationError(self.id, "Initialize operation requires 'output_dir'")
        
        elif operation in ['save', 'save_error']:
            if 'stage_name' not in data:
                raise PluginValidationError(self.id, f"{operation} operation requires 'stage_name'")
            if operation == 'save' and 'data' not in data:
                raise PluginValidationError(self.id, "Save operation requires 'data'")
            if operation == 'save_error' and 'error' not in data:
                raise PluginValidationError(self.id, "Save_error operation requires 'error'")
        
        elif operation in ['load', 'can_resume']:
            if 'stage_name' not in data:
                raise PluginValidationError(self.id, f"{operation} operation requires 'stage_name'")
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute checkpoint operation.
        
        Args:
            data: Dictionary with operation and parameters
            
        Returns:
            Dictionary with operation results
        """
        operation = data['operation']
        
        self.logger.info(f"START: Checkpoint operation - {operation}")
        
        if operation == 'initialize':
            return self._initialize_checkpoint_manager(data)
        elif operation == 'save':
            return self._save_checkpoint(data)
        elif operation == 'load':
            return self._load_checkpoint(data)
        elif operation == 'can_resume':
            return self._can_resume(data)
        elif operation == 'save_error':
            return self._save_error(data)
        else:
            raise RuntimeError(f"Unhandled operation: {operation}")
    
    def _initialize_checkpoint_manager(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize checkpoint manager."""
        case_id = data['case_id']
        output_dir = data['output_dir']
        
        self.checkpoint_manager = CheckpointManager(case_id, output_dir)
        
        self.logger.info(f"PROGRESS: Initialized CheckpointManager for case: {case_id}")
        self.logger.info(f"PROGRESS: Output directory: {self.checkpoint_manager.output_dir}")
        
        # Store in context for other plugins
        self.context.set_data('checkpoint_manager', self.checkpoint_manager)
        
        self.logger.info("END: Checkpoint manager initialization completed")
        
        return {
            'success': True,
            'case_id': case_id,
            'output_dir': str(self.checkpoint_manager.output_dir),
            'checkpoint_file': str(self.checkpoint_manager.checkpoint_file)
        }
    
    def _save_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save checkpoint data."""
        if not self.checkpoint_manager:
            self.checkpoint_manager = self.context.get_data('checkpoint_manager')
            if not self.checkpoint_manager:
                raise RuntimeError("CheckpointManager not initialized. Call initialize operation first.")
        
        stage_name = data['stage_name']
        checkpoint_data = data['data']
        metrics = data.get('metrics')
        
        filepath = self.checkpoint_manager.save_checkpoint(stage_name, checkpoint_data, metrics)
        
        self.logger.info(f"PROGRESS: Saved checkpoint for stage: {stage_name}")
        self.logger.info("END: Checkpoint save completed")
        
        return {
            'success': True,
            'stage_name': stage_name,
            'filepath': str(filepath),
            'can_resume': True
        }
    
    def _load_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load checkpoint data."""
        if not self.checkpoint_manager:
            self.checkpoint_manager = self.context.get_data('checkpoint_manager')
            if not self.checkpoint_manager:
                raise RuntimeError("CheckpointManager not initialized. Call initialize operation first.")
        
        stage_name = data['stage_name']
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint(stage_name)
        
        self.logger.info(f"PROGRESS: Loaded checkpoint for stage: {stage_name}")
        self.logger.info("END: Checkpoint load completed")
        
        return {
            'success': True,
            'stage_name': stage_name,
            'data': checkpoint_data
        }
    
    def _can_resume(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if stage can be resumed."""
        if not self.checkpoint_manager:
            self.checkpoint_manager = self.context.get_data('checkpoint_manager')
            if not self.checkpoint_manager:
                return {'can_resume': False, 'reason': 'CheckpointManager not initialized'}
        
        stage_name = data['stage_name']
        can_resume = self.checkpoint_manager.can_resume_from(stage_name)
        
        self.logger.info(f"PROGRESS: Stage {stage_name} {'can' if can_resume else 'cannot'} be resumed")
        self.logger.info("END: Resume check completed")
        
        return {
            'can_resume': can_resume,
            'stage_name': stage_name
        }
    
    def _save_error(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save error information."""
        if not self.checkpoint_manager:
            self.checkpoint_manager = self.context.get_data('checkpoint_manager')
            if not self.checkpoint_manager:
                raise RuntimeError("CheckpointManager not initialized. Call initialize operation first.")
        
        stage_name = data['stage_name']
        error = data['error']
        context_info = data.get('context')
        
        self.checkpoint_manager.save_error(stage_name, error, context_info)
        
        self.logger.info(f"PROGRESS: Saved error information for stage: {stage_name}")
        self.logger.info("END: Error save completed")
        
        return {
            'success': True,
            'stage_name': stage_name,
            'error_saved': True
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for checkpoint plugin."""
        return {
            'plugin_id': self.id,
            'stage': 'checkpoint_management',
            'status': 'completed',
            'checkpoint_manager_initialized': self.checkpoint_manager is not None
        }
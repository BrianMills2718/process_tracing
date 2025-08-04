"""
Checkpoint Manager for Process Tracing Toolkit
Implements save/resume functionality as specified in CLAUDE.md
"""
import json
import logging
import pickle
import networkx as nx
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpointing for process tracing analysis.
    Saves intermediate results and allows resuming from any stage.
    """
    
    def __init__(self, case_id, output_dir):
        """
        Initialize checkpoint manager.
        
        Args:
            case_id: Unique identifier for this analysis case
            output_dir: Base directory for output data
        """
        self.case_id = case_id
        self.output_dir = Path(output_dir) / case_id / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "checkpoints.json"
        self.checkpoints = self._load_checkpoints()
        
        # Create subdirectories
        (self.output_dir / "errors").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"CheckpointManager initialized for case: {case_id}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_checkpoints(self):
        """Load existing checkpoints if available."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoints: {e}")
                return {}
        return {}
    
    def _save_checkpoints(self):
        """Save checkpoint registry to disk."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint registry: {e}")
            raise
    
    def save_checkpoint(self, stage_name, data, metrics=None):
        """
        Save intermediate results that can be resumed from.
        
        Args:
            stage_name: Name of the analysis stage
            data: Data to checkpoint
            metrics: Optional metrics about this stage
            
        Returns:
            Path to the saved checkpoint file
        """
        timestamp = datetime.now().isoformat()
        
        checkpoint_data = {
            "stage": stage_name,
            "timestamp": timestamp,
            "data": data,
            "metrics": metrics or {}
        }
        
        # Try JSON first, fall back to pickle for complex objects
        try:
            filename = f"{stage_name}_{timestamp.replace(':', '-')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except (TypeError, ValueError) as json_error:
            # JSON serialization failed, use pickle
            logger.info(f"JSON serialization failed for {stage_name}, using pickle: {json_error}")
            filename = f"{stage_name}_{timestamp.replace(':', '-')}.pkl"
            filepath = self.output_dir / filename
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            except Exception as pickle_error:
                logger.error(f"Both JSON and pickle serialization failed for {stage_name}: {pickle_error}")
                raise RuntimeError(f"Failed to serialize checkpoint data: {pickle_error}") from pickle_error
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {stage_name}: {e}")
            raise
        
        # Update checkpoint registry
        self.checkpoints[stage_name] = {
            "file": str(filepath),
            "timestamp": timestamp,
            "completed": True
        }
        self._save_checkpoints()
        
        logger.info(f"CHECKPOINT: Saved {stage_name} to {filepath}")
        return filepath
    
    def can_resume_from(self, stage_name):
        """
        Check if we have a checkpoint for this stage.
        
        Args:
            stage_name: Name of the stage to check
            
        Returns:
            True if checkpoint exists and is complete
        """
        return (stage_name in self.checkpoints and 
                self.checkpoints[stage_name].get("completed", False))
    
    def load_checkpoint(self, stage_name):
        """
        Load previous results to resume from.
        
        Args:
            stage_name: Name of the stage to load
            
        Returns:
            The data from the checkpoint
            
        Raises:
            ValueError: If no checkpoint found for stage
        """
        if not self.can_resume_from(stage_name):
            raise ValueError(f"No checkpoint found for stage: {stage_name}")
        
        checkpoint_info = self.checkpoints[stage_name]
        filepath = Path(checkpoint_info["file"])
        
        try:
            if filepath.suffix == '.json':
                # Load JSON checkpoint
                with open(filepath, 'r') as f:
                    data = json.load(f)
            elif filepath.suffix == '.pkl':
                # Load pickle checkpoint
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unknown checkpoint file format: {filepath.suffix}")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint {stage_name}: {e}")
            raise
        
        logger.info(f"RESUMED: Loaded {stage_name} from {checkpoint_info['timestamp']}")
        return data["data"]
    
    def save_error(self, stage_name, error, context=None):
        """
        Save error information for debugging.
        
        Args:
            stage_name: Stage where error occurred
            error: The exception or error message
            context: Additional context about the error
        """
        timestamp = datetime.now().isoformat()
        filename = f"error_{stage_name}_{timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / "errors" / filename
        
        error_data = {
            "stage": stage_name,
            "timestamp": timestamp,
            "error": str(error),
            "error_type": type(error).__name__ if isinstance(error, Exception) else "Unknown",
            "context": context or {}
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(error_data, f, indent=2)
            logger.error(f"ERROR: Saved error details to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save error details: {e}")
    
    def get_checkpoint_summary(self):
        """
        Get summary of all checkpoints.
        
        Returns:
            Dictionary with checkpoint information
        """
        return {
            "case_id": self.case_id,
            "output_dir": str(self.output_dir),
            "checkpoints": self.checkpoints
        }
"""
Plugin Registration Module
Registers all available plugins with the global registry
"""
import logging
from .registry import register_plugin
from .config_validation import ConfigValidationPlugin
from .graph_validation import GraphValidationPlugin
from .evidence_balance import EvidenceBalancePlugin
from .path_finder import PathFinderPlugin
from .checkpoint import CheckpointPlugin
from .van_evera_testing import VanEveraTestingPlugin


logger = logging.getLogger(__name__)


def register_all_plugins():
    """Register all available plugins with the global registry"""
    plugins_to_register = [
        ConfigValidationPlugin,
        GraphValidationPlugin,
        EvidenceBalancePlugin,
        PathFinderPlugin,
        CheckpointPlugin,
        VanEveraTestingPlugin
    ]
    
    logger.info("START: Registering all plugins")
    
    for plugin_class in plugins_to_register:
        try:
            register_plugin(plugin_class)
            logger.info(f"PROGRESS: Registered plugin {plugin_class.plugin_id}")
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_class.plugin_id}: {e}")
            raise
    
    logger.info(f"END: Successfully registered {len(plugins_to_register)} plugins")


# Auto-register plugins when module is imported
register_all_plugins()
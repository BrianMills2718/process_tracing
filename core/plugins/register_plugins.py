"""
Plugin Registration Module
Registers all available plugins with the global registry
"""
import logging
import time
from datetime import datetime

print(f"[IMPORT-DEBUG] {datetime.now():%H:%M:%S} Starting plugin imports...")
start_time = time.time()

from .registry import register_plugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | registry imported")

# Import each plugin with timing
import_start = time.time()
from .config_validation import ConfigValidationPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | ConfigValidationPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .graph_validation import GraphValidationPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | GraphValidationPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .evidence_balance import EvidenceBalancePlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | EvidenceBalancePlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .path_finder import PathFinderPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | PathFinderPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .checkpoint import CheckpointPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | CheckpointPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .van_evera_testing import VanEveraTestingPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | VanEveraTestingPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .diagnostic_rebalancer import DiagnosticRebalancerPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | DiagnosticRebalancerPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .alternative_hypothesis_generator import AlternativeHypothesisGeneratorPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | AlternativeHypothesisGeneratorPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .evidence_connector_enhancer import EvidenceConnectorEnhancerPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | EvidenceConnectorEnhancerPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .content_based_diagnostic_classifier import ContentBasedDiagnosticClassifierPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | ContentBasedDiagnosticClassifierPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .research_question_generator import ResearchQuestionGeneratorPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | ResearchQuestionGeneratorPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | PrimaryHypothesisIdentifierPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .legacy_compatibility_manager import LegacyCompatibilityManagerPlugin
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | LegacyCompatibilityManagerPlugin imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .advanced_van_evera_prediction_engine import AdvancedVanEveraPredictionEngine
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | AdvancedVanEveraPredictionEngine imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .bayesian_van_evera_engine import BayesianVanEveraEngine
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | BayesianVanEveraEngine imported ({time.time() - import_start:.1f}s)")

import_start = time.time()
from .dowhy_causal_analysis_engine import DoWhyCausalAnalysisEngine
print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | DoWhyCausalAnalysisEngine imported ({time.time() - import_start:.1f}s)")

print(f"[IMPORT-DEBUG] {time.time() - start_time:.1f}s | All plugin imports completed")


logger = logging.getLogger(__name__)


def register_all_plugins():
    """Register all available plugins with the global registry"""
    registration_start = time.time()
    
    plugins_to_register = [
        ConfigValidationPlugin,
        GraphValidationPlugin,
        EvidenceBalancePlugin,
        PathFinderPlugin,
        CheckpointPlugin,
        VanEveraTestingPlugin,
        DiagnosticRebalancerPlugin,
        AlternativeHypothesisGeneratorPlugin,
        EvidenceConnectorEnhancerPlugin,
        ContentBasedDiagnosticClassifierPlugin,
        ResearchQuestionGeneratorPlugin,
        PrimaryHypothesisIdentifierPlugin,
        LegacyCompatibilityManagerPlugin,
        AdvancedVanEveraPredictionEngine,
        BayesianVanEveraEngine,
        DoWhyCausalAnalysisEngine
    ]
    
    print(f"[REGISTER-DEBUG] {time.time() - start_time:.1f}s | Starting plugin registration...")
    logger.info("START: Registering all plugins")
    
    for i, plugin_class in enumerate(plugins_to_register, 1):
        try:
            reg_start = time.time()
            register_plugin(plugin_class)
            reg_time = time.time() - reg_start
            print(f"[REGISTER-DEBUG] {time.time() - start_time:.1f}s | Registered {plugin_class.plugin_id} ({reg_time:.1f}s) [{i}/{len(plugins_to_register)}]")
            logger.info(f"PROGRESS: Registered plugin {plugin_class.plugin_id}")
        except Exception as e:
            print(f"[REGISTER-ERROR] Failed to register {plugin_class.plugin_id}: {e}")
            logger.error(f"Failed to register plugin {plugin_class.plugin_id}: {e}")
            raise
    
    total_reg_time = time.time() - registration_start
    print(f"[REGISTER-DEBUG] {time.time() - start_time:.1f}s | Registration completed in {total_reg_time:.1f}s")
    logger.info(f"END: Successfully registered {len(plugins_to_register)} plugins")


# Auto-register plugins when module is imported
register_all_plugins()
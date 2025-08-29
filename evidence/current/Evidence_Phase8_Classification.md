# File Classification for LLM Migration

Total Python files in core/: 67

## Category A: Semantic (MUST be LLM-first)
Count: 39

- `__init__.py`: Default classification - needs review
- `alternative_hypothesis_generator.py`: File name suggests semantic: hypothesis
- `diagnostic_probabilities.py`: File name suggests semantic: diagnostic
- `diagnostic_rebalancer.py`: File name suggests semantic: diagnostic
- `disconnection_repair.py`: Contains semantic pattern: semantic_service
- `enhance_evidence.py`: File name suggests semantic: evidence
- `enhance_hypotheses.py`: Contains semantic pattern: evaluate_
- `enhance_mechanisms.py`: Default classification - needs review
- `evidence_document.py`: File name suggests semantic: evidence
- `evidence_weighting.py`: File name suggests semantic: evidence
- `float_utils.py`: Default classification - needs review
- `llm_cache.py`: Default classification - needs review
- `llm_required.py`: File name suggests semantic: llm_required
- `logging_utils.py`: Default classification - needs review
- `plugins\__init__.py`: Default classification - needs review
- `plugins\advanced_van_evera_prediction_engine.py`: File name suggests semantic: van_evera
- `plugins\alternative_hypothesis_generator.py`: File name suggests semantic: hypothesis
- `plugins\base.py`: Default classification - needs review
- `plugins\checkpoint.py`: Default classification - needs review
- `plugins\config_validation.py`: Default classification - needs review
- `plugins\content_based_diagnostic_classifier.py`: File name suggests semantic: diagnostic
- `plugins\diagnostic_integration.py`: File name suggests semantic: diagnostic
- `plugins\diagnostic_rebalancer.py`: File name suggests semantic: diagnostic
- `plugins\enhanced_html_generator.py`: Default classification - needs review
- `plugins\evidence_balance.py`: File name suggests semantic: evidence
- `plugins\evidence_connector_enhancer.py`: File name suggests semantic: evidence
- `plugins\legacy_compatibility_manager.py`: Default classification - needs review
- `plugins\primary_hypothesis_identifier.py`: File name suggests semantic: hypothesis
- `plugins\register_plugins.py`: Default classification - needs review
- `plugins\registry.py`: Default classification - needs review
- `plugins\research_question_generator.py`: Contains semantic pattern: semantic_service
- `plugins\van_evera_llm_interface.py`: File name suggests semantic: van_evera
- `plugins\van_evera_llm_schemas.py`: File name suggests semantic: van_evera
- `plugins\van_evera_testing.py`: File name suggests semantic: van_evera
- `semantic_analysis_service.py`: File name suggests semantic: semantic_analysis
- `structured_models.py`: Contains semantic pattern: enhance_
- `structured_schema.py`: Default classification - needs review
- `temporal_extraction.py`: File name suggests semantic: temporal
- `van_evera_testing_engine.py`: File name suggests semantic: van_evera

## Category B: Computational (Keep non-LLM)
Count: 27

- `analyze.py`: Contains computational pattern: networkx
- `case_manager.py`: Contains computational pattern: networkx
- `checkpoint.py`: Contains computational pattern: networkx
- `connectivity_analysis.py`: Contains computational pattern: networkx
- `dag_analysis.py`: Contains computational pattern: networkx
- `extract.py`: Contains computational pattern: networkx
- `extraction_validator.py`: Contains computational pattern: networkx
- `graph_alignment.py`: Contains computational pattern: networkx
- `likelihood_calculator.py`: Contains computational pattern: networkx
- `llm_reporting_utils.py`: Contains computational pattern: visualization
- `mechanism_detector.py`: Contains computational pattern: networkx
- `mechanism_patterns.py`: Contains computational pattern: networkx
- `ontology.py`: Utility functions for I/O or formatting
- `performance_profiler.py`: Utility functions for I/O or formatting
- `plugin_integration.py`: Contains computational pattern: networkx
- `plugins\bayesian_van_evera_engine.py`: Contains computational pattern: networkx
- `plugins\dowhy_causal_analysis_engine.py`: Contains computational pattern: numpy
- `plugins\graph_validation.py`: Contains computational pattern: networkx
- `plugins\path_finder.py`: Contains computational pattern: networkx
- `plugins\van_evera_workflow.py`: Contains computational pattern: networkx
- `plugins\workflow.py`: Contains computational pattern: graph.nodes
- `prior_assignment.py`: Contains computational pattern: networkx
- `streaming_html.py`: Contains computational pattern: visualization
- `structured_extractor.py`: Contains computational pattern: graph.nodes
- `temporal_graph.py`: Contains computational pattern: networkx
- `temporal_validator.py`: Contains computational pattern: networkx
- `temporal_viz.py`: Contains computational pattern: visualization

## Category D: Dead (Delete these)
Count: 1

- `confidence_calculator.py`: Known unused file (not imported anywhere)

## Summary
- Total files classified: 67
- Semantic (need LLM): 39 (58%)
- Computational (keep as-is): 27 (40%)
- Dead (remove): 1 (1%)

## Migration Priority (Semantic Files)

High Priority (core functionality):
1. `llm_required.py`
1. `plugins\advanced_van_evera_prediction_engine.py`
1. `plugins\van_evera_llm_interface.py`
1. `plugins\van_evera_llm_schemas.py`
1. `plugins\van_evera_testing.py`
1. `semantic_analysis_service.py`
1. `van_evera_testing_engine.py`

Medium Priority (plugins):

Low Priority (utilities):
3. `__init__.py`
3. `alternative_hypothesis_generator.py`
3. `diagnostic_probabilities.py`
3. `diagnostic_rebalancer.py`
3. `disconnection_repair.py`
3. `enhance_evidence.py`
3. `enhance_hypotheses.py`
3. `enhance_mechanisms.py`
3. `evidence_document.py`
3. `evidence_weighting.py`

## Update: Added llm_gateway.py
- `llm_gateway.py`: Core gateway implementation (Category A: Semantic)

Total files classified: 68

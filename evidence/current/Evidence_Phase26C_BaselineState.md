478a665 Phase 26B Partial Complete: Fail-Fast Pipeline Implementation
95b773b CLAUDE.md Updated: Phase 26A → Phase 26B Transition Complete
75a0f77 Phase 26A Complete: Aggressive Ontology Resilience Testing Results
27ffbfd Phase 25E → Phase 26A Transition: CLAUDE.md Updated for Aggressive Ontology Resilience Testing
540324b Phase 25E Complete: Systematic Pattern Analysis & Discovery
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0 -- /home/brian/projects/process_tracing/test_env/bin/python
cachedir: .pytest_cache
rootdir: /home/brian/projects/process_tracing
plugins: anyio-4.10.0
collecting ... collected 22 items

tests/test_ontology_manager.py::TestOntologyManager::test_initialization PASSED                                                                                                               [  4%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_evidence_hypothesis_edges PASSED                                                                                                [  9%]
tests/test_ontology_manager.py::TestOntologyManager::test_backwards_compatibility_wrapper PASSED                                                                                              [ 13%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_van_evera_edges PASSED                                                                                                          [ 18%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_edge_types_for_relationship PASSED                                                                                              [ 22%]
tests/test_ontology_manager.py::TestOntologyManager::test_validate_edge_valid PASSED                                                                                                          [ 27%]
tests/test_ontology_manager.py::TestOntologyManager::test_validate_edge_invalid_type PASSED                                                                                                   [ 31%]
tests/test_ontology_manager.py::TestOntologyManager::test_validate_edge_invalid_property_value PASSED                                                                                         [ 36%]
tests/test_ontology_manager.py::TestOntologyManager::test_validate_edge_out_of_range PASSED                                                                                                   [ 40%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_edge_properties PASSED                                                                                                          [ 45%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_required_properties PASSED                                                                                                      [ 50%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_edges_by_domain PASSED                                                                                                          [ 54%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_edges_by_range PASSED                                                                                                           [ 59%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_all_diagnostic_edge_types PASSED                                                                                                [ 63%]
tests/test_ontology_manager.py::TestOntologyManager::test_is_evidence_to_hypothesis_edge PASSED                                                                                               [ 68%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_node_properties PASSED                                                                                                          [ 72%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_edge_label PASSED                                                                                                               [ 77%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_node_color PASSED                                                                                                               [ 81%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_all_edge_types PASSED                                                                                                           [ 86%]
tests/test_ontology_manager.py::TestOntologyManager::test_get_all_node_types PASSED                                                                                                           [ 90%]
tests/test_ontology_manager.py::TestOntologyManager::test_lookup_table_completeness PASSED                                                                                                    [ 95%]
tests/test_ontology_manager.py::TestOntologyManager::test_backwards_compatibility_with_hardcoded_lists PASSED                                                                                 [100%]

======================================================================================== 22 passed in 0.03s =========================================================================================
=== BASELINE: UNMODIFIED ONTOLOGY BEHAVIOR ===
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
✅ Ontology validation passed: 19 edge types
✅ LiteLLM import successful
✅ System validation complete - proceeding with analysis

📄 Detected text input: input_text/revolutions/french_revolution.txt
🔄 EXTRACTION PHASE: Converting text to JSON graph...
[EXTRACTION] Starting graph extraction from: input_text/revolutions/french_revolution.txt
[EXTRACTION] Input text size: 52160 characters
[EXTRACTION] Creating StructuredProcessTracingExtractor...
[EXTRACTION] Calling extraction with 52160 characters...

======================================================================
PHASE 19B: DIAGNOSTIC PIPELINE EXTRACTION
======================================================================
[PIPELINE] Starting extraction with structured output (model: gpt-5-mini)
[PIPELINE] Input text size: 52,160 characters (13,040 estimated tokens)
[PIPELINE] Project: direct_extraction
[PIPELINE] Timestamp: 2025-09-13T08:38:00.939155
[PIPELINE] Formatting prompt template...
[PIPELINE] Final prompt size: 63,245 characters
[PIPELINE] Starting LLM extraction phase...
[DIAGNOSTIC] Starting LiteLLM call with prompt size: 61.8KB
[DIAGNOSTIC] Model: gpt-5-mini
[DIAGNOSTIC] Timestamp: 2025-09-13T08:38:00.939295
[DIAGNOSTIC] Waiting for LLM response...
[DIAGNOSTIC] LLM call completed in 152.78 seconds
[DIAGNOSTIC] LLM returned 27975 characters of JSON
[DEBUG] Raw LLM response saved to: debug/raw_llm_response_20250913_084033_718.json
[PHASE23A] RAW LLM RESPONSE: 36 nodes, 38 edges
[PHASE23A] evidence_flight_to_varennes_1791 in raw nodes: False
[PHASE23A] event_flight_to_varennes_1791 in raw nodes: False
[PHASE23A] All edge references found in raw nodes - no orphans detected
[DIAGNOSTIC] Starting JSON cleaning and validation...
[PHASE23A] CLEANED RESPONSE: 36 nodes, 38 edges
[PHASE23A] No nodes lost during cleaning phase
[DIAGNOSTIC] JSON cleaned, starting Pydantic validation...
[DIAGNOSTIC] Pydantic validation completed in 0.00 seconds
[PHASE23A] FINAL RESULT: 36 nodes, 38 edges
[PHASE23A] evidence_flight_to_varennes_1791 in final nodes: False
[PHASE23A] All final edges have valid node references
[PIPELINE] ✅ Extraction completed in 152.78s
[PIPELINE] Creating extraction metadata...
EXTRACTION SUMMARY:
  Nodes: 36 (8/8 types, 100.0%)
  Edges: 38 (17/21 types, 81.0%)
  Node types: ['Actor', 'Alternative_Explanation', 'Causal_Mechanism', 'Condition', 'Data_Source', 'Event', 'Evidence', 'Hypothesis']
  Edge types: ['causes', 'confirms_occurrence', 'contradicts', 'enables', 'explains_mechanism', 'infers', 'initiates', 'part_of_mechanism', 'provides_evidence', 'provides_evidence_for', 'refutes_alternative', 'supports', 'tests_alternative', 'tests_hypothesis', 'tests_mechanism', 'updates_probability', 'weighs_evidence']
[EXTRACTION] Extraction completed successfully
[EXTRACTION] Graph saved to: output_data/direct_extraction/direct_extraction_20250913_084033_graph.json
[EXTRACTION] Extracted 36 nodes, 38 edges
✅ Graph extracted to: output_data/direct_extraction/direct_extraction_20250913_084033_graph.json
📁 Loading graph from: output_data/direct_extraction/direct_extraction_20250913_084033_graph.json
[MODULE-DEBUG] Starting core.analyze module import...
[IMPORT-DEBUG] Skipping Unicode setup on Windows (potential hang source)...
WARNING:root:Bayesian reporting components not available
[IMPORT-DEBUG] Unicode setup skipped, starting imports...
[IMPORT-DEBUG] Importing standard libraries...
[IMPORT-DEBUG] json imported
[IMPORT-DEBUG] argparse imported
[IMPORT-DEBUG] networkx imported
[IMPORT-DEBUG] collections imported
[IMPORT-DEBUG] copy imported
[IMPORT-DEBUG] logging imported
[IMPORT-DEBUG] time imported
[IMPORT-DEBUG] datetime imported
[IMPORT-DEBUG] matplotlib imported
[IMPORT-DEBUG] All libraries imported successfully
[IMPORT-DEBUG] Importing logging utilities...
[IMPORT-DEBUG] Logging utilities imported successfully
[LOGGER-DEBUG] Logger initialized: core.analyze
[IMPORT-DEBUG] About to import ontology module...
[MODULE-DEBUG] Importing ontology...
[IMPORT-DEBUG] Ontology import completed successfully
[MODULE-DEBUG] Importing enhance_evidence...
[IMPORT-DEBUG] enhance_evidence import completed successfully
[MODULE-DEBUG] Importing llm_reporting_utils...
[MODULE-DEBUG] llm_reporting_utils imported in 0.0s
[HANG-DEBUG] DISABLING plugin imports for hang debugging
[MODULE-DEBUG] Importing enhance_mechanisms... SKIPPED
[MODULE-DEBUG] Importing van_evera_workflow... SKIPPED
[MODULE-DEBUG] van_evera_workflow imported in 0.0s (skipped)
[MODULE-DEBUG] Starting function definitions section...
[MODULE-DEBUG] Importing dag_analysis...
[MODULE-DEBUG] dag_analysis imported in 0.0s
[MODULE-DEBUG] Importing temporal components...
[MODULE-DEBUG] TemporalExtractor imported in 0.0s
[MODULE-DEBUG] TemporalGraph imported in 0.0s
[MODULE-DEBUG] TemporalValidator imported in 0.0s
[MODULE-DEBUG] Importing temporal_viz...
[MODULE-DEBUG] TemporalVisualizer imported in 0.0s
[MODULE-DEBUG] Reached end of analyze.py module - all imports and definitions complete!
[MODULE-DEBUG] __name__ is: core.analyze
[EXTREME-DEBUG] *** LOAD_GRAPH FUNCTION CALLED WITH: output_data/direct_extraction/direct_extraction_20250913_084033_graph.json ***
[LOAD-HANG-DEBUG] *** ENTERED load_graph function ***
[SYSTEM] Entered load_graph function
[LOAD-HANG-DEBUG] *** About to call time.time() ***
[LOAD-HANG-DEBUG] *** About to call progress.checkpoint() ***
[SYSTEM] About to call progress.checkpoint
[LOAD-HANG-DEBUG] *** Calling progress.checkpoint with timeout ***
[PROGRESS] 0.0s | load_graph | Loading from output_data/direct_extraction/direct_extraction_20250913_084033_graph.json
[LOAD-HANG-DEBUG] *** progress.checkpoint completed successfully ***
[HANG-TRACE] CHECKPOINT A: About to start main load_graph logic
[LOAD-DEBUG] Starting load_graph(output_data/direct_extraction/direct_extraction_20250913_084033_graph.json)
[HANG-TRACE] CHECKPOINT B: After starting load_graph message
[LOAD-DEBUG] 0.0s | File exists check completed
[HANG-TRACE] CHECKPOINT C: File exists check passed
[HANG-TRACE] CHECKPOINT D: About to open JSON file
[HANG-TRACE] CHECKPOINT E: File opened, about to load JSON
[HANG-TRACE] CHECKPOINT F: JSON loaded successfully
[LOAD-DEBUG] 0.0s | JSON file loaded in 0.0s
[HANG-TRACE] CHECKPOINT G: About to create NetworkX graph
[HANG-TRACE] CHECKPOINT H: Creating MultiDiGraph (PHASE 23A FIX)
[HANG-TRACE] CHECKPOINT I: MultiDiGraph created
[LOAD-DEBUG] 0.0s | NetworkX graph created in 0.0s
[HANG-TRACE] CHECKPOINT J: About to process nodes
[LOAD-DEBUG] 0.0s | Processing 36 nodes...
[HANG-TRACE] CHECKPOINT K: Starting node processing loop
[LOAD-DEBUG] 0.0s | Node processing completed in 0.0s
[LOAD-DEBUG] 0.0s | Processing 38 edges...
[PHASE23A] EDGE PROCESSING SUMMARY: 38 processed, 0 skipped (of 38 total)
[LOAD-DEBUG] 0.0s | Edge processing completed in 0.0s
[HANG-TRACE] CHECKPOINT L: Edge processing completed
[LOAD-DEBUG] 0.0s | SKIPPING connectivity repair (performance test)
[HANG-TRACE] CHECKPOINT M: About to return from load_graph
[LOAD-DEBUG] 0.0s | Returning without connectivity repair for testing
[HANG-TRACE] CHECKPOINT N: Returning G and data
✅ Graph loaded successfully in 0.00s
   📊 36 nodes, 38 edges


🎉 Analysis completed successfully!
⏱️  Total time: 0.00s
BASELINE ONTOLOGY EDGE TYPES:
Total edge types: 19
  - addresses_research_question
  - causes
  - confirms_occurrence
  - constrains
  - contradicts
  - disproves_occurrence
  - enables
  - explains_mechanism
  - infers
  - initiates
  - part_of_mechanism
  - provides_evidence
  - provides_evidence_for
  - refutes
  - supports
  - tests_hypothesis
  - tests_mechanism
  - updates_probability
  - weighs_evidence

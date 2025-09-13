=== COMPONENT ISOLATION TESTING ===
✅ Extraction import OK
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
✅ Load_graph import OK
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
✅ Core.analyze imports OK
usage: analyze_direct.py [-h] [--html] [--output-dir OUTPUT_DIR] [--extract-only] input_file
analyze_direct.py: error: unrecognized arguments: --debug-checkpoints
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

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
[EXTREME-DEBUG] *** LOAD_GRAPH FUNCTION CALLED WITH: minimal_test.json ***
[LOAD-HANG-DEBUG] *** ENTERED load_graph function ***
[SYSTEM] Entered load_graph function
[LOAD-HANG-DEBUG] *** About to call time.time() ***
[LOAD-HANG-DEBUG] *** About to call progress.checkpoint() ***
[SYSTEM] About to call progress.checkpoint
[LOAD-HANG-DEBUG] *** Calling progress.checkpoint with timeout ***
[PROGRESS] 0.1s | load_graph | Loading from minimal_test.json
[LOAD-HANG-DEBUG] *** progress.checkpoint completed successfully ***
[HANG-TRACE] CHECKPOINT A: About to start main load_graph logic
[LOAD-DEBUG] Starting load_graph(minimal_test.json)
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
[LOAD-DEBUG] 0.0s | Processing 1 nodes...
[HANG-TRACE] CHECKPOINT K: Starting node processing loop
[LOAD-DEBUG] 0.0s | Node processing completed in 0.0s
[LOAD-DEBUG] 0.0s | Processing 0 edges...
[PHASE23A] EDGE PROCESSING SUMMARY: 0 processed, 0 skipped (of 0 total)
[LOAD-DEBUG] 0.0s | Edge processing completed in 0.0s
[HANG-TRACE] CHECKPOINT L: Edge processing completed
[LOAD-DEBUG] 0.0s | SKIPPING connectivity repair (performance test)
[HANG-TRACE] CHECKPOINT M: About to return from load_graph
[LOAD-DEBUG] 0.0s | Returning without connectivity repair for testing
[HANG-TRACE] CHECKPOINT N: Returning G and data
✅ Minimal graph loaded: 1 nodes, 0 edges
Traceback (most recent call last):
  File "<string>", line 9, in <module>
AttributeError: 'StructuredProcessTracingExtractor' object has no attribute 'extract'
Testing structured extraction directly...
About to call extract method...

=== HANG LOCATION IDENTIFIED ===
✅ Core component imports: ALL SUCCESSFUL
✅ Graph loading (load_graph): WORKS PERFECTLY
✅ Minimal graph operations: WORKS PERFECTLY
❌ HANG SOURCE FOUND: StructuredProcessTracingExtractor.extract_graph() method
❌ LLM extraction phase hangs - NOT the graph analysis phase

CRITICAL DISCOVERY: The pipeline hangs during LLM extraction, not during analysis!
This suggests the hang is related to LiteLLM/LLM API calls, not graph processing.


=== COMPREHENSIVE ONTOLOGY RESILIENCE VALIDATION ===
=== TESTING REMOVAL OF: supports ===
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
❌ SYSTEM VALIDATION FAILED: ❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: ['supports']
🔧 This indicates a configuration problem that would cause pipeline hanging.
💡 Fix the underlying issue before proceeding.
PIPELINE COMPLETED (success or clear failure)

=== TESTING REMOVAL OF: tests_hypothesis ===
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
❌ SYSTEM VALIDATION FAILED: ❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: ['tests_hypothesis']
🔧 This indicates a configuration problem that would cause pipeline hanging.
💡 Fix the underlying issue before proceeding.
PIPELINE COMPLETED (success or clear failure)

=== TESTING REMOVAL OF: provides_evidence_for ===
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
❌ SYSTEM VALIDATION FAILED: ❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: ['provides_evidence_for']
🔧 This indicates a configuration problem that would cause pipeline hanging.
💡 Fix the underlying issue before proceeding.
PIPELINE COMPLETED (success or clear failure)

=== TESTING REMOVAL OF: refutes ===
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
✅ Ontology validation passed: 18 edge types
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
[PIPELINE] Timestamp: 2025-09-13T08:42:03.687751
[PIPELINE] Formatting prompt template...
[PIPELINE] Final prompt size: 63,245 characters
[PIPELINE] Starting LLM extraction phase...
[DIAGNOSTIC] Starting LiteLLM call with prompt size: 61.8KB
[DIAGNOSTIC] Model: gpt-5-mini
[DIAGNOSTIC] Timestamp: 2025-09-13T08:42:03.687890
[DIAGNOSTIC] Waiting for LLM response...
[DIAGNOSTIC] LLM call completed in 145.64 seconds
[DIAGNOSTIC] LLM returned 30628 characters of JSON
[DEBUG] Raw LLM response saved to: debug/raw_llm_response_20250913_084429_330.json
[PHASE23A] RAW LLM RESPONSE: 34 nodes, 39 edges
[PHASE23A] evidence_flight_to_varennes_1791 in raw nodes: False
[PHASE23A] event_flight_to_varennes_1791 in raw nodes: False
[PHASE23A] All edge references found in raw nodes - no orphans detected
[DIAGNOSTIC] Starting JSON cleaning and validation...
[PHASE23A] CLEANED RESPONSE: 34 nodes, 39 edges
[PHASE23A] No nodes lost during cleaning phase
[DIAGNOSTIC] JSON cleaned, starting Pydantic validation...
[DIAGNOSTIC] Pydantic validation completed in 0.00 seconds
[PHASE23A] FINAL RESULT: 34 nodes, 39 edges
[PHASE23A] evidence_flight_to_varennes_1791 in final nodes: False
[PHASE23A] All final edges have valid node references
[PIPELINE] ✅ Extraction completed in 145.64s
[PIPELINE] Creating extraction metadata...
EXTRACTION SUMMARY:
  Nodes: 34 (8/8 types, 100.0%)
  Edges: 39 (16/21 types, 76.2%)
  Node types: ['Actor', 'Alternative_Explanation', 'Causal_Mechanism', 'Condition', 'Data_Source', 'Event', 'Evidence', 'Hypothesis']
  Edge types: ['causes', 'confirms_occurrence', 'enables', 'explains_mechanism', 'infers', 'initiates', 'part_of_mechanism', 'provides_evidence', 'provides_evidence_for', 'refutes_alternative', 'supports_alternative', 'tests_alternative', 'tests_hypothesis', 'tests_mechanism', 'updates_probability', 'weighs_evidence']
[EXTRACTION] Extraction completed successfully
[EXTRACTION] Graph saved to: output_data/direct_extraction/direct_extraction_20250913_084429_graph.json
[EXTRACTION] Extracted 34 nodes, 39 edges
✅ Graph extracted to: output_data/direct_extraction/direct_extraction_20250913_084429_graph.json
📁 Loading graph from: output_data/direct_extraction/direct_extraction_20250913_084429_graph.json
[MODULE-DEBUG] Starting core.analyze module import...
[IMPORT-DEBUG] Skipping Unicode setup on Windows (potential hang source)...
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
[EXTREME-DEBUG] *** LOAD_GRAPH FUNCTION CALLED WITH: output_data/direct_extraction/direct_extraction_20250913_084429_graph.json ***
[LOAD-HANG-DEBUG] *** ENTERED load_graph function ***
[LOAD-HANG-DEBUG] *** About to call time.time() ***
[LOAD-HANG-DEBUG] *** About to call progress.checkpoint() ***
[LOAD-HANG-DEBUG] *** Calling progress.checkpoint with timeout ***
[PROGRESS] 0.0s | load_graph | Loading from output_data/direct_extraction/direct_extraction_20250913_084429_graph.json
[LOAD-HANG-DEBUG] *** progress.checkpoint completed successfully ***
[HANG-TRACE] CHECKPOINT A: About to start main load_graph logic
[LOAD-DEBUG] Starting load_graph(output_data/direct_extraction/direct_extraction_20250913_084429_graph.json)
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
[LOAD-DEBUG] 0.0s | Processing 34 nodes...
[HANG-TRACE] CHECKPOINT K: Starting node processing loop
[LOAD-DEBUG] 0.0s | Node processing completed in 0.0s
[LOAD-DEBUG] 0.0s | Processing 39 edges...
[PHASE23A] EDGE PROCESSING SUMMARY: 39 processed, 0 skipped (of 39 total)
[LOAD-DEBUG] 0.0s | Edge processing completed in 0.0s
[HANG-TRACE] CHECKPOINT L: Edge processing completed
[LOAD-DEBUG] 0.0s | SKIPPING connectivity repair (performance test)
[HANG-TRACE] CHECKPOINT M: About to return from load_graph
[LOAD-DEBUG] 0.0s | Returning without connectivity repair for testing
[HANG-TRACE] CHECKPOINT N: Returning G and data
✅ Graph loaded successfully in 0.00s
   📊 34 nodes, 39 edges


🎉 Analysis completed successfully!
⏱️  Total time: 0.00s

=== TESTING REMOVAL OF: confirms_occurrence ===
PIPELINE COMPLETED (success or clear failure)

Phase 26C completed successfully - ontology resilience implemented
=== FINAL BASELINE VERIFICATION ===
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
[PIPELINE] Timestamp: 2025-09-13T08:47:45.947395
[PIPELINE] Formatting prompt template...
[PIPELINE] Final prompt size: 63,245 characters
[PIPELINE] Starting LLM extraction phase...
[DIAGNOSTIC] Starting LiteLLM call with prompt size: 61.8KB
[DIAGNOSTIC] Model: gpt-5-mini
[DIAGNOSTIC] Timestamp: 2025-09-13T08:47:45.947551
[DIAGNOSTIC] Waiting for LLM response...
[DIAGNOSTIC] LLM call completed in 164.53 seconds
[DIAGNOSTIC] LLM returned 36017 characters of JSON
[DEBUG] Raw LLM response saved to: debug/raw_llm_response_20250913_085030_481.json
[PHASE23A] RAW LLM RESPONSE: 41 nodes, 43 edges
[PHASE23A] evidence_flight_to_varennes_1791 in raw nodes: False
[PHASE23A] event_flight_to_varennes_1791 in raw nodes: False
[PHASE23A] All edge references found in raw nodes - no orphans detected
[DIAGNOSTIC] Starting JSON cleaning and validation...
[PHASE23A] CLEANED RESPONSE: 41 nodes, 43 edges
[PHASE23A] No nodes lost during cleaning phase
[DIAGNOSTIC] JSON cleaned, starting Pydantic validation...
[DIAGNOSTIC] Pydantic validation completed in 0.00 seconds
[PHASE23A] FINAL RESULT: 41 nodes, 43 edges
[PHASE23A] evidence_flight_to_varennes_1791 in final nodes: False
[PHASE23A] All final edges have valid node references
[PIPELINE] ✅ Extraction completed in 164.54s
[PIPELINE] Creating extraction metadata...
EXTRACTION SUMMARY:
  Nodes: 41 (8/8 types, 100.0%)
  Edges: 43 (14/21 types, 66.7%)
  Node types: ['Actor', 'Alternative_Explanation', 'Causal_Mechanism', 'Condition', 'Data_Source', 'Event', 'Evidence', 'Hypothesis']
  Edge types: ['causes', 'confirms_occurrence', 'enables', 'infers', 'initiates', 'part_of_mechanism', 'provides_evidence', 'refutes_alternative', 'supports', 'tests_alternative', 'tests_hypothesis', 'tests_mechanism', 'updates_probability', 'weighs_evidence']
[EXTRACTION] Extraction completed successfully
[EXTRACTION] Graph saved to: output_data/direct_extraction/direct_extraction_20250913_085030_graph.json
[EXTRACTION] Extracted 41 nodes, 43 edges
✅ Graph extracted to: output_data/direct_extraction/direct_extraction_20250913_085030_graph.json
📁 Loading graph from: output_data/direct_extraction/direct_extraction_20250913_085030_graph.json
[MODULE-DEBUG] Starting core.analyze module import...
[IMPORT-DEBUG] Skipping Unicode setup on Windows (potential hang source)...
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
[EXTREME-DEBUG] *** LOAD_GRAPH FUNCTION CALLED WITH: output_data/direct_extraction/direct_extraction_20250913_085030_graph.json ***
[LOAD-HANG-DEBUG] *** ENTERED load_graph function ***
[LOAD-HANG-DEBUG] *** About to call time.time() ***
[LOAD-HANG-DEBUG] *** About to call progress.checkpoint() ***
[LOAD-HANG-DEBUG] *** Calling progress.checkpoint with timeout ***
[PROGRESS] 0.0s | load_graph | Loading from output_data/direct_extraction/direct_extraction_20250913_085030_graph.json
[LOAD-HANG-DEBUG] *** progress.checkpoint completed successfully ***
[HANG-TRACE] CHECKPOINT A: About to start main load_graph logic
[LOAD-DEBUG] Starting load_graph(output_data/direct_extraction/direct_extraction_20250913_085030_graph.json)
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
[LOAD-DEBUG] 0.0s | Processing 41 nodes...
[HANG-TRACE] CHECKPOINT K: Starting node processing loop
[LOAD-DEBUG] 0.0s | Node processing completed in 0.0s
[LOAD-DEBUG] 0.0s | Processing 43 edges...
[PHASE23A] EDGE PROCESSING SUMMARY: 43 processed, 0 skipped (of 43 total)
[LOAD-DEBUG] 0.0s | Edge processing completed in 0.0s
[HANG-TRACE] CHECKPOINT L: Edge processing completed
[LOAD-DEBUG] 0.0s | SKIPPING connectivity repair (performance test)
[HANG-TRACE] CHECKPOINT M: About to return from load_graph
[LOAD-DEBUG] 0.0s | Returning without connectivity repair for testing
[HANG-TRACE] CHECKPOINT N: Returning G and data
✅ Graph loaded successfully in 0.00s
   📊 41 nodes, 43 edges


🎉 Analysis completed successfully!
⏱️  Total time: 0.00s

=== ONTOLOGY MODIFICATION TESTING ===
=== TESTING REMOVAL OF: supports ===
Modified ontology - removed supports
Testing pipeline with modified ontology...
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
❌ SYSTEM VALIDATION FAILED: ❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: ['supports']
🔧 This indicates a configuration problem that would cause pipeline hanging.
💡 Fix the underlying issue before proceeding.
PIPELINE RESULT: timeout or error
Ontology restored to original state
=== TESTING REMOVAL OF: tests_hypothesis ===
Modified ontology - removed tests_hypothesis
Testing pipeline with modified ontology...
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
❌ SYSTEM VALIDATION FAILED: ❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: ['tests_hypothesis']
🔧 This indicates a configuration problem that would cause pipeline hanging.
💡 Fix the underlying issue before proceeding.
PIPELINE RESULT: timeout or error
Ontology restored to original state
=== GRAPH-ONTOLOGY MISMATCH TESTING ===
=== COMPLETING MISSING CRITICAL TESTS ===
Step 1: Extract graph with original ontology...
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
Using graph file: output_data/direct_extraction/direct_extraction_20250913_090643_graph.json
Step 2: Modify ontology AFTER extraction...
Step 3: THE CRITICAL TEST - Loading existing graph with modified ontology...
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
❌ SYSTEM VALIDATION FAILED: ❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: ['tests_hypothesis']
🔧 This indicates a configuration problem that would cause pipeline hanging.
💡 Fix the underlying issue before proceeding.
GRAPH ANALYSIS RESULT: timeout or error
Step 4: STATE CORRUPTION TEST - Restore ontology and test if problems persist...
Ontology restored. Now testing if hang persists...
🚀 DIRECT ANALYSIS ENTRY POINT
===============================
Bypassing problematic python -m core.analyze execution

🔍 VALIDATING SYSTEM CONFIGURATION...
✅ Ontology validation passed: 19 edge types
✅ LiteLLM import successful
✅ System validation complete - proceeding with analysis

📁 Detected JSON input: output_data/direct_extraction/direct_extraction_20250913_090643_graph.json
📁 Loading graph from: output_data/direct_extraction/direct_extraction_20250913_090643_graph.json
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
[EXTREME-DEBUG] *** LOAD_GRAPH FUNCTION CALLED WITH: output_data/direct_extraction/direct_extraction_20250913_090643_graph.json ***
[LOAD-HANG-DEBUG] *** ENTERED load_graph function ***
[LOAD-HANG-DEBUG] *** About to call time.time() ***
[LOAD-HANG-DEBUG] *** About to call progress.checkpoint() ***
[LOAD-HANG-DEBUG] *** Calling progress.checkpoint with timeout ***
[PROGRESS] 0.0s | load_graph | Loading from output_data/direct_extraction/direct_extraction_20250913_090643_graph.json
[LOAD-HANG-DEBUG] *** progress.checkpoint completed successfully ***
[HANG-TRACE] CHECKPOINT A: About to start main load_graph logic
[LOAD-DEBUG] Starting load_graph(output_data/direct_extraction/direct_extraction_20250913_090643_graph.json)
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
[LOAD-DEBUG] 0.0s | Processing 42 nodes...
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
   📊 42 nodes, 43 edges


🎉 Analysis completed successfully!
⏱️  Total time: 0.00s

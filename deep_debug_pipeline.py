#!/usr/bin/env python3
"""
Deep diagnostic investigation of the process tracing pipeline.
Tests each stage individually to identify the exact failure point.
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def debug_stage(stage_name, func):
    """Execute a stage with comprehensive error handling and timing."""
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] STAGE: {stage_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = func()
        duration = time.time() - start_time
        print(f"[{timestamp()}] SUCCESS: {stage_name} completed in {duration:.2f}s")
        return result, True
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"[{timestamp()}] FAILED: {stage_name} failed after {duration:.2f}s")
        print(f"[{timestamp()}] ERROR: {str(e)}")
        print(f"[{timestamp()}] TRACEBACK:")
        traceback.print_exc()
        return None, False

def main():
    print(f"[{timestamp()}] DEEP PIPELINE INVESTIGATION STARTING")
    print(f"System: Python {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Add project root to path
    sys.path.insert(0, os.getcwd())
    
    # Stage 1: Test basic imports
    def test_imports():
        print(f"[{timestamp()}] Testing imports...")
        from core.structured_extractor import StructuredProcessTracingExtractor
        from core.extract import parse_json, PROMPT_TEMPLATE, analyze_graph_connectivity
        from core.plugins.van_evera_llm_interface import get_van_evera_llm
        print(f"[{timestamp()}] All imports successful")
        return {"extractor": StructuredProcessTracingExtractor, "van_evera": get_van_evera_llm}
    
    imports, success = debug_stage("IMPORT_TEST", test_imports)
    if not success:
        return
    
    # Stage 2: Test input reading
    def test_input_reading():
        print(f"[{timestamp()}] Reading test input...")
        test_file = "input_text/test_simple/test.txt"
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        with open(test_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        print(f"[{timestamp()}] Read {len(text)} characters: '{text[:50]}...'")
        return text
    
    text, success = debug_stage("INPUT_READING", test_input_reading)
    if not success:
        return
    
    # Stage 3: Test structured extraction
    def test_extraction():
        print(f"[{timestamp()}] Testing structured extraction...")
        extractor = imports["extractor"]()
        print(f"[{timestamp()}] Using model: {extractor.model_name}")
        print(f"[{timestamp()}] API key configured: {'Yes' if extractor.api_key else 'No'}")
        
        result = extractor.extract_graph(text)
        
        print(f"[{timestamp()}] Extraction completed")
        print(f"[{timestamp()}] Nodes: {len(result.graph.nodes)}")
        print(f"[{timestamp()}] Edges: {len(result.graph.edges)}")
        
        # Convert to format expected by pipeline
        graph_data = {
            'nodes': [node.model_dump() for node in result.graph.nodes],
            'edges': [edge.model_dump() for edge in result.graph.edges]
        }
        
        return graph_data
    
    graph_data, success = debug_stage("EXTRACTION", test_extraction)
    if not success:
        return
    
    # Stage 4: Test connectivity analysis
    def test_connectivity():
        print(f"[{timestamp()}] Testing connectivity analysis...")
        from core.extract import analyze_graph_connectivity
        
        connectivity = analyze_graph_connectivity(graph_data)
        print(f"[{timestamp()}] Connectivity result: {connectivity}")
        return connectivity
    
    connectivity, success = debug_stage("CONNECTIVITY_ANALYSIS", test_connectivity)
    if not success:
        return
    
    # Stage 5: Test graph saving
    def test_graph_saving():
        print(f"[{timestamp()}] Testing graph saving...")
        output_dir = Path("output_data/debug_deep_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        graph_json_path = output_dir / "test_graph.json"
        
        with open(graph_json_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"[{timestamp()}] Graph saved to: {graph_json_path}")
        print(f"[{timestamp()}] File size: {os.path.getsize(graph_json_path)} bytes")
        
        return graph_json_path
    
    graph_json_path, success = debug_stage("GRAPH_SAVING", test_graph_saving)
    if not success:
        return
    
    # Stage 6: Test Van Evera LLM interface
    def test_van_evera():
        print(f"[{timestamp()}] Testing Van Evera LLM interface...")
        van_evera_llm = imports["van_evera"]()
        
        # Test a simple assessment
        from core.plugins.van_evera_llm_schemas import ProbativeValueAssessment
        
        test_assessment = van_evera_llm.assess_probative_value(
            evidence_description="Economic sanctions were imposed",
            hypothesis_description="Sanctions cause protests",
            context="Testing LLM interface"
        )
        
        print(f"[{timestamp()}] Van Evera test result: {test_assessment}")
        return test_assessment
    
    van_evera_result, success = debug_stage("VAN_EVERA_LLM", test_van_evera)
    if not success:
        return
    
    # Stage 7: Test core.analyze module loading
    def test_analyze_import():
        print(f"[{timestamp()}] Testing core.analyze import...")
        import core.analyze
        print(f"[{timestamp()}] core.analyze imported successfully")
        return core.analyze
    
    analyze_module, success = debug_stage("ANALYZE_IMPORT", test_analyze_import)
    if not success:
        return
    
    # Stage 8: Test subprocess call to core.analyze
    def test_analyze_subprocess():
        print(f"[{timestamp()}] Testing subprocess call to core.analyze...")
        import subprocess
        
        # Create a minimal test
        minimal_graph = {
            "nodes": [{"id": "test_node", "type": "Event", "properties": {"description": "Test event"}}],
            "edges": []
        }
        
        test_graph_path = Path("test_minimal_graph.json")
        with open(test_graph_path, "w") as f:
            json.dump(minimal_graph, f)
        
        print(f"[{timestamp()}] Created test graph: {test_graph_path}")
        
        # Try the subprocess call with short timeout
        cmd = [sys.executable, '-m', 'core.analyze', str(test_graph_path), '--html']
        print(f"[{timestamp()}] Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print(f"[{timestamp()}] Return code: {result.returncode}")
        print(f"[{timestamp()}] STDOUT length: {len(result.stdout)}")
        print(f"[{timestamp()}] STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print(f"[{timestamp()}] STDOUT sample: {result.stdout[:200]}...")
        if result.stderr:
            print(f"[{timestamp()}] STDERR sample: {result.stderr[:200]}...")
        
        # Clean up
        if test_graph_path.exists():
            os.unlink(test_graph_path)
        
        return result
    
    analyze_result, success = debug_stage("ANALYZE_SUBPROCESS", test_analyze_subprocess)
    if not success:
        return
    
    # Stage 9: Test the exact execute_single_case_processing function
    def test_full_processing():
        print(f"[{timestamp()}] Testing execute_single_case_processing...")
        from process_trace_advanced import execute_single_case_processing
        
        result = execute_single_case_processing(
            "input_text/test_simple/test.txt",
            "output_data/debug_full_processing",
            "test_simple"
        )
        
        print(f"[{timestamp()}] Function returned: {result}")
        
        # Check what files were created
        output_dir = Path("output_data/debug_full_processing")
        if output_dir.exists():
            files = list(output_dir.iterdir())
            print(f"[{timestamp()}] Files created: {[f.name for f in files]}")
        else:
            print(f"[{timestamp()}] Output directory was not created")
        
        return result
    
    full_result, success = debug_stage("FULL_PROCESSING", test_full_processing)
    
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] INVESTIGATION COMPLETE")
    print(f"{'='*60}")
    
    if success:
        print(f"[{timestamp()}] ALL STAGES PASSED - Pipeline is working!")
        
        # Check for HTML output
        html_files = list(Path("output_data/debug_full_processing").glob("*.html"))
        if html_files:
            print(f"[{timestamp()}] HTML generated: {html_files[0]}")
            print(f"[{timestamp()}] Full path: {html_files[0].absolute()}")
        else:
            print(f"[{timestamp()}] WARNING: No HTML files found in output directory")
    else:
        print(f"[{timestamp()}] Pipeline failed at the final stage")

if __name__ == "__main__":
    main()
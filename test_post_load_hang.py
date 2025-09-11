#!/usr/bin/env python3
"""
Post-load_graph Hang Investigation
==================================

Since load_graph() works perfectly in isolation, the hang must be happening
in the main analysis pipeline AFTER load_graph() returns successfully.
"""

import sys
import os
import time
sys.path.insert(0, '/home/brian/projects/process_tracing')

def test_post_load_operations():
    """Test what happens after load_graph returns"""
    print("="*60)
    print("TESTING POST-LOAD_GRAPH OPERATIONS")
    print("="*60)
    
    try:
        # Step 1: Call load_graph (we know this works)
        from core.analyze import load_graph
        json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
        
        print("Step 1: Calling load_graph...")
        G, data = load_graph(json_file)
        print(f"✅ load_graph completed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Step 2: Test logger operations (from main analysis)
        import logging
        logger = logging.getLogger(__name__)
        
        print("Step 2: Testing logger operations...")
        logger.info("Successfully loaded graph", extra={'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(), 'operation': 'graph_loading'})
        print("✅ Logger operations completed")
        
        # Step 3: Test complexity analysis (from main analysis)
        print("Step 3: Testing complexity analysis...")
        
        def analyze_complexity(graph, data):
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            
            # Count by type
            node_types = {}
            for node in nodes:
                node_type = node.get('type', 'Unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            edge_types = {}
            for edge in edges:
                edge_type = edge.get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            return {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'node_types': node_types,
                'edge_types': edge_types,
                'complexity_score': len(nodes) * len(edges)
            }
        
        complexity = analyze_complexity(G, data)
        print(f"✅ Complexity analysis completed: {complexity['total_nodes']} nodes, {complexity['total_edges']} edges")
        
        # Step 4: Test other operations from main analysis
        print("Step 4: Testing various analysis operations...")
        
        # Test NetworkX operations
        print(f"   NetworkX nodes: {G.number_of_nodes()}")
        print(f"   NetworkX edges: {G.number_of_edges()}")
        print(f"   Is directed: {G.is_directed()}")
        
        # Test data access patterns
        sample_nodes = data.get('nodes', [])[:5]  # First 5 nodes
        print(f"   Sample nodes: {len(sample_nodes)}")
        
        sample_edges = data.get('edges', [])[:5]  # First 5 edges  
        print(f"   Sample edges: {len(sample_edges)}")
        
        print("✅ All post-load operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in post-load operations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exact_main_sequence():
    """Test the exact sequence from main() function after load_graph"""
    print("="*60)
    print("TESTING EXACT MAIN SEQUENCE")
    print("="*60)
    
    try:
        # Replicate exact main() setup
        import time
        import logging
        import argparse
        from core.analyze import load_graph
        
        # Mock args like in main()
        class Args:
            json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
            html = True
        
        args = Args()
        logger = logging.getLogger(__name__)
        
        print("Executing exact main() sequence...")
        
        # 1. load_graph call (we know this works)
        load_start_main = time.time()
        print("[MAIN-SEQUENCE] About to call load_graph...")
        sys.stdout.flush()
        
        G, data = load_graph(args.json_file)
        
        print(f"[MAIN-SEQUENCE] load_graph completed in {time.time() - load_start_main:.1f}s")
        print("[MAIN-SEQUENCE] About to continue with post-load operations...")
        sys.stdout.flush()
        
        # 2. This is the EXACT next line from main()
        print("[MAIN-SEQUENCE] Starting logger.info for successful load...")
        logger.info("Successfully loaded graph", extra={'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(), 'operation': 'graph_loading'})
        print("[MAIN-SEQUENCE] Logger.info completed")
        
        # 3. Complexity analysis (exact from main())
        print("[MAIN-SEQUENCE] Starting complexity analysis...")
        def analyze_complexity(graph, data):
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            return {'nodes': len(nodes), 'edges': len(edges)}
        
        complexity = analyze_complexity(G, data)
        print(f"[MAIN-SEQUENCE] Complexity analysis completed: {complexity}")
        
        print("✅ Exact main sequence completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in exact main sequence: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_exact_imports():
    """Test with the exact same imports as main analysis"""
    print("="*60)
    print("TESTING WITH EXACT MAIN ANALYSIS IMPORTS")
    print("="*60)
    
    try:
        # These are the exact imports from the main analyze.py file
        import json
        import argparse
        import time
        import sys
        import os
        import logging
        import signal
        from pathlib import Path
        from typing import Dict, List, Optional, Tuple, Set, Any
        from collections import defaultdict, Counter
        import networkx as nx
        
        # Import from core modules (exact as main)
        from core.analyze import load_graph
        from core.llm_reporting_utils import setup_logging
        
        print("All imports completed, testing load_graph call...")
        
        json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
        G, data = load_graph(json_file)
        
        print(f"✅ SUCCESS with exact imports: {G.number_of_nodes()} nodes")
        return True
        
    except Exception as e:
        print(f"❌ ERROR with exact imports: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("POST-LOAD_GRAPH HANG INVESTIGATION")
    print("="*60)
    
    os.chdir('/home/brian/projects/process_tracing')
    
    tests = [
        ("Post-Load Operations", test_post_load_operations),
        ("Exact Main Sequence", test_exact_main_sequence), 
        ("Exact Imports", test_with_exact_imports),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print()
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ TEST CRASHED: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("="*60)
    print("POST-LOAD INVESTIGATION SUMMARY")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎯 CONCLUSION: The hang is NOT in load_graph or immediate post-load operations")
        print("🔍 NEXT: The hang must be in the specific execution context of python -m core.analyze")
    else:
        failing_tests = [name for name, success in results.items() if not success]
        print(f"🔍 Focus on failing operations: {', '.join(failing_tests)}")

if __name__ == "__main__":
    main()
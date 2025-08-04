#!/usr/bin/env python3
"""
Test for Critical Issue #18: Path Finding Hangs
Verify that path finding completes within reasonable time limits.
"""

import networkx as nx
import time
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_complex_test_graph():
    """Create a complex graph that could cause path finding to hang"""
    G = nx.DiGraph()
    
    # Create a dense graph with many interconnections
    # This type of graph can cause exponential path explosion
    nodes = []
    
    # Add multiple layers of nodes
    for layer in range(5):
        for i in range(8):
            node_id = f"L{layer}_N{i}"
            node_type = ['Event', 'Hypothesis', 'Evidence'][i % 3]
            G.add_node(node_id, type=node_type, description=f"Node {node_id}")
            nodes.append(node_id)
    
    # Connect nodes densely within and between layers
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j and abs(i - j) <= 10:  # Connect nearby nodes
                G.add_edge(node1, node2, type='causes', probative_value=0.5)
    
    print(f"Created complex graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def test_path_finding_completes_quickly():
    """Test that path finding completes within reasonable time"""
    print("Testing Issue #18: Path Finding Hangs...")
    
    # Create complex graph
    G = create_complex_test_graph()
    
    # Test the fixed functions
    from core.dag_analysis import identify_causal_pathways
    
    print("Testing DAG analysis path finding...")
    start_time = time.time()
    
    try:
        # This should complete quickly now with the fixes
        results = identify_causal_pathways(G)
        duration = time.time() - start_time
        
        print(f"DAG analysis completed in {duration:.2f} seconds")
        print(f"Found {len(results)} pathways")
        
        if duration < 30.0:  # Should complete within 30 seconds
            print("SUCCESS: DAG path finding completes within time limit")
            return True
        else:
            print(f"FAILED: DAG path finding took too long ({duration:.2f}s)")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"ERROR during DAG analysis after {duration:.2f}s: {e}")
        return False

def test_cross_domain_analysis_performance():
    """Test cross-domain analysis performance"""
    print("Testing cross-domain analysis path finding...")
    
    G = create_complex_test_graph()
    
    from core.cross_domain_analysis import analyze_cross_domain_patterns
    
    start_time = time.time()
    
    try:
        results = analyze_cross_domain_patterns(G)
        duration = time.time() - start_time
        
        print(f"Cross-domain analysis completed in {duration:.2f} seconds")
        
        if duration < 30.0:  # Should complete within 30 seconds
            print("SUCCESS: Cross-domain path finding completes within time limit")
            return True
        else:
            print(f"FAILED: Cross-domain path finding took too long ({duration:.2f}s)")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"ERROR during cross-domain analysis after {duration:.2f}s: {e}")
        return False

if __name__ == "__main__":
    print("Running path finding performance tests...")
    
    test1_success = test_path_finding_completes_quickly()
    test2_success = test_cross_domain_analysis_performance()
    
    if test1_success and test2_success:
        print("\nSUCCESS: All path finding tests passed - Issue #18 fixed!")
        sys.exit(0)
    else:
        print("\nFAILED: Some path finding tests failed")
        sys.exit(1)
#!/usr/bin/env python3
"""
Test for Critical Issue #34: Graph State Corruption
Verify that original graph remains unchanged during analysis.
"""

import networkx as nx
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_graph_immutable_during_analysis():
    """Test that original graph remains unchanged during analysis"""
    print("Testing Issue #34: Graph State Corruption...")
    
    # Create a test graph
    G = nx.DiGraph()
    G.add_node('H1', type='Hypothesis', description='Test hypothesis')
    G.add_node('E1', type='Evidence', description='Test evidence')
    G.add_edge('E1', 'H1', type='supports', probative_value=0.7)
    
    # Record original state
    original_nodes = len(G.nodes())
    original_edges = len(G.edges())
    original_node_list = list(G.nodes())
    original_edge_list = list(G.edges())
    
    print(f"Original graph: {original_nodes} nodes, {original_edges} edges")
    print(f"Original nodes: {original_node_list}")
    print(f"Original edges: {original_edge_list}")
    
    # Import and run analysis function
    from core.analyze import analyze_graph
    
    try:
        # Run analysis
        results = analyze_graph(G)
        
        # Check if original graph is unchanged
        after_nodes = len(G.nodes())
        after_edges = len(G.edges())
        after_node_list = list(G.nodes())
        after_edge_list = list(G.edges())
        
        print(f"After analysis: {after_nodes} nodes, {after_edges} edges")
        print(f"After nodes: {after_node_list}")
        print(f"After edges: {after_edge_list}")
        
        # Test results
        nodes_unchanged = (original_nodes == after_nodes)
        edges_unchanged = (original_edges == after_edges)
        node_list_unchanged = (original_node_list == after_node_list)
        edge_list_unchanged = (original_edge_list == after_edge_list)
        
        print(f"Nodes count unchanged: {nodes_unchanged}")
        print(f"Edges count unchanged: {edges_unchanged}")
        print(f"Node list unchanged: {node_list_unchanged}")
        print(f"Edge list unchanged: {edge_list_unchanged}")
        
        if nodes_unchanged and edges_unchanged and node_list_unchanged and edge_list_unchanged:
            print("SUCCESS: Original graph remains unchanged during analysis")
            return True
        else:
            print("FAILED: Original graph was modified during analysis")
            return False
            
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_immutable_during_analysis()
    if not success:
        sys.exit(1)
    else:
        print("\nIssue #34 appears to be already fixed!")
#!/usr/bin/env python3
"""
Demonstration script for enhanced LLM-powered disconnection repair.

This script demonstrates the second-pass LLM repair approach that uses
full original text context to connect disconnected entities.
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.connectivity_analysis import analyze_connectivity, print_connectivity_report
from core.disconnection_repair import repair_graph_connectivity, save_repaired_graph


def test_llm_repair_with_context():
    """Test LLM repair functionality with original text context."""
    
    # Use the weighs_evidence_test data which has known disconnections
    graph_file = "output_data/weighs_evidence_test/weighs_evidence_test_20250805_054414_graph.json"
    text_file = "input_text/test_mechanism/test_full_ontology.txt" 
    
    if not os.path.exists(graph_file):
        print(f"Error: Graph file not found: {graph_file}")
        return False
    
    if not os.path.exists(text_file):
        print(f"Error: Text file not found: {text_file}")
        return False
    
    print("=== ENHANCED LLM REPAIR DEMONSTRATION ===")
    print(f"Graph: {graph_file}")
    print(f"Text context: {text_file}\n")
    
    # Load the graph data
    with open(graph_file, 'r', encoding='utf-8') as f:
        original_graph = json.load(f)
    
    # Load original text for context
    with open(text_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    print("ORIGINAL GRAPH CONNECTIVITY:")
    print("-" * 50)
    
    # Analyze original connectivity
    original_analysis = analyze_connectivity(original_graph)
    print_connectivity_report(original_analysis)
    
    print("\n" + "=" * 60)
    print("TESTING LLM REPAIR WITH FULL TEXT CONTEXT...")
    print("=" * 60)
    
    # Test repair with original text context
    repaired_graph = repair_graph_connectivity(original_graph, original_text)
    
    print("\nREPAIRED GRAPH CONNECTIVITY:")
    print("-" * 50)
    
    # Analyze repaired connectivity
    repaired_analysis = analyze_connectivity(repaired_graph)
    print_connectivity_report(repaired_analysis)
    
    # Show repair results
    original_edge_count = len(original_graph['edges'])
    repaired_edge_count = len(repaired_graph['edges'])
    added_edges = repaired_edge_count - original_edge_count
    
    print(f"\nREPAIR RESULTS:")
    print(f"Original edges: {original_edge_count}")
    print(f"Repaired edges: {repaired_edge_count}")
    print(f"Added edges: {added_edges}")
    
    improvement = original_analysis['connected_components'] - repaired_analysis['connected_components']
    print(f"Components: {original_analysis['connected_components']} -> {repaired_analysis['connected_components']} (improvement: {improvement})")
    
    # Show the new edges that were added by LLM
    if added_edges > 0:
        print(f"\nLLM-SUGGESTED EDGES:")
        new_edges = repaired_graph['edges'][original_edge_count:]
        for edge in new_edges:
            if edge['properties'].get('llm_generated'):
                print(f"  {edge['source']} --{edge['type']}--> {edge['target']}")
                reasoning = edge['properties'].get('reasoning', 'No reasoning provided')
                print(f"    Reasoning: {reasoning}")
                confidence = edge['properties'].get('confidence', 'Unknown')
                print(f"    Confidence: {confidence}")
    
    # Success assessment
    print(f"\n{'='*60}")
    if repaired_analysis['connected_components'] == 1:
        print("SUCCESS: LLM achieved full connectivity!")
        success = True
    elif improvement > 0:
        print(f"PARTIAL SUCCESS: LLM reduced components by {improvement}")
        success = True
    else:
        print("LIMITED SUCCESS: LLM could not improve connectivity")
        success = False
    
    # Save results if successful
    if added_edges > 0:
        repaired_path = save_repaired_graph(repaired_graph, graph_file)
        print(f"Enhanced graph saved to: {repaired_path}")
    
    return success


def demonstrate_llm_features():
    """Demonstrate key features of the enhanced LLM repair system."""
    
    print("\n" + "=" * 60)
    print("ENHANCED LLM REPAIR SYSTEM FEATURES")
    print("=" * 60)
    
    features = [
        {
            'feature': 'Full Original Text Context',
            'description': 'LLM receives the complete original text to understand semantic relationships',
            'benefit': 'Better connection suggestions based on actual content rather than just node descriptions'
        },
        {
            'feature': 'Complete Graph Context',
            'description': 'LLM sees full descriptions of all potential target nodes',
            'benefit': 'More informed decisions about which nodes to connect'
        },
        {
            'feature': 'Ontology Validation',
            'description': 'All LLM suggestions are validated against edge type constraints',
            'benefit': 'Ensures generated connections are methodologically valid'
        },
        {
            'feature': 'Patch-Based Approach',
            'description': 'Only adds new edges, never removes existing connections',
            'benefit': 'Preserves all original extracted relationships'
        },
        {
            'feature': 'Progressive Repair',
            'description': 'Applies inference rules first, then LLM for remaining issues',
            'benefit': 'Efficient two-stage approach combining automated rules with AI reasoning'
        },
        {
            'feature': 'Process Tracing Methodology',
            'description': 'Incorporates Van Evera, Beach & Pedersen methodological requirements',
            'benefit': 'Connections follow established process tracing best practices'
        }
    ]
    
    for feature in features:
        print(f"\n{feature['feature']}:")
        print(f"  Description: {feature['description']}")
        print(f"  Benefit: {feature['benefit']}")
    
    print(f"\nThe enhanced system represents a significant improvement over")
    print(f"basic inference rules by leveraging LLM reasoning with full context.")


if __name__ == "__main__":
    try:
        print("Testing enhanced LLM disconnection repair system...\n")
        
        success = test_llm_repair_with_context()
        demonstrate_llm_features()
        
        if success:
            print(f"\n✅ LLM repair demonstration completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  LLM repair demonstration completed with limited results.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during LLM repair demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_trace_advanced import execute_single_case_processing
import json

def test_focused_extraction():
    """Test the new focused extraction prompt"""
    
    # Test with a simple historical text
    test_text = """The French and Indian War ended in 1763 with Britain's victory over France. 
    To pay for the war debt, Britain imposed new taxes on the American colonies including the Stamp Act.
    
    Historians argue that British taxation policies caused colonial rebellion because they violated colonial understanding of their rights as Englishmen.
    The colonists protested that taxation without representation violated their rights as Englishmen.
    
    This led to increasing tensions and eventually the Boston Tea Party in 1773."""
    
    # Save test text to file
    test_file = "test_focused_input.txt"
    with open(test_file, "w") as f:
        f.write(test_text)
    
    try:
        # Extract graph and analyze
        analysis_file = execute_single_case_processing(
            case_file_path_str=test_file,
            output_dir_for_case_str="test_focused_output",
            project_name_str="focused_test"
        )
        
        print(f"Analysis file: {analysis_file}")
        
        # Load the analysis results
        if analysis_file and os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            print("\n=== HYPOTHESES FOUND ===")
            hypotheses = analysis.get('hypotheses_evaluation', [])
            print(f"Number of hypotheses: {len(hypotheses)}")
            
            for i, hyp in enumerate(hypotheses):
                print(f"\nHypothesis {i+1}:")
                print(f"  ID: {hyp.get('id', 'N/A')}")
                print(f"  Description: {hyp.get('description', 'N/A')}")
                print(f"  Evidence count: {len(hyp.get('evidence_assessments', []))}")
                
                for j, ev in enumerate(hyp.get('evidence_assessments', [])):
                    print(f"    Evidence {j+1}:")
                    print(f"      Probative value: {ev.get('probative_value', 0.0)}")
                    print(f"      Source quote: {ev.get('source_text_quote', 'N/A')}")
                    print(f"      Type: {ev.get('refined_evidence_type', 'N/A')}")
        
        # Also check the raw graph file
        import glob
        graph_files = glob.glob("test_focused_output/*_graph.json")
        if graph_files:
            print(f"\n=== RAW GRAPH DATA ===")
            with open(graph_files[0], 'r') as f:
                graph = json.load(f)
            
            print(f"Total nodes: {len(graph['nodes'])}")
            print(f"Total edges: {len(graph['edges'])}")
            
            # Count node types
            node_types = {}
            for node in graph['nodes']:
                node_type = node['type']
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
                if node_type in ['Hypothesis', 'Evidence']:
                    print(f"\n{node_type} node {node['id']}:")
                    print(f"  Description: {node['properties'].get('description', 'N/A')}")
            
            print(f"\nNode type counts: {node_types}")
            
            # Check Evidence-Hypothesis connections
            from core.ontology_manager import ontology_manager
            evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
            evidence_edges = [e for e in graph['edges'] if e['type'] in evidence_hypothesis_edges]
            print(f"Evidence-Hypothesis edges: {len(evidence_edges)}")
            for edge in evidence_edges:
                print(f"  {edge['source']} -[{edge['type']}]-> {edge['target']}")
                quote = edge['properties'].get('source_text_quote', '')
                probative = edge['properties'].get('probative_value', 0.0)
                print(f"    Quote: {quote}")
                print(f"    Probative value: {probative}")
                
    finally:
        # Keep files for inspection
        print(f"\nTest files saved in test_focused_output/ for inspection")

if __name__ == "__main__":
    test_focused_extraction()
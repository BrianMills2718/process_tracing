#!/usr/bin/env python3
"""Validate Phase 4B integration is complete and working"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Need to check if we have test data first
test_files = [
    "test_data/american_revolution_graph.json",
    "data/american_revolution_graph.json",
    "tests/test_data/american_revolution_graph.json"
]

def find_test_file():
    """Find a test graph file"""
    for file_path in test_files:
        if Path(file_path).exists():
            return file_path
    # If no test file, create a simple one
    return create_test_graph()

def create_test_graph():
    """Create a simple test graph for validation"""
    test_graph = {
        "nodes": [
            {"id": "h1", "type": "Hypothesis", "description": "Economic grievances caused the revolution"},
            {"id": "h2", "type": "Hypothesis", "description": "Ideological beliefs drove the revolution"},
            {"id": "h3", "type": "Hypothesis", "description": "British oppression triggered resistance"},
            {"id": "e1", "type": "Evidence", "description": "Colonists protested taxation without representation"},
            {"id": "e2", "type": "Evidence", "description": "Merchants organized boycotts of British goods"},
            {"id": "e3", "type": "Evidence", "description": "Philosophers wrote about natural rights and liberty"}
        ],
        "edges": [
            {"id": "edge1", "source": "e1", "target": "h1", "type": "supports", "properties": {"probative_value": 0.7}},
            {"id": "edge2", "source": "e1", "target": "h2", "type": "supports", "properties": {"probative_value": 0.6}},
            {"id": "edge3", "source": "e2", "target": "h1", "type": "supports", "properties": {"probative_value": 0.8}},
            {"id": "edge4", "source": "e2", "target": "h3", "type": "challenges", "properties": {"probative_value": 0.3}},
            {"id": "edge5", "source": "e3", "target": "h2", "type": "supports", "properties": {"probative_value": 0.9}},
            {"id": "edge6", "source": "e3", "target": "h1", "type": "neutral", "properties": {"probative_value": 0.4}}
        ]
    }
    
    # Save test graph
    test_path = "test_graph_temp.json"
    with open(test_path, 'w') as f:
        json.dump(test_graph, f, indent=2)
    return test_path

def validate_integration():
    try:
        from core.semantic_analysis_service import get_semantic_service
    except ImportError as e:
        print(f"[FAIL] Could not import semantic service: {e}")
        return False
    
    # Get or create test data
    test_graph_path = find_test_file()
    print(f"Using test graph: {test_graph_path}")
    
    # Clear cache and reset counters
    semantic_service = get_semantic_service()
    semantic_service.clear_cache()
    initial_calls = semantic_service._stats.get('llm_calls', 0)
    
    # Instead of running full analysis, let's test the batch evaluation directly
    try:
        # Load the graph
        with open(test_graph_path, 'r') as f:
            graph_data = json.load(f)
        
        # Count nodes
        evidence_nodes = [n for n in graph_data['nodes'] if n['type'] == 'Evidence']
        hypothesis_nodes = [n for n in graph_data['nodes'] if n['type'] == 'Hypothesis']
        
        evidence_count = len(evidence_nodes)
        hypothesis_count = len(hypothesis_nodes)
        
        print(f"Evidence nodes: {evidence_count}")
        print(f"Hypothesis nodes: {hypothesis_count}")
        
        # Test batch evaluation directly
        if evidence_count > 0 and hypothesis_count > 0:
            # Take first evidence and all hypotheses
            test_evidence = evidence_nodes[0]
            test_hypotheses = [{'id': h['id'], 'text': h.get('description', '')} 
                             for h in hypothesis_nodes]
            
            # Call batch evaluation
            result = semantic_service.evaluate_evidence_against_hypotheses_batch(
                test_evidence['id'],
                test_evidence.get('description', ''),
                test_hypotheses,
                context="Validation test"
            )
            
            # Check results
            total_calls = semantic_service._stats.get('llm_calls', 0) - initial_calls
            
            print(f"LLM calls made: {total_calls}")
            print(f"Expected with batching: 1")
            print(f"Expected without batching: {hypothesis_count}")
            
            if total_calls == 1:
                print("[OK] Batching is working perfectly!")
                
                # Show batch results
                print("\nBatch evaluation results:")
                for eval_result in result.evaluations:
                    print(f"  - {eval_result.hypothesis_id}: {eval_result.relationship_type} "
                          f"(confidence: {eval_result.confidence:.2f})")
                
                # Clean up temp file if created
                if test_graph_path == "test_graph_temp.json":
                    os.remove(test_graph_path)
                    
                return True
            else:
                print(f"[FAIL] Expected 1 call but got {total_calls}")
                return False
        else:
            print("[FAIL] Test graph has no evidence or hypotheses")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_integration()
    sys.exit(0 if success else 1)
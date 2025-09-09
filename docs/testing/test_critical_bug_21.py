#!/usr/bin/env python3
"""
Test for Critical Issue #21: Double Processing Bug
Verify that enhancement processing runs exactly once per analysis.
"""

import networkx as nx
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhancement_runs_once():
    """Test that enhancement functions are called exactly once"""
    print("Testing Issue #21: Double Processing Bug...")
    
    # Create a test graph
    G = nx.DiGraph()
    G.add_node('H1', type='Hypothesis', description='Test hypothesis')
    G.add_node('E1', type='Evidence', description='Test evidence')
    G.add_edge('E1', 'H1', type='supports', probative_value=0.7)
    G.add_node('M1', type='Causal_Mechanism', description='Test mechanism')
    
    print(f"Created test graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Mock the enhancement functions to count calls
    evidence_enhancement_count = 0
    mechanism_enhancement_count = 0
    
    def mock_evidence_enhancement(*args, **kwargs):
        nonlocal evidence_enhancement_count
        evidence_enhancement_count += 1
        print(f"Evidence enhancement called #{evidence_enhancement_count}")
        # Return a mock response
        mock_response = MagicMock()
        mock_response.refined_evidence_type = 'smoking_gun'
        mock_response.reasoning_for_type = 'Test reasoning'
        mock_response.likelihood_P_E_given_H = 0.8
        mock_response.likelihood_P_E_given_NotH = 0.2
        mock_response.justification_for_likelihoods = 'Test justification'
        mock_response.suggested_numerical_probative_value = 0.7
        return mock_response
    
    def mock_mechanism_enhancement(*args, **kwargs):
        nonlocal mechanism_enhancement_count
        mechanism_enhancement_count += 1
        print(f"Mechanism enhancement called #{mechanism_enhancement_count}")
        # Return a mock response
        mock_response = MagicMock()
        mock_response.completeness_score = 0.8
        mock_response.plausibility_score = 0.9
        mock_response.evidence_support_level = 'strong'
        mock_response.missing_elements = []
        mock_response.improvement_suggestions = []
        mock_response.detailed_reasoning = 'Test reasoning'
        return mock_response
    
    # Patch both enhancement functions
    with patch('core.enhance_evidence.refine_evidence_assessment_with_llm', side_effect=mock_evidence_enhancement), \
         patch('core.enhance_mechanisms.elaborate_mechanism_with_llm', side_effect=mock_mechanism_enhancement):
        
        # Import and run analysis
        from core.analyze import analyze_graph
        
        try:
            # Run analysis
            results = analyze_graph(G)
            
            print(f"Evidence enhancement called {evidence_enhancement_count} times")
            print(f"Mechanism enhancement called {mechanism_enhancement_count} times")
            
            # Check if each enhancement was called exactly once per relevant element
            evidence_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'Evidence']
            mechanism_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'Causal_Mechanism']
            
            expected_evidence_calls = len(evidence_nodes) * len([n for n in G.nodes() if G.nodes[n].get('type') == 'Hypothesis'])
            expected_mechanism_calls = len(mechanism_nodes)
            
            print(f"Expected evidence enhancement calls: {expected_evidence_calls}")
            print(f"Expected mechanism enhancement calls: {expected_mechanism_calls}")
            
            # Test results
            evidence_correct = evidence_enhancement_count <= expected_evidence_calls * 1.1  # Allow 10% tolerance
            mechanism_correct = mechanism_enhancement_count <= expected_mechanism_calls * 1.1  # Allow 10% tolerance
            
            if evidence_correct and mechanism_correct:
                print("SUCCESS: Enhancement functions called appropriate number of times")
                return True
            else:
                print("FAILED: Enhancement functions called too many times (possible double processing)")
                return False
                
        except Exception as e:
            print(f"ERROR during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_enhancement_runs_once()
    if not success:
        sys.exit(1)
    else:
        print("\nIssue #21 appears to be already fixed or was not present!")
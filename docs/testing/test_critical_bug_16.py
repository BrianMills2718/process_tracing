#!/usr/bin/env python3
"""
Test for Critical Issue #16: Evidence Balance Math Error
Verify that evidence balance calculation works correctly - supporting evidence should be positive, refuting negative.
"""

import networkx as nx
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_evidence_balance_calculation():
    """Test that positive evidence increases balance, negative decreases it"""
    print("Testing Issue #16: Evidence Balance Math Error...")
    
    # Create a test graph with supporting and refuting evidence
    G = nx.DiGraph()
    
    # Add hypothesis
    G.add_node('H1', type='Hypothesis', description='Test hypothesis')
    
    # Add supporting evidence
    G.add_node('E1', type='Evidence', description='Supporting evidence', credibility=0.8)
    G.add_edge('E1', 'H1', type='supports', probative_value=0.7)
    
    # Add refuting evidence  
    G.add_node('E2', type='Evidence', description='Refuting evidence', credibility=0.9)
    G.add_edge('E2', 'H1', type='refutes', probative_value=0.6)
    
    print(f"Created test graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Import analyze function and run analysis
    from core.analyze import analyze_evidence
    
    try:
        # Analyze the graph 
        results = analyze_evidence(G)
        print(f"Results type: {type(results)}")
        print(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        # Look for hypothesis results
        hyp_results = results.get('by_hypothesis', {})
        
        if 'H1' in hyp_results:
            hyp_result = hyp_results['H1']
            balance = hyp_result.get('balance', 0.0)
            supporting = hyp_result.get('supporting_evidence', [])
            refuting = hyp_result.get('refuting_evidence', [])
            
            print(f"Hypothesis balance: {balance}")
            print(f"Supporting evidence count: {len(supporting)}")
            print(f"Refuting evidence count: {len(refuting)}")
            
            # Expected balance = 0.7 (supporting) - 0.6 (refuting) = 0.1
            expected_balance = 0.7 - 0.6
            
            if abs(balance - expected_balance) < 0.01:  # Allow small floating point differences
                print(f"SUCCESS: Balance calculation correct ({balance:.3f} ~= {expected_balance:.3f})")
                
                # Additional checks
                if len(supporting) == 1 and len(refuting) == 1:
                    print("SUCCESS: Evidence categorized correctly")
                    return True
                else:
                    print("FAILED: Evidence not categorized correctly")
                    return False
            else:
                print(f"FAILED: Balance calculation incorrect (got {balance:.3f}, expected {expected_balance:.3f})")
                return False
        else:
            print("FAILED: No hypothesis results found")
            return False
            
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_evidence_balance_calculation()
    if not success:
        sys.exit(1)
    else:
        print("\nIssue #16 appears to be already fixed!")
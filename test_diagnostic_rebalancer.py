#!/usr/bin/env python3
"""
Test script for DiagnosticRebalancerPlugin
Demonstrates Van Evera diagnostic rebalancing functionality
"""

import json
import sys
import os

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.plugins.diagnostic_integration import (
    validate_diagnostic_distribution,
    rebalance_graph_diagnostics,
    create_van_evera_rebalancing_report
)


def load_test_graph():
    """Load the revolutions graph for testing"""
    graph_path = "output_data/revolutions/revolutions_20250805_122000_graph.json"
    
    if not os.path.exists(graph_path):
        print(f"ERROR: Graph file not found at {graph_path}")
        return None
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_current_distribution(graph_data):
    """Analyze and display current diagnostic distribution"""
    print("ANALYZING CURRENT DIAGNOSTIC DISTRIBUTION")
    print("="*50)
    
    analysis = validate_diagnostic_distribution(graph_data)
    
    print(f"Total evidence relationships: {analysis['total_evidence_relationships']}")
    print(f"Academic compliance score: {analysis['academic_compliance_score']:.1f}%")
    print(f"Van Evera compliant: {'YES' if analysis['van_evera_compliant'] else 'NO'}")
    print(f"Needs rebalancing: {'YES' if analysis['needs_rebalancing'] else 'NO'}")
    print()
    
    print("CURRENT vs TARGET DISTRIBUTION:")
    print(f"{'Test Type':<15} {'Current':<10} {'Target':<10} {'Gap':<10} {'Needs Fix'}")
    print("-"*60)
    
    for test_type, gap_info in analysis['distribution_gaps'].items():
        current = gap_info['current'] * 100
        target = gap_info['target'] * 100
        gap = gap_info['gap'] * 100
        needs_fix = 'YES' if gap_info['needs_adjustment'] else 'NO'
        
        print(f"{test_type:<15} {current:>7.1f}% {target:>8.1f}% {gap:>+8.1f}% {needs_fix:>9}")
    
    print()
    return analysis


def test_diagnostic_rebalancing():
    """Test the diagnostic rebalancing functionality"""
    print("VAN EVERA DIAGNOSTIC REBALANCER TEST")
    print("="*60)
    print()
    
    # Load test data
    print("Loading test graph data...")
    graph_data = load_test_graph()
    if not graph_data:
        return
    
    print(f"Loaded graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    print()
    
    # Analyze current distribution
    current_analysis = analyze_current_distribution(graph_data)
    
    if not current_analysis['needs_rebalancing']:
        print("Graph already meets Van Evera standards - no rebalancing needed!")
        return
    
    # Perform rebalancing
    print("PERFORMING VAN EVERA DIAGNOSTIC REBALANCING")
    print("="*50)
    print("Starting rebalancing process...")
    
    try:
        # Note: No LLM function provided, will use rule-based assessment
        rebalancing_result = rebalance_graph_diagnostics(
            graph_data, 
            llm_query_func=None  # Rule-based mode for testing
        )
        
        print("Rebalancing completed successfully!")
        print()
        
        # Generate and display report
        report = create_van_evera_rebalancing_report(rebalancing_result)
        print(report)
        
        # Validate results
        updated_graph = rebalancing_result['updated_graph_data']
        final_analysis = validate_diagnostic_distribution(updated_graph)
        
        print("\nFINAL VALIDATION:")
        print(f"Final compliance score: {final_analysis['academic_compliance_score']:.1f}%")
        print(f"Van Evera compliant: {'YES' if final_analysis['van_evera_compliant'] else 'NO'}")
        
        improvement = (final_analysis['academic_compliance_score'] - 
                      current_analysis['academic_compliance_score'])
        print(f"Improvement: {improvement:+.1f}%")
        
        # Save results
        output_path = "output_data/revolutions/rebalanced_graph.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_graph, f, indent=2, ensure_ascii=False)
        print(f"\nRebalanced graph saved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR during rebalancing: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_plugin_features():
    """Demonstrate various plugin features"""
    print("\nDEMONSTRATING PLUGIN FEATURES")
    print("="*40)
    
    # Load test data
    graph_data = load_test_graph()
    if not graph_data:
        return
    
    # Test 1: Validation only
    print("1. VALIDATION-ONLY MODE:")
    analysis = validate_diagnostic_distribution(graph_data)
    print(f"   Academic compliance: {analysis['academic_compliance_score']:.1f}%")
    print(f"   Total relationships: {analysis['total_evidence_relationships']}")
    print()
    
    # Test 2: Custom target distribution
    print("2. CUSTOM TARGET DISTRIBUTION:")
    custom_target = {
        'hoop': 0.30,           # 30% hoop tests
        'smoking_gun': 0.30,    # 30% smoking gun
        'doubly_decisive': 0.20, # 20% doubly decisive  
        'straw_in_wind': 0.20   # 20% straw-in-wind
    }
    
    try:
        custom_result = rebalance_graph_diagnostics(
            graph_data, 
            llm_query_func=None,
            target_distribution=custom_target
        )
        
        print(f"   Custom rebalancing completed!")
        print(f"   Compliance score: {custom_result['academic_compliance']['rebalanced_score']:.1f}%")
        print(f"   Rebalanced items: {custom_result['rebalancing_summary']['rebalanced_count']}")
        
    except Exception as e:
        print(f"   Custom rebalancing failed: {e}")
    
    print("\nPlugin demonstration completed!")


if __name__ == "__main__":
    try:
        test_diagnostic_rebalancing()
        demonstrate_plugin_features()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
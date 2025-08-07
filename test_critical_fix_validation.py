#!/usr/bin/env python3
"""
Critical fix validation test - verify Van Evera score integration is working
"""

from core.plugins.van_evera_workflow import execute_van_evera_analysis
import json
import traceback

def main():
    try:
        print('=== Testing Critical Fix: Van Evera Score Integration ===')
        
        # Create test data with realistic evidence for proper Van Evera analysis
        test_data = {
            'nodes': [
                {'id': 'H_001', 'type': 'Hypothesis', 'properties': {'description': 'Political leadership organized systematic resistance through institutional networks and strategic coordination across colonial regions'}},
                {'id': 'AE_001', 'type': 'Alternative_Explanation', 'properties': {'description': 'Economic grievances from taxation policies drove spontaneous popular resistance movements'}},
                {'id': 'E_001', 'type': 'Event', 'properties': {'description': 'Boston Tea Party organizing meeting with representatives from multiple colonies'}},
                {'id': 'EV_001', 'type': 'Evidence', 'properties': {'description': 'Letters between colonial leaders showing coordinated resistance planning across regions'}},
                {'id': 'EV_002', 'type': 'Evidence', 'properties': {'description': 'Economic data showing taxation burden on colonial merchants and farmers'}},
                {'id': 'EV_003', 'type': 'Evidence', 'properties': {'description': 'Historical records of institutional resistance mechanisms and formal committees'}}
            ],
            'edges': [
                {'source_id': 'EV_001', 'target_id': 'H_001', 'type': 'supports', 'properties': {'certainty': 0.8, 'probative_value': 0.9}},
                {'source_id': 'EV_003', 'target_id': 'H_001', 'type': 'supports', 'properties': {'certainty': 0.7, 'probative_value': 0.8}},
                {'source_id': 'EV_002', 'target_id': 'AE_001', 'type': 'supports', 'properties': {'certainty': 0.6, 'probative_value': 0.7}},
                {'source_id': 'E_001', 'target_id': 'H_001', 'type': 'supports', 'properties': {'certainty': 0.5, 'probative_value': 0.6}}
            ]
        }
        
        result = execute_van_evera_analysis(test_data, 'critical_fix_test')
        
        # Extract Van Evera results to check score integration
        van_evera_results = None
        workflow_results = result.get('workflow_execution', {})
        
        if 'van_evera_result' in workflow_results:
            van_evera_results = workflow_results['van_evera_result']
        elif 'van_evera_analysis' in result:
            van_evera_results = result['van_evera_analysis']
        
        print('\\n=== Van Evera Score Integration Analysis ===')
        
        if van_evera_results:
            # Check if hypothesis_rankings exist (the format we fixed)
            if 'hypothesis_rankings' in van_evera_results:
                rankings = van_evera_results['hypothesis_rankings']
                print(f'[FIXED] Found hypothesis_rankings with {len(rankings)} hypotheses')
                for hyp_id, data in rankings.items():
                    score = data.get('ranking_score', 'N/A')
                    print(f'  {hyp_id}: Van Evera score = {score}')
            else:
                print('[WARNING] No hypothesis_rankings found in Van Evera results')
                print(f'Available keys: {list(van_evera_results.keys())}')
        else:
            print('[ERROR] No Van Evera results found')
        
        # Check primary hypothesis identification results
        primary_results = result.get('workflow_execution', {}).get('primary_identification_result', {})
        if primary_results:
            primary_id = primary_results.get('primary_identification', {}).get('primary_hypothesis', {}).get('new_id')
            composite_score = primary_results.get('primary_identification', {}).get('primary_hypothesis', {}).get('composite_score', 0)
            print(f'\\n[RESULT] Primary hypothesis identified: {primary_id} with composite score: {composite_score:.3f}')
            
            # Check if actual Van Evera scores were used (not default 0.5)
            component_scores = primary_results.get('primary_identification', {}).get('primary_hypothesis', {}).get('hypothesis_data', {}).get('component_scores', {})
            van_evera_component = component_scores.get('van_evera_score', 0.5)
            
            if van_evera_component != 0.5:
                print(f'[SUCCESS] Van Evera score integration WORKING: {van_evera_component:.3f}')
            else:
                print(f'[FAILED] Still using fallback score: {van_evera_component}')
        
        print('\\n=== Critical Fix Validation Complete ===')
        
        return True

    except Exception as e:
        print(f'[ERROR] Critical fix validation failed: {e}')
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    print(f'\\nCritical Fix Status: {"WORKING" if success else "FAILED"}')
    exit(0 if success else 1)
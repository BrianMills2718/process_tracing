#!/usr/bin/env python3
"""
Debug ID mismatch between Van Evera testing and primary hypothesis identifier
"""

from core.plugins.van_evera_testing import VanEveraTestingPlugin
from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin
from core.plugins.base import PluginContext
import json

def main():
    # Simple test data with migrated Q_H1/Q_H2 format
    test_data = {
        'nodes': [
            {'id': 'Q_H1', 'type': 'Hypothesis', 'properties': {'description': 'Primary hypothesis'}},
            {'id': 'Q_H2', 'type': 'Hypothesis', 'properties': {'description': 'Alternative hypothesis'}},
            {'id': 'EV_001', 'type': 'Evidence', 'properties': {'description': 'Evidence 1'}},
            {'id': 'EV_002', 'type': 'Evidence', 'properties': {'description': 'Evidence 2'}}
        ],
        'edges': [
            {'source_id': 'EV_001', 'target_id': 'Q_H1', 'type': 'supports', 'properties': {'certainty': 0.8}},
            {'source_id': 'EV_002', 'target_id': 'Q_H2', 'type': 'supports', 'properties': {'certainty': 0.6}}
        ]
    }
    
    print('=== Debug ID Mismatch Issue ===')
    
    # Step 1: Run Van Evera testing to see what IDs it produces
    print('\\n1. Van Evera Testing Plugin Output:')
    context = PluginContext({'debug': True})
    van_evera_plugin = VanEveraTestingPlugin('van_evera_testing', context)
    
    try:
        van_evera_results = van_evera_plugin.execute({'graph_data': test_data})
        
        if 'hypothesis_rankings' in van_evera_results:
            rankings = van_evera_results['hypothesis_rankings']
            print(f'  Found hypothesis_rankings with keys: {list(rankings.keys())}')
            for hyp_id, data in rankings.items():
                score = data.get('ranking_score', 'N/A')
                print(f'    {hyp_id}: ranking_score = {score}')
        else:
            print('  No hypothesis_rankings found!')
            print(f'  Available keys: {list(van_evera_results.keys())}')
            
    except Exception as e:
        print(f'  Van Evera testing failed: {e}')
        return False
    
    # Step 2: Test primary hypothesis identifier with Van Evera results
    print('\\n2. Primary Hypothesis Identifier Test:')
    context2 = PluginContext({'debug': True})
    primary_plugin = PrimaryHypothesisIdentifierPlugin('primary_hypothesis_identifier', context2)
    
    try:
        primary_results = primary_plugin.execute({
            'graph_data': test_data,
            'van_evera_results': van_evera_results
        })
        
        primary_id = primary_results.get('primary_identification', {}).get('primary_hypothesis', {}).get('new_id', 'N/A')
        composite_score = primary_results.get('primary_identification', {}).get('primary_hypothesis', {}).get('composite_score', 0)
        
        print(f'  Selected primary: {primary_id} with composite score: {composite_score:.3f}')
        
        # Check component scores
        component_scores = primary_results.get('primary_identification', {}).get('primary_hypothesis', {}).get('hypothesis_data', {}).get('component_scores', {})
        van_evera_component = component_scores.get('van_evera_score', 0.5)
        
        if van_evera_component != 0.5:
            print(f'  SUCCESS: Van Evera score integration working: {van_evera_component:.3f}')
            return True
        else:
            print(f'  FAILED: Still using fallback score: {van_evera_component}')
            return False
            
    except Exception as e:
        print(f'  Primary identification failed: {e}')
        return False

if __name__ == '__main__':
    success = main()
    print(f'\\nID Mismatch Debug: {"RESOLVED" if success else "STILL BROKEN"}')
    exit(0 if success else 1)
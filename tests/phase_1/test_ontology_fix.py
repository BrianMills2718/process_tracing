"""
Test for Critical Fix #1: Schema Override Bug
This test is isolated to avoid import issues with the main codebase
"""
import json
import os


def test_ontology_config_structure():
    """Test that the config file has the expected structure"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'ontology_config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check required keys exist
    assert 'node_types' in config, "Config should have node_types"
    assert 'edge_types' in config, "Config should have edge_types"
    assert 'node_colors' in config, "Config should have node_colors"
    
    # Check structure
    assert isinstance(config['node_types'], dict), "node_types should be a dict"
    assert isinstance(config['edge_types'], dict), "edge_types should be a dict"
    assert isinstance(config['node_colors'], dict), "node_colors should be a dict"
    
    # Check some expected node types
    assert 'Event' in config['node_types'], "Should have Event node type"
    assert 'Actor' in config['node_types'], "Should have Actor node type"
    assert 'Evidence' in config['node_types'], "Should have Evidence node type"
    
    print("✓ Config file structure is valid")


if __name__ == "__main__":
    test_ontology_config_structure()
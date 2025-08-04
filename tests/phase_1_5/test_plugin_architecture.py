"""
Phase 1.5 Plugin Architecture Tests
Tests for plugin system foundation and critical plugins
"""
import pytest
import json
import os
import sys
import tempfile
import networkx as nx
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.plugins import (
    ProcessTracingPlugin, 
    PluginContext, 
    PluginRegistry,
    PluginExecutionError,
    PluginValidationError,
    register_plugin,
    get_global_registry
)
from core.plugins.config_validation import ConfigValidationPlugin
from core.plugins.graph_validation import GraphValidationPlugin
from core.plugins.evidence_balance import EvidenceBalancePlugin
from core.plugins.path_finder import PathFinderPlugin
from core.plugins.checkpoint import CheckpointPlugin
from core.plugins.workflow import PluginWorkflow, PHASE_1_CRITICAL_WORKFLOW


class TestPluginBase:
    """Test plugin system foundation"""
    
    def test_plugin_context_creation(self):
        """Plugin context should be created with config and data bus"""
        config = {'test_key': 'test_value'}
        context = PluginContext(config)
        
        assert context.get_config('test_key') == 'test_value'
        assert context.get_config('missing_key', 'default') == 'default'
        assert context.data_bus == {}
    
    def test_plugin_context_data_bus(self):
        """Plugin context data bus should store and retrieve data"""
        context = PluginContext({})
        
        context.set_data('key1', 'value1')
        assert context.get_data('key1') == 'value1'
        assert context.has_data('key1') is True
        assert context.has_data('missing_key') is False
        assert context.get_data('missing_key', 'default') == 'default'
    
    def test_plugin_registry_registration(self):
        """Plugin registry should register and create plugins"""
        registry = PluginRegistry()
        
        # Create test plugin class
        class TestPlugin(ProcessTracingPlugin):
            plugin_id = "test_plugin"
            
            def validate_input(self, data):
                pass
            
            def execute(self, data):
                return {'result': 'test'}
            
            def get_checkpoint_data(self):
                return {'test': 'checkpoint'}
        
        # Register plugin
        registry.register(TestPlugin)
        assert registry.has_plugin('test_plugin')
        assert 'test_plugin' in registry.list_plugins()
        
        # Create plugin instance
        context = PluginContext({})
        plugin = registry.create_plugin('test_plugin', context)
        assert isinstance(plugin, TestPlugin)
        assert plugin.id == 'test_plugin'
    
    def test_plugin_registry_conflicts(self):
        """Plugin registry should fail loud on conflicts"""
        registry = PluginRegistry()
        
        class TestPlugin1(ProcessTracingPlugin):
            plugin_id = "duplicate_id"
            def validate_input(self, data): pass
            def execute(self, data): return {}
            def get_checkpoint_data(self): return {}
        
        class TestPlugin2(ProcessTracingPlugin):
            plugin_id = "duplicate_id"
            def validate_input(self, data): pass
            def execute(self, data): return {}
            def get_checkpoint_data(self): return {}
        
        registry.register(TestPlugin1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(TestPlugin2)
    
    def test_plugin_registry_unknown_plugin(self):
        """Plugin registry should fail loud on unknown plugins"""
        registry = PluginRegistry()
        context = PluginContext({})
        
        with pytest.raises(KeyError, match="Unknown plugin"):
            registry.create_plugin('nonexistent_plugin', context)


class TestConfigValidationPlugin:
    """Test config validation plugin"""
    
    def test_config_validation_success(self):
        """Config validation should load valid config successfully"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'node_types': {'Event': {'description': 'test'}},
                'edge_types': {'causes': {'description': 'test'}},
                'node_colors': {'Event': '#blue'}
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            context = PluginContext({})
            plugin = ConfigValidationPlugin('config_validation', context)
            
            result = plugin.execute({'config_path': config_path})
            
            assert 'config' in result
            assert result['config'] == config_data
            assert result['stats']['node_types'] == 1
            assert result['stats']['edge_types'] == 1
            assert result['stats']['node_colors'] == 1
        finally:
            os.unlink(config_path)
    
    def test_config_validation_missing_file(self):
        """Config validation should fail loud on missing file"""
        context = PluginContext({})
        plugin = ConfigValidationPlugin('config_validation', context)
        
        with pytest.raises(PluginValidationError, match="Config file not found"):
            plugin.validate_input({'config_path': 'nonexistent.json'})
    
    def test_config_validation_invalid_json(self):
        """Config validation should fail loud on invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            config_path = f.name
        
        try:
            context = PluginContext({})
            plugin = ConfigValidationPlugin('config_validation', context)
            
            with pytest.raises(RuntimeError, match="Invalid JSON"):
                plugin.execute({'config_path': config_path})
        finally:
            os.unlink(config_path)


class TestGraphValidationPlugin:
    """Test graph validation plugin"""
    
    def test_graph_validation_success(self):
        """Graph validation should create working copy successfully"""
        # Create test graph
        original_graph = nx.DiGraph()
        original_graph.add_node('N1', type='event', data='original')
        original_graph.add_node('N2', type='event', data='original')
        original_graph.add_edge('N1', 'N2', relationship='causes')
        
        context = PluginContext({})
        plugin = GraphValidationPlugin('graph_validation', context)
        
        result = plugin.execute({'graph': original_graph})
        
        assert 'original_graph' in result
        assert 'working_graph' in result
        assert result['stats']['node_count'] == 2
        assert result['stats']['edge_count'] == 1
        
        # Verify original is unchanged after getting working copy
        assert original_graph.nodes['N1']['data'] == 'original'
        
        # Verify working copy is separate
        working_graph = result['working_graph']
        working_graph.nodes['N1']['data'] = 'modified'
        assert original_graph.nodes['N1']['data'] == 'original'  # Original unchanged
    
    def test_graph_validation_missing_attributes(self):
        """Graph validation should fail on missing node attributes"""
        graph = nx.DiGraph()
        graph.add_node('N1')  # Missing 'type' attribute
        
        context = PluginContext({})
        plugin = GraphValidationPlugin('graph_validation', context)
        
        with pytest.raises(PluginValidationError, match="Missing required node attributes"):
            plugin.validate_input({'graph': graph})


class TestEvidenceBalancePlugin:
    """Test evidence balance plugin"""
    
    def test_evidence_balance_correct_math(self):
        """Evidence balance should use correct probative value math"""
        hypothesis = {'balance': 0.0}
        evidence_list = [
            {'id': 'E1', 'probative_value': 0.8, 'description': 'Supporting evidence'},
            {'id': 'E2', 'probative_value': -0.3, 'description': 'Contradicting evidence'},
            {'id': 'E3', 'probative_value': 0.2, 'description': 'More support'}
        ]
        
        context = PluginContext({})
        plugin = EvidenceBalancePlugin('evidence_balance', context)
        
        result = plugin.execute({
            'hypothesis': hypothesis,
            'evidence_list': evidence_list
        })
        
        # Verify correct balance calculation: 0.0 + 0.8 - 0.3 + 0.2 = 0.7
        assert result['hypothesis']['balance'] == 0.7
        assert result['calculation_stats']['net_effect'] == 0.7
        assert result['calculation_stats']['positive_evidence_count'] == 2
        assert result['calculation_stats']['negative_evidence_count'] == 1
        assert result['calculation_stats']['total_positive_effect'] == 1.0
        assert result['calculation_stats']['total_negative_effect'] == -0.3
    
    def test_evidence_balance_validation(self):
        """Evidence balance should validate input properly"""
        context = PluginContext({})
        plugin = EvidenceBalancePlugin('evidence_balance', context)
        
        # Missing hypothesis
        with pytest.raises(PluginValidationError, match="Missing required key 'hypothesis'"):
            plugin.validate_input({'evidence_list': []})
        
        # Invalid probative value
        with pytest.raises(PluginValidationError, match="probative_value must be numeric"):
            plugin.validate_input({
                'hypothesis': {'balance': 0.0},
                'evidence_list': [{'probative_value': 'invalid'}]
            })


class TestPathFinderPlugin:
    """Test path finder plugin"""
    
    def test_path_finder_bounded_search(self):
        """Path finder should respect bounds and complete quickly"""
        # Create test graph with paths
        graph = nx.DiGraph()
        for i in range(10):
            graph.add_node(i, type='event')
            if i > 0:
                graph.add_edge(i-1, i)
        
        context = PluginContext({
            'path_finder.max_paths': 50,
            'path_finder.max_path_length': 15,
            'path_finder.max_execution_time': 2.0
        })
        plugin = PathFinderPlugin('path_finder', context)
        
        result = plugin.execute({
            'graph': graph,
            'source': 0,
            'target': 9
        })
        
        assert 'paths' in result
        assert len(result['paths']) == 1  # Only one path in linear chain
        assert result['paths'][0] == list(range(10))  # Path 0->1->2->...->9
        assert result['path_stats']['execution_time_seconds'] < 2.0
        assert result['path_stats']['total_paths'] == 1
    
    def test_path_finder_limits_applied(self):
        """Path finder should apply limits correctly"""
        # Create complete graph that would have many paths
        graph = nx.complete_graph(8, create_using=nx.DiGraph())
        for node in graph.nodes():
            graph.nodes[node]['type'] = 'event'
        
        context = PluginContext({
            'path_finder.max_paths': 10,
            'path_finder.max_path_length': 3,
            'path_finder.max_execution_time': 1.0
        })
        plugin = PathFinderPlugin('path_finder', context)
        
        result = plugin.execute({
            'graph': graph,
            'source': 0,
            'target': 7
        })
        
        # Should respect limits
        assert len(result['paths']) <= 10
        assert all(len(path) <= 4 for path in result['paths'])  # cutoff=3 means max length 4 (cutoff+1)
        assert result['path_stats']['execution_time_seconds'] < 1.0


class TestCheckpointPlugin:
    """Test checkpoint integration plugin"""
    
    def test_checkpoint_plugin_operations(self):
        """Checkpoint plugin should handle all operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            context = PluginContext({})
            plugin = CheckpointPlugin('checkpoint', context)
            
            # Initialize checkpoint manager
            result = plugin.execute({
                'operation': 'initialize',
                'case_id': 'test_case',
                'output_dir': tmpdir
            })
            assert result['success'] is True
            
            # Save checkpoint
            test_data = {'test': 'data'}
            result = plugin.execute({
                'operation': 'save',
                'stage_name': 'test_stage',
                'data': test_data
            })
            assert result['success'] is True
            
            # Check if can resume
            result = plugin.execute({
                'operation': 'can_resume',
                'stage_name': 'test_stage'
            })
            assert result['can_resume'] is True
            
            # Load checkpoint
            result = plugin.execute({
                'operation': 'load',
                'stage_name': 'test_stage'
            })
            assert result['data'] == test_data


class TestPluginWorkflow:
    """Test plugin workflow orchestration"""
    
    def test_workflow_execution(self):
        """Plugin workflow should execute steps in order"""
        # Register test plugins
        registry = get_global_registry()
        
        class SimplePlugin(ProcessTracingPlugin):
            plugin_id = "simple_test"
            
            def validate_input(self, data):
                # Accept either 'input' or 'output' key
                assert ('input' in data) or ('output' in data)
            
            def execute(self, data):
                if 'input' in data:
                    return {'output': f"processed_{data['input']}"}
                else:
                    # Process previous output
                    return {'output': f"processed_{data['output']}"}
            
            def get_checkpoint_data(self):
                return {'plugin': 'simple'}
        
        registry.register(SimplePlugin)
        
        try:
            context = PluginContext({})
            workflow = PluginWorkflow('test_workflow', context)
            
            steps = [
                {
                    'plugin_id': 'simple_test',
                    'input_key': None,
                    'output_key': 'result1',
                    'input_data': {'input': 'test1'}
                },
                {
                    'plugin_id': 'simple_test', 
                    'input_key': 'result1',
                    'output_key': 'result2'
                }
            ]
            
            results = workflow.execute_workflow(steps)
            
            assert 'result1' in results
            assert 'result2' in results
            assert results['result1']['output'] == 'processed_test1'
            assert results['result2']['output'] == 'processed_processed_test1'
        finally:
            registry.unregister('simple_test')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
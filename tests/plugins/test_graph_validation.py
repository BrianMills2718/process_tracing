"""Unit tests for GraphValidationPlugin - focused on plugin logic, not integration"""

import pytest
import networkx as nx
from unittest.mock import Mock, patch
from core.plugins.graph_validation import GraphValidationPlugin
from core.plugins.base import PluginContext, PluginValidationError

class TestGraphValidationPlugin:
    """Unit tests for GraphValidationPlugin"""
    
    @pytest.fixture
    def plugin_context(self):
        """Mock plugin context with minimal dependencies"""
        context = Mock(spec=PluginContext)
        context.config = {}
        context.data_bus = {}
        return context
        
    @pytest.fixture  
    def plugin(self, plugin_context):
        """Plugin instance with mocked context"""
        return GraphValidationPlugin("graph_validation", plugin_context)
        
    @pytest.fixture
    def minimal_graph(self):
        """Minimal test graph with known characteristics"""
        G = nx.DiGraph()
        G.add_node("H1", type="Hypothesis", description="Test hypothesis")
        G.add_node("E1", type="Evidence", description="Test evidence")
        G.add_edge("E1", "H1", type="supports")
        return G
        
    class TestInputValidation:
        """Test validate_input() method thoroughly"""
        
        def test_valid_input_accepted(self, plugin, minimal_graph):
            """Plugin accepts valid graph input"""
            # Should not raise exception
            plugin.validate_input(minimal_graph)
            
        def test_valid_dict_input_accepted(self, plugin, minimal_graph):
            """Plugin accepts valid dict with graph key"""
            graph_dict = {"graph": minimal_graph}
            plugin.validate_input(graph_dict)
            
        def test_invalid_input_rejected(self, plugin):
            """Plugin rejects invalid input with clear error"""
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input("invalid_input")
            assert "must be" in str(exc_info.value).lower()
            
        def test_invalid_dict_without_graph_rejected(self, plugin):
            """Plugin rejects dict without graph key"""
            with pytest.raises(PluginValidationError):
                plugin.validate_input({"data": "invalid"})
            
        def test_missing_attributes_detected(self, plugin):
            """Plugin detects missing node attributes"""
            G = nx.DiGraph() 
            G.add_node("H1")  # Missing 'type' attribute
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(G)
            assert "missing" in str(exc_info.value).lower()
                
    class TestExecutionLogic:
        """Test execute() method with various scenarios"""
        
        def test_execute_with_valid_input(self, plugin, minimal_graph):
            """Plugin executes successfully with valid input"""
            result = plugin.execute(minimal_graph)
            assert result is not None
            # Should return a dict with validated graph
            assert isinstance(result, dict)
            assert "graph" in result
            
        def test_execute_dict_input(self, plugin, minimal_graph):
            """Plugin executes successfully with dict input"""
            graph_dict = {"graph": minimal_graph}
            result = plugin.execute(graph_dict)
            assert result is not None
            assert isinstance(result, dict)
            assert "graph" in result
            
        def test_execute_idempotent(self, plugin, minimal_graph):
            """Plugin execution is idempotent (same input = same output)"""
            result1 = plugin.execute(minimal_graph.copy())
            result2 = plugin.execute(minimal_graph.copy())
            # Compare relevant aspects of results
            assert type(result1) == type(result2)
            assert "graph" in result1 and "graph" in result2
            
        def test_graph_immutability(self, plugin, minimal_graph):
            """Plugin creates immutable copy and doesn't modify original"""
            original_nodes = set(minimal_graph.nodes())
            original_edges = set(minimal_graph.edges())
            
            result = plugin.execute(minimal_graph)
            
            # Original graph should be unchanged
            assert set(minimal_graph.nodes()) == original_nodes
            assert set(minimal_graph.edges()) == original_edges
            
            # Result should contain a copy
            result_graph = result["graph"]
            assert result_graph is not minimal_graph  # Different object
            assert set(result_graph.nodes()) == original_nodes  # Same content
            
    class TestErrorHandling:
        """Test error conditions and edge cases"""
        
        def test_empty_graph_handling(self, plugin):
            """Plugin handles empty graph appropriately"""
            empty_graph = nx.DiGraph()
            # Should either work or fail gracefully with clear error
            try:
                result = plugin.execute(empty_graph)
                assert result is not None  # If it succeeds, result should be valid
                assert isinstance(result, dict)
            except PluginValidationError as e:
                assert len(str(e)) > 0  # Error message should be informative
                
        def test_invalid_graph_type(self, plugin):
            """Plugin rejects non-networkx graph types"""
            with pytest.raises(PluginValidationError):
                plugin.validate_input({"nodes": [], "edges": []})
                
        def test_node_without_required_type(self, plugin):
            """Plugin detects nodes missing required 'type' attribute"""
            G = nx.DiGraph()
            G.add_node("N1", description="Node without type")
            
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(G)
            assert "missing 'type'" in str(exc_info.value)
            
        def test_multiple_missing_attributes(self, plugin):
            """Plugin reports all missing node attributes"""
            G = nx.DiGraph()
            G.add_node("N1")  # Missing 'type'
            G.add_node("N2")  # Missing 'type'  
            G.add_node("N3", type="Event")  # Valid node
            
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(G)
            error_msg = str(exc_info.value)
            assert "N1" in error_msg
            assert "N2" in error_msg
            assert "missing 'type'" in error_msg
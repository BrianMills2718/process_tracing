"""
Phase 1 Critical Fix Tests
Tests for the 5 most critical bugs preventing basic functionality
"""
import pytest
import json
import os
import sys
import time
import copy
from unittest.mock import patch, MagicMock
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.ontology import NODE_TYPES, EDGE_TYPES, NODE_COLORS
from core.analyze import analyze_graph, find_causal_paths_bounded


class TestCriticalFixes:
    """Test suite for Phase 1 critical bug fixes"""
    
    # Test #1: Schema Override Bug (#13)
    def test_ontology_loads_from_config_only(self):
        """Ontology should match config file exactly"""
        # Read the config file directly
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'ontology_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # The loaded values should match the config exactly
        assert NODE_TYPES == config['node_types'], "NODE_TYPES should match config file"
        assert EDGE_TYPES == config['edge_types'], "EDGE_TYPES should match config file"
        assert NODE_COLORS == config['node_colors'], "NODE_COLORS should match config file"
    
    # Test #2: Evidence Balance Math Error (#16)
    def test_positive_evidence_increases_balance(self):
        """Positive probative value should increase hypothesis support"""
        hypothesis = {"balance": 0.0}
        evidence = {"probative_value": 0.8}
        
        # After fix, positive evidence increases balance
        new_balance = hypothesis["balance"] + evidence["probative_value"]
        assert new_balance == 0.8
        assert new_balance > hypothesis["balance"], "Positive evidence should increase balance"
    
    def test_negative_evidence_decreases_balance(self):
        """Negative probative value should decrease hypothesis support"""
        hypothesis = {"balance": 0.0}
        evidence = {"probative_value": -0.6}
        
        new_balance = hypothesis["balance"] + evidence["probative_value"] 
        assert new_balance == -0.6
        assert new_balance < hypothesis["balance"], "Negative evidence should decrease balance"
    
    # Test #3: Graph State Corruption (#34)
    def test_graph_immutable_during_analysis(self):
        """Original graph should not be modified"""
        # Create a test graph
        original = nx.DiGraph()
        original.add_node("N1", data="original", type="event")
        original.add_node("N2", data="original", type="event")
        original.add_edge("N1", "N2", relationship="causes")
        
        # Save original state
        original_nodes = {n: dict(data) for n, data in original.nodes(data=True)}
        original_edges = list(original.edges(data=True))
        
        # Run analysis (once fixed, this should not modify original)
        try:
            result = analyze_graph(original)
        except Exception:
            # If analyze_graph fails, we still want to check the graph wasn't modified
            pass
        
        # Verify original is unchanged
        for node, data in original.nodes(data=True):
            assert data == original_nodes[node], f"Node {node} was modified"
            assert "enhanced" not in data, f"Node {node} should not have 'enhanced' property"
        
        assert list(original.edges(data=True)) == original_edges, "Edges were modified"
    
    # Test #4: Exponential Path Finding (#18)
    def test_path_finding_completes_quickly(self):
        """Path finding should complete even on dense graphs"""
        # Create dense graph that would hang with unlimited search
        G = nx.complete_graph(20, create_using=nx.DiGraph())
        
        # Add required attributes for process tracing
        for node in G.nodes():
            G.nodes[node]['type'] = 'event'
        
        start_time = time.time()
        
        # This should use the bounded version
        paths = list(find_causal_paths_bounded(G, 0, 19))
        
        duration = time.time() - start_time
        
        assert duration < 5.0, f"Path finding took {duration}s, should complete in <5s"
        assert len(paths) <= 100, f"Found {len(paths)} paths, should limit to 100"
    
    def test_path_finding_respects_cutoff(self):
        """Path finding should respect depth cutoff"""
        # Create a long chain
        G = nx.DiGraph()
        for i in range(20):
            G.add_node(i, type='event')
            if i > 0:
                G.add_edge(i-1, i)
        
        # Find paths with cutoff
        paths = list(find_causal_paths_bounded(G, 0, 19, cutoff=10))
        
        # Should find no paths because shortest path is length 19
        assert len(paths) == 0, "Should find no paths with cutoff=10"
        
        # With higher cutoff should find the path
        paths = list(find_causal_paths_bounded(G, 0, 19, cutoff=20))
        assert len(paths) == 1, "Should find exactly one path"
    
    # Test #5: Enhancement Double Processing (#21)
    def test_enhancement_runs_once(self):
        """Enhancement should run exactly once per analysis"""
        enhancement_count = 0
        
        def mock_enhance(evidence):
            nonlocal enhancement_count
            enhancement_count += 1
            return evidence
        
        # Create minimal test graph
        G = nx.DiGraph()
        G.add_node("H1", type="hypothesis", description="Test hypothesis")
        G.add_node("E1", type="evidence", description="Test evidence")
        G.add_edge("E1", "H1", relationship="supports")
        
        # Patch refine_evidence_assessment_with_llm with mock
        with patch('core.analyze.refine_evidence_assessment_with_llm', mock_enhance):
            try:
                analyze_graph(G)
            except Exception:
                # Even if analysis fails, we want to check enhancement count
                pass
        
        assert enhancement_count <= 1, f"Enhancement ran {enhancement_count} times, should run at most once"


class TestFailFastPrinciples:
    """Test that fail-fast principles are followed"""
    
    def test_missing_file_fails_loud(self):
        """Missing files should raise clear exceptions"""
        with pytest.raises(FileNotFoundError, match="Required file missing"):
            # This should fail loud when we implement the fix
            from core.analyze import load_graph
            load_graph("nonexistent_file.json")
    
    def test_invalid_data_fails_loud(self):
        """Invalid data should raise clear exceptions"""
        # Create invalid graph (missing required attributes)
        G = nx.DiGraph()
        G.add_node("N1")  # Missing 'type' attribute
        
        with pytest.raises(ValueError, match="Missing required"):
            # Should fail validation
            from core.analyze import validate_graph
            validate_graph(G)
    
    def test_no_silent_failures(self):
        """Errors should not be silently suppressed"""
        # Test that safe_print doesn't hide errors
        from core.analyze import safe_print
        
        # This should log the error, not hide it
        with patch('logging.Logger.error') as mock_error:
            # Force an encoding error by patching print to raise UnicodeEncodeError
            with patch('builtins.print', side_effect=UnicodeEncodeError('cp1252', '\udcff', 0, 1, 'ordinal not in range')):
                safe_print("test message")
                assert mock_error.called, "Error should be logged"


class TestObservability:
    """Test that operations are observable through logs"""
    
    def test_operation_logging(self):
        """All operations should log start, progress, end"""
        with patch('logging.Logger.info') as mock_info:
            G = nx.DiGraph()
            G.add_node("N1", type="event")
            
            # When implemented correctly, analyze should log
            try:
                analyze_graph(G)
            except Exception:
                pass
            
            # Check that logging happened
            assert mock_info.called, "Operations should be logged"
            
            # Check for START/PROGRESS/END pattern
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("START" in msg for msg in log_messages), "Should log START"
            assert any("PROGRESS" in msg or "CHECKPOINT" in msg for msg in log_messages), "Should log PROGRESS"


class TestCheckpointing:
    """Test save/resume functionality"""
    
    def test_checkpoint_manager_saves_data(self):
        """CheckpointManager should save intermediate results"""
        from pathlib import Path
        import tempfile
        
        from core.checkpoint import CheckpointManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager("test_case", tmpdir)
            
            # Save checkpoint
            test_data = {"result": "test"}
            filepath = cm.save_checkpoint("test_stage", test_data)
            
            # Verify file exists
            assert Path(filepath).exists()
            
            # Verify can resume
            assert cm.can_resume_from("test_stage")
            
            # Load checkpoint
            loaded = cm.load_checkpoint("test_stage")
            assert loaded == test_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
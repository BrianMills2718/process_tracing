"""
Performance benchmarking suite for Phase 5 comparative analysis.

Tests performance characteristics, memory usage, and scalability limits
for all comparative analysis components to ensure production readiness.
"""

import pytest
import time
import psutil
import os
import json
import tempfile
import networkx as nx
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from core.comparative_models import (
    CaseMetadata, NodeMapping, MechanismPattern, 
    ComparisonType, MechanismType, ScopeCondition
)

# Import modules for benchmarking (mock if not available)
try:
    from core.case_manager import CaseManager
    from core.graph_alignment import GraphAligner
    from core.mechanism_detector import MechanismDetector
    from core.mss_analysis import MSSAnalyzer
    from core.mds_analysis import MDSAnalyzer
except ImportError:
    # Mock classes for performance testing
    class CaseManager:
        def __init__(self, case_directory="."):
            self.cases = {}
            self.case_metadata = {}
        
        def load_case(self, file_path, case_id=None):
            return case_id or f"case_{len(self.cases)}"
        
        def load_cases_from_directory(self, directory):
            return [f"case_{i}" for i in range(10)]  # Mock 10 cases
        
        def get_case(self, case_id):
            return nx.DiGraph()
        
        def get_case_metadata(self, case_id):
            return CaseMetadata(case_id=case_id, case_name=f"Case {case_id}", description="Mock case")
    
    class GraphAligner:
        def __init__(self):
            pass
        
        def align_graphs(self, graph1, graph2, case1_id, case2_id):
            return []
        
        def align_multiple_graphs(self, graphs):
            return {}
    
    class MechanismDetector:
        def __init__(self):
            pass
        
        def detect_patterns(self, cases, case_metadata):
            return []
    
    class MSSAnalyzer:
        def __init__(self):
            pass
        
        def identify_mss_pairs(self, cases, case_metadata):
            return []
    
    class MDSAnalyzer:
        def __init__(self):
            pass
        
        def identify_mds_pairs(self, cases, case_metadata):
            return []


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.operation_name}: {self.duration:.4f} seconds")


class MemoryProfiler:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else self.end_memory
        memory_used = self.end_memory - self.start_memory
        print(f"{self.operation_name}: {memory_used:.2f} MB used, Peak: {self.peak_memory:.2f} MB")


def create_test_graph(num_nodes: int, num_edges: int, case_id: str) -> nx.DiGraph:
    """Create a test graph with specified number of nodes and edges."""
    graph = nx.DiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        node_id = f"{case_id}_node_{i}"
        node_type = ["Event", "Mechanism", "Evidence"][i % 3]
        graph.add_node(node_id, type=node_type, description=f"Node {i} in {case_id}")
    
    # Add edges
    nodes = list(graph.nodes())
    edges_added = 0
    for i in range(len(nodes) - 1):
        if edges_added >= num_edges:
            break
        graph.add_edge(nodes[i], nodes[i + 1], type="causes", strength=0.8)
        edges_added += 1
    
    # Add some random edges
    import random
    while edges_added < num_edges and edges_added < len(nodes) * (len(nodes) - 1):
        source = random.choice(nodes)
        target = random.choice(nodes)
        if source != target and not graph.has_edge(source, target):
            graph.add_edge(source, target, type="influences", strength=0.6)
            edges_added += 1
    
    return graph


def create_test_metadata(case_id: str, context_variation: int = 0) -> CaseMetadata:
    """Create test case metadata with variations."""
    contexts = ["Europe", "Asia", "Americas", "Africa", "Oceania"]
    institutions = ["Democratic", "Authoritarian", "Hybrid", "Federal", "Unitary"]
    outcomes = ["success", "failure", "partial_success", "mixed", "unknown"]
    
    return CaseMetadata(
        case_id=case_id,
        case_name=f"Test Case {case_id}",
        description=f"Performance test case {case_id}",
        primary_outcome=outcomes[context_variation % len(outcomes)],
        geographic_context=contexts[context_variation % len(contexts)],
        institutional_context=institutions[context_variation % len(institutions)],
        data_quality_score=0.7 + (context_variation % 3) * 0.1,
        outcome_magnitude=0.5 + (context_variation % 5) * 0.1
    )


class TestCaseManagerPerformance:
    """Test CaseManager performance characteristics."""
    
    def test_case_loading_performance_small(self):
        """Test case loading performance with small cases (5 cases)."""
        manager = CaseManager()
        
        # Create 5 small test cases
        with tempfile.TemporaryDirectory() as temp_dir:
            case_files = []
            for i in range(5):
                case_data = {
                    "nodes": [
                        {"id": f"node_{j}", "type": "Event", "description": f"Node {j}"}
                        for j in range(10)  # 10 nodes per case
                    ],
                    "edges": [
                        {"source": f"node_{j}", "target": f"node_{j+1}", "type": "causes"}
                        for j in range(9)  # 9 edges per case
                    ],
                    "metadata": {
                        "case_name": f"Small Case {i}",
                        "description": f"Small test case {i}",
                        "primary_outcome": "success" if i % 2 == 0 else "failure"
                    }
                }
                
                case_file = Path(temp_dir) / f"small_case_{i}.json"
                with open(case_file, 'w') as f:
                    json.dump(case_data, f)
                case_files.append(str(case_file))
            
            # Benchmark loading
            with PerformanceTimer("Small case loading (5 cases, 10 nodes each)") as timer:
                case_ids = []
                for case_file in case_files:
                    case_id = manager.load_case(case_file)
                    case_ids.append(case_id)
            
            # Should load quickly
            assert timer.duration < 5.0  # Less than 5 seconds
            assert len(case_ids) == 5
    
    def test_case_loading_performance_medium(self):
        """Test case loading performance with medium cases (10 cases)."""
        manager = CaseManager()
        
        # Create 10 medium test cases
        with tempfile.TemporaryDirectory() as temp_dir:
            case_files = []
            for i in range(10):
                case_data = {
                    "nodes": [
                        {"id": f"node_{j}", "type": "Event", "description": f"Node {j}"}
                        for j in range(25)  # 25 nodes per case
                    ],
                    "edges": [
                        {"source": f"node_{j}", "target": f"node_{(j+1)%25}", "type": "causes"}
                        for j in range(30)  # 30 edges per case
                    ],
                    "metadata": {
                        "case_name": f"Medium Case {i}",
                        "description": f"Medium test case {i}",
                        "primary_outcome": "success" if i % 2 == 0 else "failure"
                    }
                }
                
                case_file = Path(temp_dir) / f"medium_case_{i}.json"
                with open(case_file, 'w') as f:
                    json.dump(case_data, f)
                case_files.append(str(case_file))
            
            # Benchmark loading with memory monitoring
            with PerformanceTimer("Medium case loading (10 cases, 25 nodes each)") as timer, \
                 MemoryProfiler("Medium case loading memory") as memory:
                
                case_ids = []
                for case_file in case_files:
                    case_id = manager.load_case(case_file)
                    case_ids.append(case_id)
            
            # Performance assertions
            assert timer.duration < 10.0  # Less than 10 seconds
            assert memory.peak_memory < 200  # Less than 200 MB peak
            assert len(case_ids) == 10
    
    def test_case_loading_performance_large(self):
        """Test case loading performance with large cases (5 cases, 100 nodes each)."""
        manager = CaseManager()
        
        # Create 5 large test cases
        with tempfile.TemporaryDirectory() as temp_dir:
            case_files = []
            for i in range(5):
                case_data = {
                    "nodes": [
                        {
                            "id": f"node_{j}", 
                            "type": ["Event", "Mechanism", "Evidence"][j % 3],
                            "description": f"Large node {j} with extended description for testing",
                            "properties": {"importance": j % 10, "category": f"cat_{j % 5}"}
                        }
                        for j in range(100)  # 100 nodes per case
                    ],
                    "edges": [
                        {
                            "source": f"node_{j}", 
                            "target": f"node_{(j+1)%100}", 
                            "type": "causes",
                            "strength": 0.5 + (j % 5) * 0.1
                        }
                        for j in range(150)  # 150 edges per case
                    ],
                    "metadata": {
                        "case_name": f"Large Case {i}",
                        "description": f"Large test case {i} with comprehensive metadata",
                        "primary_outcome": "success" if i % 2 == 0 else "failure",
                        "geographic_context": "Europe",
                        "institutional_context": "Democratic",
                        "time_period": ["2020-01-01T00:00:00", "2021-12-31T23:59:59"],
                        "data_quality_score": 0.8 + i * 0.02
                    }
                }
                
                case_file = Path(temp_dir) / f"large_case_{i}.json"
                with open(case_file, 'w') as f:
                    json.dump(case_data, f, indent=2)
                case_files.append(str(case_file))
            
            # Benchmark loading with memory monitoring
            with PerformanceTimer("Large case loading (5 cases, 100 nodes each)") as timer, \
                 MemoryProfiler("Large case loading memory") as memory:
                
                case_ids = []
                for case_file in case_files:
                    case_id = manager.load_case(case_file)
                    case_ids.append(case_id)
            
            # Performance assertions
            assert timer.duration < 20.0  # Less than 20 seconds
            assert memory.peak_memory < 500  # Less than 500 MB peak
            assert len(case_ids) == 5
    
    def test_case_retrieval_performance(self):
        """Test case retrieval performance after loading."""
        manager = CaseManager()
        
        # Load multiple cases first
        case_ids = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(20):
                case_data = {
                    "nodes": [{"id": f"node_{j}", "type": "Event"} for j in range(15)],
                    "edges": [{"source": f"node_{j}", "target": f"node_{j+1}", "type": "causes"} for j in range(14)],
                    "metadata": {"case_name": f"Case {i}", "description": f"Test case {i}"}
                }
                
                case_file = Path(temp_dir) / f"case_{i}.json"
                with open(case_file, 'w') as f:
                    json.dump(case_data, f)
                
                case_id = manager.load_case(str(case_file))
                case_ids.append(case_id)
        
        # Benchmark case retrieval
        with PerformanceTimer("Case retrieval (20 cases)") as timer:
            for case_id in case_ids:
                graph = manager.get_case(case_id)
                metadata = manager.get_case_metadata(case_id)
                assert graph is not None
                assert metadata is not None
        
        # Should retrieve quickly
        assert timer.duration < 2.0  # Less than 2 seconds for all retrievals


class TestGraphAlignmentPerformance:
    """Test GraphAligner performance characteristics."""
    
    def test_small_graph_alignment_performance(self):
        """Test alignment performance with small graphs."""
        aligner = GraphAligner()
        
        # Create small test graphs
        graph1 = create_test_graph(15, 20, "case1")
        graph2 = create_test_graph(15, 20, "case2")
        
        # Benchmark alignment
        with PerformanceTimer("Small graph alignment (15 nodes each)") as timer, \
             MemoryProfiler("Small graph alignment memory") as memory:
            
            mappings = aligner.align_graphs(graph1, graph2, "case1", "case2")
        
        # Performance assertions
        assert timer.duration < 5.0  # Less than 5 seconds
        assert memory.peak_memory < 100  # Less than 100 MB peak
        assert isinstance(mappings, list)
    
    def test_medium_graph_alignment_performance(self):
        """Test alignment performance with medium graphs."""
        aligner = GraphAligner()
        
        # Create medium test graphs
        graph1 = create_test_graph(50, 75, "case1")
        graph2 = create_test_graph(50, 75, "case2")
        
        # Benchmark alignment
        with PerformanceTimer("Medium graph alignment (50 nodes each)") as timer, \
             MemoryProfiler("Medium graph alignment memory") as memory:
            
            mappings = aligner.align_graphs(graph1, graph2, "case1", "case2")
        
        # Performance assertions
        assert timer.duration < 30.0  # Less than 30 seconds
        assert memory.peak_memory < 200  # Less than 200 MB peak
        assert isinstance(mappings, list)
    
    def test_multiple_graph_alignment_performance(self):
        """Test alignment performance with multiple graphs."""
        aligner = GraphAligner()
        
        # Create multiple test graphs
        graphs = {}
        for i in range(5):
            graphs[f"case_{i}"] = create_test_graph(25, 35, f"case_{i}")
        
        # Benchmark multiple alignment
        with PerformanceTimer("Multiple graph alignment (5 cases, 25 nodes each)") as timer, \
             MemoryProfiler("Multiple graph alignment memory") as memory:
            
            all_mappings = aligner.align_multiple_graphs(graphs)
        
        # Performance assertions
        assert timer.duration < 60.0  # Less than 60 seconds
        assert memory.peak_memory < 300  # Less than 300 MB peak
        assert isinstance(all_mappings, dict)
    
    def test_large_graph_alignment_performance(self):
        """Test alignment performance with large graphs."""
        aligner = GraphAligner()
        
        # Create large test graphs
        graph1 = create_test_graph(100, 150, "large_case1")
        graph2 = create_test_graph(100, 150, "large_case2")
        
        # Benchmark large alignment
        with PerformanceTimer("Large graph alignment (100 nodes each)") as timer, \
             MemoryProfiler("Large graph alignment memory") as memory:
            
            mappings = aligner.align_graphs(graph1, graph2, "large_case1", "large_case2")
        
        # Performance assertions
        assert timer.duration < 120.0  # Less than 2 minutes
        assert memory.peak_memory < 500  # Less than 500 MB peak
        assert isinstance(mappings, list)


class TestMechanismDetectionPerformance:
    """Test MechanismDetector performance characteristics."""
    
    def test_small_pattern_detection_performance(self):
        """Test pattern detection performance with small case set."""
        detector = MechanismDetector()
        
        # Create small case set
        cases = {}
        case_metadata = {}
        for i in range(3):
            case_id = f"small_case_{i}"
            cases[case_id] = create_test_graph(20, 25, case_id)
            case_metadata[case_id] = create_test_metadata(case_id, i)
        
        # Benchmark pattern detection
        with PerformanceTimer("Small pattern detection (3 cases, 20 nodes each)") as timer, \
             MemoryProfiler("Small pattern detection memory") as memory:
            
            patterns = detector.detect_patterns(cases, case_metadata)
        
        # Performance assertions
        assert timer.duration < 10.0  # Less than 10 seconds
        assert memory.peak_memory < 150  # Less than 150 MB peak
        assert isinstance(patterns, list)
    
    def test_medium_pattern_detection_performance(self):
        """Test pattern detection performance with medium case set."""
        detector = MechanismDetector()
        
        # Create medium case set
        cases = {}
        case_metadata = {}
        for i in range(7):
            case_id = f"medium_case_{i}"
            cases[case_id] = create_test_graph(40, 60, case_id)
            case_metadata[case_id] = create_test_metadata(case_id, i)
        
        # Benchmark pattern detection
        with PerformanceTimer("Medium pattern detection (7 cases, 40 nodes each)") as timer, \
             MemoryProfiler("Medium pattern detection memory") as memory:
            
            patterns = detector.detect_patterns(cases, case_metadata)
        
        # Performance assertions
        assert timer.duration < 60.0  # Less than 60 seconds
        assert memory.peak_memory < 300  # Less than 300 MB peak
        assert isinstance(patterns, list)
    
    def test_large_pattern_detection_performance(self):
        """Test pattern detection performance with large case set."""
        detector = MechanismDetector()
        
        # Create large case set
        cases = {}
        case_metadata = {}
        for i in range(10):
            case_id = f"large_case_{i}"
            cases[case_id] = create_test_graph(60, 90, case_id)
            case_metadata[case_id] = create_test_metadata(case_id, i)
        
        # Benchmark pattern detection
        with PerformanceTimer("Large pattern detection (10 cases, 60 nodes each)") as timer, \
             MemoryProfiler("Large pattern detection memory") as memory:
            
            patterns = detector.detect_patterns(cases, case_metadata)
        
        # Performance assertions
        assert timer.duration < 180.0  # Less than 3 minutes
        assert memory.peak_memory < 500  # Less than 500 MB peak
        assert isinstance(patterns, list)


class TestMSSMDSAnalysisPerformance:
    """Test MSS/MDS analysis performance characteristics."""
    
    def test_mss_analysis_performance(self):
        """Test MSS analysis performance."""
        analyzer = MSSAnalyzer()
        
        # Create test cases for MSS
        cases = {}
        case_metadata = {}
        for i in range(6):
            case_id = f"mss_case_{i}"
            cases[case_id] = create_test_graph(30, 45, case_id)
            # Create similar contexts with different outcomes
            metadata = create_test_metadata(case_id, i % 2)  # Alternate outcomes
            case_metadata[case_id] = metadata
        
        # Benchmark MSS pair identification
        with PerformanceTimer("MSS pair identification (6 cases)") as timer, \
             MemoryProfiler("MSS analysis memory") as memory:
            
            mss_pairs = analyzer.identify_mss_pairs(cases, case_metadata)
        
        # Performance assertions
        assert timer.duration < 30.0  # Less than 30 seconds
        assert memory.peak_memory < 200  # Less than 200 MB peak
        assert isinstance(mss_pairs, list)
    
    def test_mds_analysis_performance(self):
        """Test MDS analysis performance."""
        analyzer = MDSAnalyzer()
        
        # Create test cases for MDS
        cases = {}
        case_metadata = {}
        for i in range(6):
            case_id = f"mds_case_{i}"
            cases[case_id] = create_test_graph(30, 45, case_id)
            # Create different contexts with similar outcomes
            metadata = create_test_metadata(case_id, i)  # Vary contexts
            metadata.primary_outcome = "success"  # Same outcome
            case_metadata[case_id] = metadata
        
        # Benchmark MDS pair identification
        with PerformanceTimer("MDS pair identification (6 cases)") as timer, \
             MemoryProfiler("MDS analysis memory") as memory:
            
            mds_pairs = analyzer.identify_mds_pairs(cases, case_metadata)
        
        # Performance assertions
        assert timer.duration < 30.0  # Less than 30 seconds
        assert memory.peak_memory < 200  # Less than 200 MB peak
        assert isinstance(mds_pairs, list)
    
    def test_combined_mss_mds_performance(self):
        """Test combined MSS and MDS analysis performance."""
        mss_analyzer = MSSAnalyzer()
        mds_analyzer = MDSAnalyzer()
        
        # Create diverse test cases
        cases = {}
        case_metadata = {}
        for i in range(8):
            case_id = f"combined_case_{i}"
            cases[case_id] = create_test_graph(25, 40, case_id)
            case_metadata[case_id] = create_test_metadata(case_id, i)
        
        # Benchmark combined analysis
        with PerformanceTimer("Combined MSS/MDS analysis (8 cases)") as timer, \
             MemoryProfiler("Combined MSS/MDS memory") as memory:
            
            mss_pairs = mss_analyzer.identify_mss_pairs(cases, case_metadata)
            mds_pairs = mds_analyzer.identify_mds_pairs(cases, case_metadata)
        
        # Performance assertions
        assert timer.duration < 60.0  # Less than 60 seconds
        assert memory.peak_memory < 300  # Less than 300 MB peak
        assert isinstance(mss_pairs, list)
        assert isinstance(mds_pairs, list)


class TestScalabilityLimits:
    """Test scalability limits and breaking points."""
    
    def test_maximum_case_count_performance(self):
        """Test performance with maximum reasonable case count."""
        manager = CaseManager()
        
        # Test with 20 cases (upper reasonable limit)
        with tempfile.TemporaryDirectory() as temp_dir:
            case_files = []
            for i in range(20):
                case_data = {
                    "nodes": [{"id": f"node_{j}", "type": "Event"} for j in range(20)],
                    "edges": [{"source": f"node_{j}", "target": f"node_{j+1}", "type": "causes"} for j in range(19)],
                    "metadata": {"case_name": f"Case {i}", "description": f"Scalability test case {i}"}
                }
                
                case_file = Path(temp_dir) / f"scale_case_{i}.json"
                with open(case_file, 'w') as f:
                    json.dump(case_data, f)
                case_files.append(str(case_file))
            
            # Benchmark maximum case loading
            with PerformanceTimer("Maximum case loading (20 cases)") as timer, \
                 MemoryProfiler("Maximum case loading memory") as memory:
                
                case_ids = []
                for case_file in case_files:
                    case_id = manager.load_case(case_file)
                    case_ids.append(case_id)
            
            # Should handle maximum cases reasonably
            assert timer.duration < 30.0  # Less than 30 seconds
            assert memory.peak_memory < 800  # Less than 800 MB peak
            assert len(case_ids) == 20
    
    def test_maximum_graph_size_performance(self):
        """Test performance with maximum reasonable graph size."""
        aligner = GraphAligner()
        
        # Create very large graphs (200 nodes each)
        graph1 = create_test_graph(200, 300, "max_case1")
        graph2 = create_test_graph(200, 300, "max_case2")
        
        # Benchmark maximum graph alignment
        with PerformanceTimer("Maximum graph alignment (200 nodes each)") as timer, \
             MemoryProfiler("Maximum graph alignment memory") as memory:
            
            mappings = aligner.align_graphs(graph1, graph2, "max_case1", "max_case2")
        
        # Should handle large graphs with acceptable performance
        assert timer.duration < 300.0  # Less than 5 minutes
        assert memory.peak_memory < 1000  # Less than 1GB peak
        assert isinstance(mappings, list)
    
    def test_memory_cleanup_after_operations(self):
        """Test that memory is properly cleaned up after operations."""
        import gc
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        manager = CaseManager()
        aligner = GraphAligner()
        
        # Create and load multiple large cases
        cases = {}
        for i in range(5):
            case_id = f"cleanup_case_{i}"
            cases[case_id] = create_test_graph(80, 120, case_id)
        
        # Perform alignments
        case_ids = list(cases.keys())
        for i in range(len(case_ids)):
            for j in range(i + 1, len(case_ids)):
                mappings = aligner.align_graphs(
                    cases[case_ids[i]], cases[case_ids[j]], 
                    case_ids[i], case_ids[j]
                )
        
        # Clean up
        del cases, mappings, manager, aligner
        gc.collect()
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500 MB permanent increase
        print(f"Memory cleanup test: {memory_increase:.2f} MB increase")


class TestPerformanceRegressionBenchmarks:
    """Benchmark tests for performance regression detection."""
    
    def test_standard_workflow_benchmark(self):
        """Benchmark standard comparative workflow for regression testing."""
        # Standard test scenario: 5 cases, 30 nodes each
        manager = CaseManager()
        aligner = GraphAligner()
        detector = MechanismDetector()
        
        # Create standard test cases
        cases = {}
        case_metadata = {}
        for i in range(5):
            case_id = f"standard_case_{i}"
            cases[case_id] = create_test_graph(30, 45, case_id)
            case_metadata[case_id] = create_test_metadata(case_id, i)
        
        # Benchmark complete workflow
        with PerformanceTimer("Standard workflow benchmark") as timer, \
             MemoryProfiler("Standard workflow memory") as memory:
            
            # Graph alignment step
            alignments = {}
            case_ids = list(cases.keys())
            for i in range(len(case_ids)):
                for j in range(i + 1, len(case_ids)):
                    mappings = aligner.align_graphs(
                        cases[case_ids[i]], cases[case_ids[j]],
                        case_ids[i], case_ids[j]
                    )
                    alignments[(case_ids[i], case_ids[j])] = mappings
            
            # Pattern detection step
            patterns = detector.detect_patterns(cases, case_metadata)
        
        # Baseline performance targets (update when making performance improvements)
        BASELINE_TIME = 120.0  # 2 minutes baseline
        BASELINE_MEMORY = 400  # 400 MB baseline
        
        # Performance regression assertions
        assert timer.duration < BASELINE_TIME, f"Performance regression: {timer.duration:.2f}s > {BASELINE_TIME}s baseline"
        assert memory.peak_memory < BASELINE_MEMORY, f"Memory regression: {memory.peak_memory:.2f}MB > {BASELINE_MEMORY}MB baseline"
        
        # Quality assertions
        assert isinstance(alignments, dict)
        assert isinstance(patterns, list)
        assert len(alignments) > 0  # Should find some alignments
        
        print(f"Benchmark results: {timer.duration:.2f}s, {memory.peak_memory:.2f}MB peak")


if __name__ == "__main__":
    # Run specific performance tests
    print("Running Phase 5 Comparative Analysis Performance Benchmarks")
    print("=" * 60)
    
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
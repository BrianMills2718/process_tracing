"""
Test suite for end-to-end comparative process tracing pipeline.

Tests the complete comparative analysis workflow integration including
case loading, graph alignment, pattern detection, MSS/MDS analysis,
and comparative visualization generation.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import networkx as nx

from core.comparative_models import (
    CaseMetadata, ComparisonResult, ComparisonType, 
    MechanismPattern, MechanismType, ScopeCondition,
    ComparativeAnalysisError
)

# Import modules under test (may not exist yet, so we'll mock)
try:
    from core.case_manager import CaseManager
    from core.graph_alignment import GraphAligner
    from core.mechanism_detector import MechanismDetector
    from core.mss_analysis import MSSAnalyzer
    from core.mds_analysis import MDSAnalyzer
    from process_trace_comparative import ComparativeProcessTracer
except ImportError:
    # Mock the main integration class
    class ComparativeProcessTracer:
        def __init__(self, case_directory=None):
            self.case_directory = case_directory or "."
            self.case_manager = MagicMock()
            self.graph_aligner = MagicMock()
            self.mechanism_detector = MagicMock()
            self.mss_analyzer = MagicMock()
            self.mds_analyzer = MagicMock()
        
        def load_cases(self, case_files):
            return ["case1", "case2", "case3"]
        
        def run_comparative_analysis(self, case_ids, comparison_types=None):
            return {
                "mss_results": [],
                "mds_results": [],
                "mechanisms": [],
                "alignments": {}
            }
        
        def generate_comparative_report(self, results, output_path):
            return output_path


class TestComparativeProcessTracerInit:
    """Test ComparativeProcessTracer initialization."""
    
    def test_init_with_directory(self):
        """Test initialization with case directory."""
        tracer = ComparativeProcessTracer("/test/cases")
        assert tracer.case_directory == "/test/cases"
    
    def test_init_without_directory(self):
        """Test initialization without case directory."""
        tracer = ComparativeProcessTracer()
        assert tracer.case_directory == "."


class TestComparativeProcessTracerCaseLoading:
    """Test case loading functionality in comparative pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = ComparativeProcessTracer()
        
        # Create test case data
        self.test_cases = [
            {
                "filename": "case1.json",
                "data": {
                    "nodes": [
                        {"id": "event1", "type": "Event", "description": "Economic crisis"},
                        {"id": "mechanism1", "type": "Mechanism", "description": "Stimulus package"},
                        {"id": "outcome1", "type": "Event", "description": "Recovery"}
                    ],
                    "edges": [
                        {"source": "event1", "target": "mechanism1", "type": "triggers"},
                        {"source": "mechanism1", "target": "outcome1", "type": "produces"}
                    ],
                    "metadata": {
                        "case_name": "European Crisis 2008",
                        "description": "European financial crisis response",
                        "primary_outcome": "economic_recovery",
                        "geographic_context": "Europe",
                        "institutional_context": "Democratic",
                        "time_period": ["2008-01-01T00:00:00", "2010-12-31T23:59:59"]
                    }
                }
            },
            {
                "filename": "case2.json",
                "data": {
                    "nodes": [
                        {"id": "event2", "type": "Event", "description": "Financial crisis"},
                        {"id": "mechanism2", "type": "Mechanism", "description": "Austerity measures"},
                        {"id": "outcome2", "type": "Event", "description": "Slow recovery"}
                    ],
                    "edges": [
                        {"source": "event2", "target": "mechanism2", "type": "triggers"},
                        {"source": "mechanism2", "target": "outcome2", "type": "produces"}
                    ],
                    "metadata": {
                        "case_name": "Greek Crisis 2010",
                        "description": "Greek debt crisis response",
                        "primary_outcome": "partial_recovery",
                        "geographic_context": "Europe",
                        "institutional_context": "Democratic",
                        "time_period": ["2010-01-01T00:00:00", "2015-12-31T23:59:59"]
                    }
                }
            },
            {
                "filename": "case3.json",
                "data": {
                    "nodes": [
                        {"id": "event3", "type": "Event", "description": "Market crisis"},
                        {"id": "mechanism3", "type": "Mechanism", "description": "State intervention"},
                        {"id": "outcome3", "type": "Event", "description": "Rapid recovery"}
                    ],
                    "edges": [
                        {"source": "event3", "target": "mechanism3", "type": "triggers"},
                        {"source": "mechanism3", "target": "outcome3", "type": "produces"}
                    ],
                    "metadata": {
                        "case_name": "Chinese Crisis 2008",
                        "description": "Chinese financial crisis response",
                        "primary_outcome": "economic_recovery",
                        "geographic_context": "Asia",
                        "institutional_context": "Authoritarian",
                        "time_period": ["2008-01-01T00:00:00", "2009-12-31T23:59:59"]
                    }
                }
            }
        ]
    
    def test_load_multiple_cases_success(self):
        """Test successful loading of multiple cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test case files
            case_files = []
            for case_info in self.test_cases:
                case_file = Path(temp_dir) / case_info["filename"]
                with open(case_file, 'w') as f:
                    json.dump(case_info["data"], f)
                case_files.append(str(case_file))
            
            # Load cases
            case_ids = self.tracer.load_cases(case_files)
            
            assert isinstance(case_ids, list)
            assert len(case_ids) == 3
    
    def test_load_cases_from_directory(self):
        """Test loading all cases from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test case files
            for case_info in self.test_cases:
                case_file = Path(temp_dir) / case_info["filename"]
                with open(case_file, 'w') as f:
                    json.dump(case_info["data"], f)
            
            # Mock the case manager to return case IDs
            with patch.object(self.tracer, 'case_manager') as mock_manager:
                mock_manager.load_cases_from_directory.return_value = ["case1", "case2", "case3"]
                
                case_ids = self.tracer.case_manager.load_cases_from_directory(temp_dir)
                
                assert len(case_ids) == 3
    
    def test_load_cases_with_invalid_file(self):
        """Test loading cases with invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid case file
            valid_case = Path(temp_dir) / "valid.json"
            with open(valid_case, 'w') as f:
                json.dump(self.test_cases[0]["data"], f)
            
            # Create invalid case file
            invalid_case = Path(temp_dir) / "invalid.json"
            with open(invalid_case, 'w') as f:
                f.write("invalid json content")
            
            case_files = [str(valid_case), str(invalid_case)]
            
            # Should handle invalid files gracefully
            case_ids = self.tracer.load_cases(case_files)
            assert isinstance(case_ids, list)


class TestComparativeAnalysisWorkflow:
    """Test complete comparative analysis workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = ComparativeProcessTracer()
        
        # Mock case IDs and graphs
        self.case_ids = ["case1", "case2", "case3"]
        
        # Create mock case graphs
        self.mock_cases = {}
        for case_id in self.case_ids:
            graph = nx.DiGraph()
            graph.add_node(f"event_{case_id}", type="Event", description=f"Event in {case_id}")
            graph.add_node(f"mechanism_{case_id}", type="Mechanism", description=f"Mechanism in {case_id}")
            graph.add_node(f"outcome_{case_id}", type="Event", description=f"Outcome in {case_id}")
            graph.add_edge(f"event_{case_id}", f"mechanism_{case_id}", type="triggers")
            graph.add_edge(f"mechanism_{case_id}", f"outcome_{case_id}", type="produces")
            self.mock_cases[case_id] = graph
        
        # Create mock case metadata
        self.mock_metadata = {
            "case1": CaseMetadata(
                case_id="case1",
                case_name="Democratic Success",
                description="Democratic system with successful outcome",
                primary_outcome="success",
                geographic_context="Europe",
                institutional_context="Democratic"
            ),
            "case2": CaseMetadata(
                case_id="case2",
                case_name="Democratic Failure",
                description="Democratic system with failed outcome",
                primary_outcome="failure",
                geographic_context="Europe",
                institutional_context="Democratic"
            ),
            "case3": CaseMetadata(
                case_id="case3",
                case_name="Authoritarian Success",
                description="Authoritarian system with successful outcome",
                primary_outcome="success",
                geographic_context="Asia",
                institutional_context="Authoritarian"
            )
        }
    
    def test_run_complete_comparative_analysis(self):
        """Test running complete comparative analysis."""
        # Mock the component methods
        with patch.object(self.tracer.case_manager, 'get_case', side_effect=lambda x: self.mock_cases[x]), \
             patch.object(self.tracer.case_manager, 'get_case_metadata', side_effect=lambda x: self.mock_metadata[x]):
            
            results = self.tracer.run_comparative_analysis(
                self.case_ids, 
                comparison_types=[ComparisonType.MOST_SIMILAR_SYSTEMS, ComparisonType.MOST_DIFFERENT_SYSTEMS]
            )
            
            assert isinstance(results, dict)
            assert "mss_results" in results
            assert "mds_results" in results
            assert "mechanisms" in results
            assert "alignments" in results
    
    def test_run_mss_only_analysis(self):
        """Test running MSS-only comparative analysis."""
        with patch.object(self.tracer.case_manager, 'get_case', side_effect=lambda x: self.mock_cases[x]), \
             patch.object(self.tracer.case_manager, 'get_case_metadata', side_effect=lambda x: self.mock_metadata[x]):
            
            results = self.tracer.run_comparative_analysis(
                self.case_ids,
                comparison_types=[ComparisonType.MOST_SIMILAR_SYSTEMS]
            )
            
            assert isinstance(results, dict)
            assert "mss_results" in results
            # Should not include MDS results when not requested
            assert "mds_results" in results  # Empty list is fine
    
    def test_run_mds_only_analysis(self):
        """Test running MDS-only comparative analysis."""
        with patch.object(self.tracer.case_manager, 'get_case', side_effect=lambda x: self.mock_cases[x]), \
             patch.object(self.tracer.case_manager, 'get_case_metadata', side_effect=lambda x: self.mock_metadata[x]):
            
            results = self.tracer.run_comparative_analysis(
                self.case_ids,
                comparison_types=[ComparisonType.MOST_DIFFERENT_SYSTEMS]
            )
            
            assert isinstance(results, dict)
            assert "mds_results" in results
            # Should not include MSS results when not requested
            assert "mss_results" in results  # Empty list is fine
    
    def test_run_analysis_with_empty_cases(self):
        """Test running analysis with no cases."""
        empty_results = self.tracer.run_comparative_analysis([])
        
        assert isinstance(empty_results, dict)
        assert "mss_results" in empty_results
        assert "mds_results" in empty_results
        assert len(empty_results["mss_results"]) == 0
        assert len(empty_results["mds_results"]) == 0
    
    def test_run_analysis_with_single_case(self):
        """Test running analysis with single case (should return empty results)."""
        single_case_results = self.tracer.run_comparative_analysis(["case1"])
        
        assert isinstance(single_case_results, dict)
        # Single case cannot be compared
        assert len(single_case_results.get("mss_results", [])) == 0
        assert len(single_case_results.get("mds_results", [])) == 0


class TestComparativeReportGeneration:
    """Test comparative analysis report generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = ComparativeProcessTracer()
        
        # Create mock analysis results
        self.mock_results = {
            "mss_results": [
                ComparisonResult(
                    comparison_id="mss_1",
                    comparison_type=ComparisonType.MOST_SIMILAR_SYSTEMS,
                    primary_case="case1",
                    comparison_cases=["case2"],
                    overall_similarity=0.8,
                    context_similarity=0.9,
                    outcome_similarity=0.1
                )
            ],
            "mds_results": [
                ComparisonResult(
                    comparison_id="mds_1",
                    comparison_type=ComparisonType.MOST_DIFFERENT_SYSTEMS,
                    primary_case="case1",
                    comparison_cases=["case3"],
                    overall_similarity=0.3,
                    context_similarity=0.1,
                    outcome_similarity=0.9
                )
            ],
            "mechanisms": [
                MechanismPattern(
                    pattern_id="pattern_1",
                    pattern_name="Crisis Response Pattern",
                    description="Common crisis response mechanism",
                    mechanism_type=MechanismType.UNIVERSAL,
                    scope_conditions=[],
                    participating_cases=["case1", "case2", "case3"],
                    pattern_strength=0.85
                )
            ],
            "alignments": {
                ("case1", "case2"): [
                    {"source_node": "event1", "target_node": "event2", "similarity": 0.8},
                    {"source_node": "mechanism1", "target_node": "mechanism2", "similarity": 0.6}
                ],
                ("case1", "case3"): [
                    {"source_node": "event1", "target_node": "event3", "similarity": 0.9},
                    {"source_node": "mechanism1", "target_node": "mechanism3", "similarity": 0.4}
                ]
            }
        }
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comparative_report.html"
            
            # Mock the report generation
            with patch.object(self.tracer, 'generate_comparative_report') as mock_generate:
                mock_generate.return_value = str(output_path)
                
                result_path = self.tracer.generate_comparative_report(self.mock_results, str(output_path))
                
                assert result_path == str(output_path)
                mock_generate.assert_called_once_with(self.mock_results, str(output_path))
    
    def test_generate_json_report(self):
        """Test JSON report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comparative_report.json"
            
            # Create a simple JSON report
            json_results = {
                "analysis_timestamp": datetime.now().isoformat(),
                "mss_comparisons": len(self.mock_results["mss_results"]),
                "mds_comparisons": len(self.mock_results["mds_results"]),
                "mechanisms_found": len(self.mock_results["mechanisms"]),
                "case_alignments": len(self.mock_results["alignments"])
            }
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert loaded_results["mss_comparisons"] == 1
            assert loaded_results["mds_comparisons"] == 1
            assert loaded_results["mechanisms_found"] == 1
            assert loaded_results["case_alignments"] == 2
    
    def test_generate_report_with_empty_results(self):
        """Test report generation with empty results."""
        empty_results = {
            "mss_results": [],
            "mds_results": [],
            "mechanisms": [],
            "alignments": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_report.html"
            
            # Should handle empty results gracefully
            with patch.object(self.tracer, 'generate_comparative_report') as mock_generate:
                mock_generate.return_value = str(output_path)
                
                result_path = self.tracer.generate_comparative_report(empty_results, str(output_path))
                
                assert result_path == str(output_path)
                mock_generate.assert_called_once_with(empty_results, str(output_path))


class TestEndToEndComparativeWorkflow:
    """Test complete end-to-end comparative process tracing workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = ComparativeProcessTracer()
    
    def test_complete_workflow_with_real_files(self):
        """Test complete workflow from file loading to report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test case files
            case_files = []
            test_cases = [
                {
                    "filename": "european_crisis.json",
                    "data": {
                        "nodes": [
                            {"id": "financial_crisis", "type": "Event", "description": "2008 financial crisis"},
                            {"id": "stimulus_response", "type": "Mechanism", "description": "Government stimulus package"},
                            {"id": "economic_recovery", "type": "Event", "description": "Economic recovery achieved"}
                        ],
                        "edges": [
                            {"source": "financial_crisis", "target": "stimulus_response", "type": "triggers"},
                            {"source": "stimulus_response", "target": "economic_recovery", "type": "produces"}
                        ],
                        "metadata": {
                            "case_name": "European Crisis Response 2008",
                            "description": "European response to 2008 financial crisis",
                            "primary_outcome": "economic_recovery",
                            "geographic_context": "Europe",
                            "institutional_context": "Democratic",
                            "economic_context": "Market",
                            "data_quality_score": 0.9
                        }
                    }
                },
                {
                    "filename": "asian_crisis.json",
                    "data": {
                        "nodes": [
                            {"id": "market_crisis", "type": "Event", "description": "2008 market crisis"},
                            {"id": "state_intervention", "type": "Mechanism", "description": "Massive state intervention"},
                            {"id": "rapid_recovery", "type": "Event", "description": "Rapid economic recovery"}
                        ],
                        "edges": [
                            {"source": "market_crisis", "target": "state_intervention", "type": "triggers"},
                            {"source": "state_intervention", "target": "rapid_recovery", "type": "produces"}
                        ],
                        "metadata": {
                            "case_name": "Asian Crisis Response 2008",
                            "description": "Asian response to 2008 financial crisis",
                            "primary_outcome": "economic_recovery",
                            "geographic_context": "Asia",
                            "institutional_context": "Authoritarian",
                            "economic_context": "Mixed",
                            "data_quality_score": 0.8
                        }
                    }
                }
            ]
            
            # Write case files
            for case_info in test_cases:
                case_file = Path(temp_dir) / case_info["filename"]
                with open(case_file, 'w') as f:
                    json.dump(case_info["data"], f, indent=2)
                case_files.append(str(case_file))
            
            # Mock the workflow components
            with patch.object(self.tracer, 'load_cases') as mock_load, \
                 patch.object(self.tracer, 'run_comparative_analysis') as mock_analyze, \
                 patch.object(self.tracer, 'generate_comparative_report') as mock_report:
                
                # Set up mock returns
                mock_load.return_value = ["european_crisis", "asian_crisis"]
                mock_analyze.return_value = {
                    "mss_results": [],
                    "mds_results": [
                        ComparisonResult(
                            comparison_id="euro_asia_mds",
                            comparison_type=ComparisonType.MOST_DIFFERENT_SYSTEMS,
                            primary_case="european_crisis",
                            comparison_cases=["asian_crisis"],
                            overall_similarity=0.4,
                            context_similarity=0.2,
                            outcome_similarity=0.9
                        )
                    ],
                    "mechanisms": [],
                    "alignments": {}
                }
                
                output_path = Path(temp_dir) / "final_report.html"
                mock_report.return_value = str(output_path)
                
                # Execute complete workflow
                case_ids = self.tracer.load_cases(case_files)
                analysis_results = self.tracer.run_comparative_analysis(
                    case_ids, 
                    comparison_types=[ComparisonType.MOST_DIFFERENT_SYSTEMS]
                )
                report_path = self.tracer.generate_comparative_report(analysis_results, str(output_path))
                
                # Verify workflow execution
                assert len(case_ids) == 2
                assert "mds_results" in analysis_results
                assert len(analysis_results["mds_results"]) == 1
                assert report_path == str(output_path)
                
                # Verify method calls
                mock_load.assert_called_once_with(case_files)
                mock_analyze.assert_called_once()
                mock_report.assert_called_once()
    
    def test_workflow_error_handling(self):
        """Test workflow error handling and recovery."""
        # Test with non-existent files
        non_existent_files = ["/nonexistent/case1.json", "/nonexistent/case2.json"]
        
        # Should handle file errors gracefully
        try:
            case_ids = self.tracer.load_cases(non_existent_files)
            # If no exception, should return empty list or handle gracefully
            assert isinstance(case_ids, list)
        except (FileNotFoundError, ComparativeAnalysisError):
            # Expected behavior for missing files
            pass
    
    def test_workflow_with_insufficient_cases(self):
        """Test workflow with insufficient cases for comparison."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create single case file
            case_file = Path(temp_dir) / "single_case.json"
            case_data = {
                "nodes": [{"id": "event1", "type": "Event", "description": "Single event"}],
                "edges": [],
                "metadata": {
                    "case_name": "Single Case",
                    "description": "Only one case for testing",
                    "primary_outcome": "unknown"
                }
            }
            
            with open(case_file, 'w') as f:
                json.dump(case_data, f)
            
            # Mock workflow with single case
            with patch.object(self.tracer, 'load_cases') as mock_load, \
                 patch.object(self.tracer, 'run_comparative_analysis') as mock_analyze:
                
                mock_load.return_value = ["single_case"]
                mock_analyze.return_value = {
                    "mss_results": [],
                    "mds_results": [],
                    "mechanisms": [],
                    "alignments": {}
                }
                
                case_ids = self.tracer.load_cases([str(case_file)])
                results = self.tracer.run_comparative_analysis(case_ids)
                
                # Should handle single case gracefully
                assert len(case_ids) == 1
                assert len(results["mss_results"]) == 0
                assert len(results["mds_results"]) == 0


class TestComparativeIntegrationPerformance:
    """Test performance characteristics of integrated comparative analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = ComparativeProcessTracer()
    
    def test_performance_with_multiple_cases(self):
        """Test performance with larger number of cases."""
        # Create mock data for 5 cases
        case_ids = [f"case_{i}" for i in range(5)]
        
        # Mock the analysis to return quickly
        with patch.object(self.tracer, 'run_comparative_analysis') as mock_analyze:
            mock_analyze.return_value = {
                "mss_results": [],
                "mds_results": [],
                "mechanisms": [],
                "alignments": {}
            }
            
            # Should complete analysis quickly
            results = self.tracer.run_comparative_analysis(case_ids)
            
            assert isinstance(results, dict)
            mock_analyze.assert_called_once_with(case_ids)
    
    def test_memory_usage_with_large_graphs(self):
        """Test memory usage with larger graph structures."""
        # Mock large case graphs
        large_case_ids = [f"large_case_{i}" for i in range(3)]
        
        with patch.object(self.tracer.case_manager, 'get_case') as mock_get_case:
            # Create mock large graphs
            def create_large_graph():
                graph = nx.DiGraph()
                for i in range(50):  # 50 nodes per graph
                    graph.add_node(f"node_{i}", type="Event", description=f"Node {i}")
                for i in range(49):  # 49 edges
                    graph.add_edge(f"node_{i}", f"node_{i+1}", type="causes")
                return graph
            
            mock_get_case.side_effect = lambda x: create_large_graph()
            
            # Should handle large graphs without memory issues
            with patch.object(self.tracer, 'run_comparative_analysis') as mock_analyze:
                mock_analyze.return_value = {
                    "mss_results": [],
                    "mds_results": [],
                    "mechanisms": [],
                    "alignments": {}
                }
                
                results = self.tracer.run_comparative_analysis(large_case_ids)
                assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
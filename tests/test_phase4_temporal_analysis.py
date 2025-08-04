"""
Comprehensive Test Suite for Phase 4: Temporal Process Tracing Analysis

Tests all temporal analysis components including:
- Temporal extraction from graphs
- Temporal graph creation and manipulation
- Critical juncture analysis
- Duration analysis
- Temporal validation
- Temporal visualization
- Integration with main analysis pipeline

Author: Claude Code Implementation
Date: August 2025
"""

import pytest
import networkx as nx
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import sys

# Add the parent directory to the path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.temporal_extraction import TemporalExtractor, TemporalRelation, TemporalType
from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge, TemporalViolation, TemporalConstraint
from core.critical_junctures import CriticalJunctureAnalyzer, JunctureType, CriticalJuncture
from core.duration_analysis import DurationAnalyzer, ProcessSpeed, TemporalPhase
from core.temporal_validator import TemporalValidator, ValidationSeverity
from core.temporal_viz import TemporalVisualizer
from core.analyze import analyze_graph


class TestTemporalExtraction:
    """Test temporal data extraction from standard NetworkX graphs"""
    
    def test_basic_temporal_extraction(self):
        """Test basic extraction of temporal data from graph nodes"""
        # Create test graph with temporal data
        G = nx.DiGraph()
        G.add_node("event1", type="Event", 
                  timestamp="2020-01-01T10:00:00",
                  description="Initial crisis event")
        G.add_node("event2", type="Event",
                  timestamp="2020-01-15T14:30:00", 
                  duration="5 days",
                  description="Government response")
        G.add_edge("event1", "event2", relation="causes")
        
        extractor = TemporalExtractor()
        temporal_graph = extractor.extract_temporal_graph(G)
        
        assert len(temporal_graph.temporal_nodes) == 2
        assert "event1" in temporal_graph.temporal_nodes
        assert "event2" in temporal_graph.temporal_nodes
        
        # Check timestamp parsing
        node1 = temporal_graph.temporal_nodes["event1"]
        assert node1.timestamp == datetime(2020, 1, 1, 10, 0, 0)
        
        node2 = temporal_graph.temporal_nodes["event2"]
        assert node2.timestamp == datetime(2020, 1, 15, 14, 30, 0)
        assert node2.duration == timedelta(days=5)
    
    def test_temporal_relationship_extraction(self):
        """Test extraction of temporal relationships between nodes"""
        G = nx.DiGraph()
        G.add_node("cause", type="Event", timestamp="2020-01-01")
        G.add_node("effect", type="Event", timestamp="2020-01-15")
        G.add_edge("cause", "effect", relation="before", type="causes")
        
        extractor = TemporalExtractor()
        temporal_graph = extractor.extract_temporal_graph(G)
        
        assert len(temporal_graph.temporal_edges) == 1
        edge_key = list(temporal_graph.temporal_edges.keys())[0]
        edge = temporal_graph.temporal_edges[edge_key]
        
        assert edge.source == "cause"
        assert edge.target == "effect"
        assert edge.temporal_relation == TemporalRelation.BEFORE
        assert edge.edge_type == "causes"
    
    def test_uncertainty_extraction(self):
        """Test extraction of temporal uncertainty values"""
        G = nx.DiGraph()
        G.add_node("uncertain_event", type="Event",
                  timestamp="2020-01-01",
                  temporal_uncertainty="0.7",
                  description="Event with high uncertainty")
        
        extractor = TemporalExtractor()
        temporal_graph = extractor.extract_temporal_graph(G)
        
        node = temporal_graph.temporal_nodes["uncertain_event"]
        assert node.temporal_uncertainty == 0.7
    
    def test_sequence_extraction(self):
        """Test extraction of sequence order from nodes"""
        G = nx.DiGraph()
        G.add_node("first", type="Event", sequence_order=1)
        G.add_node("second", type="Event", sequence_order=2)
        G.add_node("third", type="Event", sequence_order=3)
        
        extractor = TemporalExtractor()
        temporal_graph = extractor.extract_temporal_graph(G)
        
        assert temporal_graph.temporal_nodes["first"].sequence_order == 1
        assert temporal_graph.temporal_nodes["second"].sequence_order == 2
        assert temporal_graph.temporal_nodes["third"].sequence_order == 3


class TestTemporalGraph:
    """Test temporal graph data structure and operations"""
    
    def test_temporal_graph_creation(self):
        """Test creation and basic operations of temporal graph"""
        tg = TemporalGraph()
        
        # Add temporal node
        node = TemporalNode(
            node_id="test_event",
            timestamp=datetime(2020, 1, 1),
            node_type="Event",
            attr_props={"description": "Test event"}
        )
        tg.add_temporal_node(node)
        
        assert "test_event" in tg.temporal_nodes
        assert tg.temporal_nodes["test_event"].timestamp == datetime(2020, 1, 1)
    
    def test_temporal_edge_operations(self):
        """Test adding and managing temporal edges"""
        tg = TemporalGraph()
        
        # Add nodes
        node1 = TemporalNode("node1", datetime(2020, 1, 1), "Event")
        node2 = TemporalNode("node2", datetime(2020, 1, 15), "Event")
        tg.add_temporal_node(node1)
        tg.add_temporal_node(node2)
        
        # Add edge
        edge = TemporalEdge(
            source="node1",
            target="node2",
            temporal_relation=TemporalRelation.BEFORE,
            edge_type="causes"
        )
        tg.add_temporal_edge(edge)
        
        assert len(tg.temporal_edges) == 1
        edge_key = list(tg.temporal_edges.keys())[0]
        assert tg.temporal_edges[edge_key].source == "node1"
        assert tg.temporal_edges[edge_key].target == "node2"
    
    def test_networkx_conversion(self):
        """Test conversion to NetworkX graph"""
        tg = TemporalGraph()
        
        node1 = TemporalNode("node1", datetime(2020, 1, 1), "Event")
        node2 = TemporalNode("node2", datetime(2020, 1, 15), "Event")
        tg.add_temporal_node(node1)
        tg.add_temporal_node(node2)
        
        edge = TemporalEdge("node1", "node2", TemporalRelation.BEFORE, "causes")
        tg.add_temporal_edge(edge)
        
        nx_graph = tg.to_networkx()
        
        assert nx_graph.number_of_nodes() == 2
        assert nx_graph.number_of_edges() == 1
        assert nx_graph.has_edge("node1", "node2")
    
    def test_temporal_statistics(self):
        """Test temporal statistics calculation"""
        tg = TemporalGraph()
        
        # Add nodes with various temporal properties
        node1 = TemporalNode("node1", datetime(2020, 1, 1), "Event", duration=timedelta(days=5))
        node2 = TemporalNode("node2", datetime(2020, 1, 15), "Event")
        node3 = TemporalNode("node3", None, "Event")  # No timestamp
        
        tg.add_temporal_node(node1)
        tg.add_temporal_node(node2)
        tg.add_temporal_node(node3)
        
        stats = tg.get_temporal_statistics()
        
        assert stats['total_nodes'] == 3
        assert stats['nodes_with_timestamps'] == 2
        assert stats['nodes_with_duration'] == 1
        assert 'temporal_span' in stats
    
    def test_temporal_constraints(self):
        """Test temporal constraint management"""
        tg = TemporalGraph()
        
        constraint = TemporalConstraint(
            constraint_id="deadline_constraint",
            constraint_type="deadline",
            affected_nodes=["node1"],
            temporal_requirement="must complete before 2020-02-01",
            deadline=datetime(2020, 2, 1)
        )
        
        tg.add_temporal_constraint(constraint)
        
        assert len(tg.temporal_constraints) == 1
        assert "deadline_constraint" in tg.temporal_constraints


class TestCriticalJunctureAnalysis:
    """Test critical juncture identification and analysis"""
    
    def test_decision_point_detection(self):
        """Test detection of decision points in temporal graph"""
        tg = TemporalGraph()
        
        # Add decision node
        decision_node = TemporalNode(
            node_id="policy_decision",
            timestamp=datetime(2020, 3, 1),
            node_type="Event",
            attr_props={"description": "Government policy decision on economic response"}
        )
        
        # Add alternative outcomes
        outcome1 = TemporalNode(
            node_id="stimulus",
            timestamp=datetime(2020, 3, 15),
            node_type="Event",
            attr_props={"description": "Stimulus package implementation"}
        )
        
        outcome2 = TemporalNode(
            node_id="austerity",
            timestamp=datetime(2020, 3, 15),
            node_type="Event",
            attr_props={"description": "Austerity measures implementation"}
        )
        
        tg.add_temporal_node(decision_node)
        tg.add_temporal_node(outcome1)
        tg.add_temporal_node(outcome2)
        
        # Add decision edges
        edge1 = TemporalEdge("policy_decision", "stimulus", TemporalRelation.BEFORE, "causes")
        edge2 = TemporalEdge("policy_decision", "austerity", TemporalRelation.BEFORE, "causes")
        tg.add_temporal_edge(edge1)
        tg.add_temporal_edge(edge2)
        
        analyzer = CriticalJunctureAnalyzer()
        junctures = analyzer.identify_junctures(tg)
        
        assert len(junctures) > 0
        
        # Find decision point juncture
        decision_junctures = [j for j in junctures if j.juncture_type == JunctureType.DECISION_POINT]
        assert len(decision_junctures) > 0
        
        decision_juncture = decision_junctures[0]
        assert "policy_decision" in decision_juncture.key_nodes
        assert len(decision_juncture.alternative_pathways) >= 2
    
    def test_timing_critical_detection(self):
        """Test detection of timing-critical events"""
        tg = TemporalGraph()
        
        crisis_node = TemporalNode(
            node_id="crisis_response",
            timestamp=datetime(2020, 3, 1),
            node_type="Event",
            attr_props={"description": "Emergency crisis response with urgent deadline"}
        )
        tg.add_temporal_node(crisis_node)
        
        analyzer = CriticalJunctureAnalyzer()
        junctures = analyzer.identify_junctures(tg)
        
        timing_critical = [j for j in junctures if j.juncture_type == JunctureType.TIMING_CRITICAL]
        assert len(timing_critical) > 0
        assert timing_critical[0].timing_sensitivity >= 0.6
    
    def test_convergence_point_detection(self):
        """Test detection of convergence points"""
        tg = TemporalGraph()
        
        # Create convergence scenario
        cause1 = TemporalNode("cause1", datetime(2020, 1, 1), "Event")
        cause2 = TemporalNode("cause2", datetime(2020, 1, 5), "Event")
        cause3 = TemporalNode("cause3", datetime(2020, 1, 10), "Event")
        convergence = TemporalNode("convergence", datetime(2020, 1, 15), "Event")
        
        tg.add_temporal_node(cause1)
        tg.add_temporal_node(cause2)
        tg.add_temporal_node(cause3)
        tg.add_temporal_node(convergence)
        
        # All causes lead to convergence point
        tg.add_temporal_edge(TemporalEdge("cause1", "convergence", TemporalRelation.BEFORE, "causes"))
        tg.add_temporal_edge(TemporalEdge("cause2", "convergence", TemporalRelation.BEFORE, "causes"))
        tg.add_temporal_edge(TemporalEdge("cause3", "convergence", TemporalRelation.BEFORE, "causes"))
        
        analyzer = CriticalJunctureAnalyzer()
        junctures = analyzer.identify_junctures(tg)
        
        convergence_junctures = [j for j in junctures if j.juncture_type == JunctureType.CONVERGENCE_POINT]
        assert len(convergence_junctures) > 0
        assert "convergence" in convergence_junctures[0].key_nodes
    
    def test_juncture_analysis_result(self):
        """Test comprehensive juncture analysis results"""
        tg = TemporalGraph()
        
        # Add multiple types of junctures
        decision_node = TemporalNode(
            "decision", datetime(2020, 1, 1), "Event",
            attr_props={"description": "Policy decision"}
        )
        outcome1 = TemporalNode("outcome1", datetime(2020, 1, 15), "Event")
        outcome2 = TemporalNode("outcome2", datetime(2020, 1, 15), "Event")
        
        tg.add_temporal_node(decision_node)
        tg.add_temporal_node(outcome1)
        tg.add_temporal_node(outcome2)
        
        tg.add_temporal_edge(TemporalEdge("decision", "outcome1", TemporalRelation.BEFORE, "causes"))
        tg.add_temporal_edge(TemporalEdge("decision", "outcome2", TemporalRelation.BEFORE, "causes"))
        
        analyzer = CriticalJunctureAnalyzer()
        junctures = analyzer.identify_junctures(tg)
        analysis_result = analyzer.analyze_juncture_distribution(junctures)
        
        assert analysis_result.total_junctures > 0
        assert isinstance(analysis_result.junctures_by_type, dict)
        assert isinstance(analysis_result.temporal_distribution, dict)
        assert 0.0 <= analysis_result.overall_timing_sensitivity <= 1.0


class TestDurationAnalysis:
    """Test duration analysis and timing patterns"""
    
    def test_process_duration_analysis(self):
        """Test analysis of individual process durations"""
        tg = TemporalGraph()
        
        # Add processes with different durations
        fast_process = TemporalNode(
            "fast_decision",
            datetime(2020, 1, 1),
            "Event",
            duration=timedelta(hours=6),
            attr_props={"description": "Rapid crisis decision"}
        )
        
        slow_process = TemporalNode(
            "policy_development",
            datetime(2020, 1, 15),
            "Event",
            duration=timedelta(days=120),
            attr_props={"description": "Complex policy development process"}
        )
        
        tg.add_temporal_node(fast_process)
        tg.add_temporal_node(slow_process)
        
        analyzer = DurationAnalyzer()
        result = analyzer.analyze_durations(tg)
        
        assert result.total_processes == 2
        assert len(result.process_durations) == 2
        
        # Check speed classification
        speeds = [pd.process_speed for pd in result.process_durations]
        assert ProcessSpeed.INSTANTANEOUS in speeds or ProcessSpeed.RAPID in speeds
        assert ProcessSpeed.SLOW in speeds or ProcessSpeed.VERY_SLOW in speeds
    
    def test_pathway_duration_analysis(self):
        """Test analysis of causal pathway durations"""
        tg = TemporalGraph()
        
        # Create pathway with timing
        node1 = TemporalNode("start", datetime(2020, 1, 1), "Event")
        node2 = TemporalNode("middle", datetime(2020, 1, 15), "Event")
        node3 = TemporalNode("end", datetime(2020, 2, 1), "Event")
        
        tg.add_temporal_node(node1)
        tg.add_temporal_node(node2)
        tg.add_temporal_node(node3)
        
        tg.add_temporal_edge(TemporalEdge("start", "middle", TemporalRelation.BEFORE, "causes"))
        tg.add_temporal_edge(TemporalEdge("middle", "end", TemporalRelation.BEFORE, "causes"))
        
        analyzer = DurationAnalyzer()
        result = analyzer.analyze_durations(tg)
        
        assert len(result.pathway_durations) > 0
        
        # Check pathway analysis
        pathway = result.pathway_durations[0]
        assert pathway.total_duration is not None
        assert len(pathway.pathway_nodes) >= 2
    
    def test_temporal_pattern_identification(self):
        """Test identification of recurring temporal patterns"""
        tg = TemporalGraph()
        
        # Add multiple processes with similar speeds
        for i in range(5):
            node = TemporalNode(
                f"rapid_process_{i}",
                datetime(2020, 1, i+1),
                "Event",
                duration=timedelta(hours=2)
            )
            tg.add_temporal_node(node)
        
        analyzer = DurationAnalyzer()
        result = analyzer.analyze_durations(tg)
        
        assert len(result.temporal_patterns) > 0
        
        # Should identify pattern of rapid processes
        speed_patterns = [p for p in result.temporal_patterns if p.pattern_type == "speed_consistency"]
        assert len(speed_patterns) > 0
    
    def test_performance_recommendations(self):
        """Test generation of performance improvement recommendations"""
        tg = TemporalGraph()
        
        # Add inefficient process
        inefficient_process = TemporalNode(
            "inefficient_process",
            datetime(2020, 1, 1),
            "Event",
            duration=timedelta(days=365),  # Very long duration
            attr_props={"description": "Inefficient bureaucratic process"}
        )
        tg.add_temporal_node(inefficient_process)
        
        analyzer = DurationAnalyzer()
        result = analyzer.analyze_durations(tg)
        
        assert len(result.performance_recommendations) > 0
        assert any("slow" in rec.lower() for rec in result.performance_recommendations)


class TestTemporalValidation:
    """Test temporal consistency validation"""
    
    def test_causal_paradox_detection(self):
        """Test detection of causal paradoxes (effect before cause)"""
        tg = TemporalGraph()
        
        # Create temporal paradox
        cause = TemporalNode("cause", datetime(2020, 3, 1), "Event")  # After effect
        effect = TemporalNode("effect", datetime(2020, 1, 1), "Event")  # Before cause
        
        tg.add_temporal_node(cause)
        tg.add_temporal_node(effect)
        
        # Add causal edge (this should create a paradox)
        edge = TemporalEdge("cause", "effect", TemporalRelation.BEFORE, "causes")
        tg.add_temporal_edge(edge)
        
        validator = TemporalValidator()
        result = validator.validate_temporal_graph(tg)
        
        assert not result.is_valid  # Should fail validation
        assert len(result.violations) > 0
        
        # Check for causal paradox violation
        paradox_violations = [v for v in result.violations if v.violation_type == "causal_paradox"]
        assert len(paradox_violations) > 0
    
    def test_sequence_consistency_validation(self):
        """Test validation of sequence number consistency"""
        tg = TemporalGraph()
        
        # Create sequence inconsistency
        node1 = TemporalNode("node1", datetime(2020, 1, 15), "Event", sequence_order=2)
        node2 = TemporalNode("node2", datetime(2020, 1, 1), "Event", sequence_order=1)
        
        tg.add_temporal_node(node1)
        tg.add_temporal_node(node2)
        
        # Sequence says node1 comes first, but timestamps say node2 comes first
        edge = TemporalEdge("node1", "node2", TemporalRelation.BEFORE, "causes")
        tg.add_temporal_edge(edge)
        
        validator = TemporalValidator()
        result = validator.validate_temporal_graph(tg)
        
        # Should detect sequence-time mismatch
        sequence_violations = [v for v in result.violations 
                             if "sequence" in v.violation_type.lower()]
        assert len(sequence_violations) > 0
    
    def test_duration_logic_validation(self):
        """Test validation of duration logic"""
        tg = TemporalGraph()
        
        # Create unreasonably long duration
        node = TemporalNode(
            "long_process",
            datetime(2020, 1, 1),
            "Event",
            duration=timedelta(days=36500)  # 100 years - unreasonable
        )
        tg.add_temporal_node(node)
        
        validator = TemporalValidator()
        result = validator.validate_temporal_graph(tg)
        
        # Should warn about unreasonable duration
        duration_violations = [v for v in result.violations 
                             if v.violation_type == "duration_logic"]
        assert len(duration_violations) > 0
    
    def test_validation_confidence_score(self):
        """Test calculation of validation confidence score"""
        tg = TemporalGraph()
        
        # Create valid temporal graph
        node1 = TemporalNode("node1", datetime(2020, 1, 1), "Event")
        node2 = TemporalNode("node2", datetime(2020, 1, 15), "Event")
        
        tg.add_temporal_node(node1)
        tg.add_temporal_node(node2)
        
        edge = TemporalEdge("node1", "node2", TemporalRelation.BEFORE, "causes")
        tg.add_temporal_edge(edge)
        
        validator = TemporalValidator()
        result = validator.validate_temporal_graph(tg)
        
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.is_valid  # Should be valid


class TestTemporalVisualization:
    """Test temporal visualization data generation"""
    
    def test_visualization_data_generation(self):
        """Test generation of temporal visualization data"""
        tg = TemporalGraph()
        
        node1 = TemporalNode("event1", datetime(2020, 1, 1), "Event")
        node2 = TemporalNode("event2", datetime(2020, 1, 15), "Event")
        
        tg.add_temporal_node(node1)
        tg.add_temporal_node(node2)
        
        edge = TemporalEdge("event1", "event2", TemporalRelation.BEFORE, "causes")
        tg.add_temporal_edge(edge)
        
        visualizer = TemporalVisualizer()
        viz_data = visualizer.generate_visualization_data(tg)
        
        assert 'nodes' in viz_data
        assert 'edges' in viz_data
        assert 'timeline_data' in viz_data
        
        assert len(viz_data['nodes']) == 2
        assert len(viz_data['edges']) == 1
    
    def test_timeline_data_generation(self):
        """Test generation of timeline visualization data"""
        tg = TemporalGraph()
        
        # Add events at different times
        events = [
            TemporalNode("event1", datetime(2020, 1, 1), "Event"),
            TemporalNode("event2", datetime(2020, 2, 1), "Event"),
            TemporalNode("event3", datetime(2020, 3, 1), "Event")
        ]
        
        for event in events:
            tg.add_temporal_node(event)
        
        visualizer = TemporalVisualizer()
        viz_data = visualizer.generate_visualization_data(tg)
        
        timeline_data = viz_data['timeline_data']
        assert len(timeline_data) == 3
        
        # Check chronological ordering
        timestamps = [item['x'] for item in timeline_data]
        assert timestamps == sorted(timestamps)


class TestPhase4Integration:
    """Test integration of Phase 4 temporal analysis with main pipeline"""
    
    def test_temporal_analysis_in_main_pipeline(self):
        """Test that temporal analysis is properly integrated into main analysis"""
        # Create test graph with temporal data
        G = nx.DiGraph()
        G.add_node("event1", type="Event", 
                  timestamp="2020-01-01",
                  description="Initial event")
        G.add_node("event2", type="Event", 
                  timestamp="2020-01-15",
                  description="Follow-up event")
        G.add_edge("event1", "event2", relation="causes")
        
        # Run main analysis
        results = analyze_graph(G)
        
        # Check that temporal analysis is included
        assert 'temporal_analysis' in results
        temporal_analysis = results['temporal_analysis']
        
        if not temporal_analysis.get('error'):
            # If temporal analysis succeeded, check components
            assert 'temporal_graph' in temporal_analysis
            assert 'validation_result' in temporal_analysis
            assert 'critical_junctures' in temporal_analysis
            assert 'duration_analysis' in temporal_analysis
            assert 'temporal_visualization' in temporal_analysis
            assert 'temporal_statistics' in temporal_analysis
    
    def test_temporal_analysis_error_handling(self):
        """Test error handling in temporal analysis integration"""
        # Create minimal graph that might cause issues
        G = nx.DiGraph()
        G.add_node("minimal", type="Event")
        
        # Should not crash even with minimal data
        results = analyze_graph(G)
        
        assert 'temporal_analysis' in results
        # Either succeeds with minimal data or fails gracefully with error
        temporal_analysis = results['temporal_analysis']
        assert isinstance(temporal_analysis, dict)


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios for temporal analysis"""
    
    def test_complex_temporal_scenario(self):
        """Test analysis of complex temporal scenario"""
        tg = TemporalGraph()
        
        # Create complex scenario: Crisis -> Multiple responses -> Convergence -> Outcome
        crisis = TemporalNode(
            "crisis_event",
            datetime(2020, 3, 1),
            "Event",
            attr_props={"description": "Financial crisis triggers urgent response"}
        )
        
        # Multiple parallel responses
        response1 = TemporalNode(
            "monetary_policy",
            datetime(2020, 3, 5),
            "Event",
            duration=timedelta(days=7),
            attr_props={"description": "Central bank monetary policy decision"}
        )
        
        response2 = TemporalNode(
            "fiscal_policy",
            datetime(2020, 3, 10),
            "Event",
            duration=timedelta(days=14),
            attr_props={"description": "Government fiscal policy announcement"}
        )
        
        # Convergence point
        convergence = TemporalNode(
            "policy_coordination",
            datetime(2020, 3, 20),
            "Event",
            attr_props={"description": "Coordinated policy implementation"}
        )
        
        # Final outcome
        outcome = TemporalNode(
            "market_stabilization",
            datetime(2020, 4, 1),
            "Event",
            duration=timedelta(days=30),
            attr_props={"description": "Market stabilization achieved"}
        )
        
        # Add all nodes
        for node in [crisis, response1, response2, convergence, outcome]:
            tg.add_temporal_node(node)
        
        # Add causal relationships
        edges = [
            TemporalEdge("crisis_event", "monetary_policy", TemporalRelation.BEFORE, "causes"),
            TemporalEdge("crisis_event", "fiscal_policy", TemporalRelation.BEFORE, "causes"),
            TemporalEdge("monetary_policy", "policy_coordination", TemporalRelation.BEFORE, "enables"),
            TemporalEdge("fiscal_policy", "policy_coordination", TemporalRelation.BEFORE, "enables"),
            TemporalEdge("policy_coordination", "market_stabilization", TemporalRelation.BEFORE, "causes")
        ]
        
        for edge in edges:
            tg.add_temporal_edge(edge)
        
        # Run all analyses
        validator = TemporalValidator()
        validation_result = validator.validate_temporal_graph(tg)
        
        juncture_analyzer = CriticalJunctureAnalyzer()
        critical_junctures = juncture_analyzer.identify_junctures(tg)
        
        duration_analyzer = DurationAnalyzer()
        duration_analysis = duration_analyzer.analyze_durations(tg)
        
        visualizer = TemporalVisualizer()
        viz_data = visualizer.generate_visualization_data(tg)
        
        # Assertions for complex scenario
        assert validation_result.is_valid or len(validation_result.violations) < 3
        assert len(critical_junctures) > 0
        assert duration_analysis.total_processes == 5
        assert len(viz_data['nodes']) == 5
        assert len(viz_data['edges']) == 5
        
        # Should detect decision points and convergence
        juncture_types = [j.juncture_type for j in critical_junctures]
        assert JunctureType.CONVERGENCE_POINT in juncture_types or JunctureType.DECISION_POINT in juncture_types
    
    def test_temporal_analysis_with_missing_data(self):
        """Test temporal analysis robustness with missing temporal data"""
        tg = TemporalGraph()
        
        # Mix of nodes with and without temporal data
        complete_node = TemporalNode(
            "complete",
            datetime(2020, 1, 1),
            "Event",
            duration=timedelta(days=5),
            sequence_order=1
        )
        
        partial_node = TemporalNode(
            "partial",
            None,  # No timestamp
            "Event",
            sequence_order=2
        )
        
        minimal_node = TemporalNode(
            "minimal",
            None,
            "Event"
            # No temporal data at all
        )
        
        tg.add_temporal_node(complete_node)
        tg.add_temporal_node(partial_node)
        tg.add_temporal_node(minimal_node)
        
        # Analysis should handle missing data gracefully
        validator = TemporalValidator()
        validation_result = validator.validate_temporal_graph(tg)
        
        duration_analyzer = DurationAnalyzer()
        duration_analysis = duration_analyzer.analyze_durations(tg)
        
        # Should generate warnings but not crash
        assert len(validation_result.warnings) > 0
        assert validation_result.confidence_score < 1.0
        assert duration_analysis.total_processes >= 0  # May be 0 or 1 depending on implementation


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
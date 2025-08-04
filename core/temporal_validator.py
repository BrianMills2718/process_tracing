"""
Temporal Process Tracing - Temporal Sequence Validation Module

Validates temporal sequences in causal chains to ensure logical consistency
and flag temporal violations in process tracing analysis.

Author: Claude Code Implementation  
Date: August 2025
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge, TemporalViolation
from core.temporal_extraction import TemporalRelation

class ValidationSeverity(Enum):
    CRITICAL = "critical"    # Fatal logical errors
    HIGH = "high"           # Serious inconsistencies  
    MEDIUM = "medium"       # Moderate issues
    LOW = "low"             # Minor warnings
    INFO = "info"           # Informational notices

@dataclass
class ValidationResult:
    """Result of temporal validation"""
    is_valid: bool
    violations: List[TemporalViolation]
    warnings: List[str]
    suggestions: List[str]
    confidence_score: float  # 0.0 = no confidence, 1.0 = high confidence
    validation_report: str

@dataclass
class TemporalPath:
    """Represents a temporal path through the graph"""
    path_id: str
    nodes: List[str]
    total_duration: Optional[timedelta]
    violations: List[TemporalViolation]
    confidence: float
    is_valid: bool

class TemporalValidator:
    """
    Validates temporal consistency in process tracing graphs and provides
    detailed analysis of temporal logic violations.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode  # Whether to treat warnings as errors
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules and thresholds"""
        return {
            'max_reasonable_gap': timedelta(days=365 * 20),  # 20 years
            'min_process_duration': timedelta(minutes=1),    # 1 minute
            'max_process_duration': timedelta(days=365 * 50), # 50 years
            'concurrent_tolerance': timedelta(hours=24),      # 24 hours for "concurrent"
            'sequence_gap_tolerance': 0.2,                   # 20% tolerance for sequence gaps
            'uncertainty_threshold': 0.8,                    # Flag high uncertainty
        }
    
    def validate_temporal_graph(self, temporal_graph: TemporalGraph) -> ValidationResult:
        """
        Comprehensive validation of temporal graph consistency.
        """
        violations = []
        warnings = []
        suggestions = []
        
        # Run all validation checks
        violations.extend(self._validate_causal_ordering(temporal_graph))
        violations.extend(self._validate_temporal_paths(temporal_graph))
        violations.extend(self._validate_sequence_consistency(temporal_graph))
        violations.extend(self._validate_duration_logic(temporal_graph))
        violations.extend(self._validate_temporal_relationships(temporal_graph))
        
        # Generate warnings for high uncertainty
        warnings.extend(self._check_uncertainty_levels(temporal_graph))
        
        # Generate suggestions for improvement
        suggestions.extend(self._generate_improvement_suggestions(temporal_graph, violations))
        
        # Calculate overall confidence
        confidence_score = self._calculate_validation_confidence(temporal_graph, violations)
        
        # Determine if graph is valid
        critical_violations = [v for v in violations if v.severity >= 0.8]
        is_valid = len(critical_violations) == 0 and (not self.strict_mode or len(violations) == 0)
        
        # Generate validation report
        report = self._generate_validation_report(temporal_graph, violations, warnings, suggestions, confidence_score)
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            suggestions=suggestions,
            confidence_score=confidence_score,
            validation_report=report
        )
    
    def _validate_causal_ordering(self, temporal_graph: TemporalGraph) -> List[TemporalViolation]:
        """Validate that causes precede effects in time"""
        violations = []
        
        for edge_key, edge in temporal_graph.temporal_edges.items():
            if edge.edge_type != "causes":
                continue
                
            source_node = temporal_graph.temporal_nodes.get(edge.source)
            target_node = temporal_graph.temporal_nodes.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # Check timestamp ordering
            if source_node.timestamp and target_node.timestamp:
                time_gap = target_node.timestamp - source_node.timestamp
                
                if time_gap < timedelta(0):
                    # Effect precedes cause - critical violation
                    violations.append(TemporalViolation(
                        violation_id=f"causal_paradox_{edge.source}_{edge.target}",
                        violation_type="causal_paradox",
                        nodes_involved=[edge.source, edge.target],
                        description=f"Causal paradox: Effect '{edge.target}' ({target_node.timestamp}) occurs before cause '{edge.source}' ({source_node.timestamp})",
                        severity=1.0,  # Critical
                        suggested_fix="Verify timestamps or reverse causal relationship"
                    ))
                elif time_gap == timedelta(0):
                    # Simultaneous cause and effect - possible issue
                    violations.append(TemporalViolation(
                        violation_id=f"simultaneous_causation_{edge.source}_{edge.target}",
                        violation_type="simultaneous_causation",
                        nodes_involved=[edge.source, edge.target],
                        description=f"Simultaneous causation: '{edge.source}' and '{edge.target}' occur at same time",
                        severity=0.6,
                        suggested_fix="Consider if instantaneous causation is plausible or adjust timing"
                    ))
            
            # Check sequence ordering
            if (source_node.sequence_order is not None and 
                target_node.sequence_order is not None):
                if source_node.sequence_order >= target_node.sequence_order:
                    violations.append(TemporalViolation(
                        violation_id=f"sequence_paradox_{edge.source}_{edge.target}",
                        violation_type="sequence_paradox",
                        nodes_involved=[edge.source, edge.target],
                        description=f"Sequence paradox: Cause '{edge.source}' (seq {source_node.sequence_order}) comes after effect '{edge.target}' (seq {target_node.sequence_order})",
                        severity=0.8,
                        suggested_fix="Verify sequence numbers or causal relationship"
                    ))
        
        return violations
    
    def _validate_temporal_paths(self, temporal_graph: TemporalGraph) -> List[TemporalViolation]:
        """Validate temporal consistency along causal paths"""
        violations = []
        
        # Find all simple paths in the graph
        graph = temporal_graph.to_networkx()
        
        # Get all simple paths up to reasonable length
        all_paths = []
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(graph, source, target, cutoff=10))
                        all_paths.extend(paths)
                    except nx.NetworkXNoPath:
                        continue
        
        # Validate each path
        for path in all_paths:
            path_violations = self._validate_single_path(temporal_graph, path)
            violations.extend(path_violations)
        
        return violations
    
    def _validate_single_path(self, temporal_graph: TemporalGraph, path: List[str]) -> List[TemporalViolation]:
        """Validate temporal consistency of a single path"""
        violations = []
        
        if len(path) < 2:
            return violations
        
        path_id = "->".join(path)
        
        # Check temporal ordering along the path
        timestamps = []
        for node_id in path:
            node = temporal_graph.temporal_nodes.get(node_id)
            if node and node.timestamp:
                timestamps.append((node.timestamp, node_id))
        
        # Verify timestamps are in order
        if len(timestamps) > 1:
            for i in range(len(timestamps) - 1):
                current_time, current_node = timestamps[i]
                next_time, next_node = timestamps[i + 1]
                
                if current_time > next_time:
                    violations.append(TemporalViolation(
                        violation_id=f"path_ordering_{path_id}_{i}",
                        violation_type="path_ordering",
                        nodes_involved=[current_node, next_node],
                        description=f"Path ordering violation: '{current_node}' ({current_time}) comes after '{next_node}' ({next_time}) in causal path",
                        severity=0.9,
                        suggested_fix="Verify timestamps along causal path"
                    ))
                
                # Check for unreasonable gaps
                gap = next_time - current_time
                if gap > self.validation_rules['max_reasonable_gap']:
                    violations.append(TemporalViolation(
                        violation_id=f"path_gap_{path_id}_{i}",
                        violation_type="unreasonable_path_gap",
                        nodes_involved=[current_node, next_node],
                        description=f"Unreasonably long gap in causal path: {gap} between '{current_node}' and '{next_node}'",
                        severity=0.5,
                        suggested_fix="Consider intermediate events or verify timestamps"
                    ))
        
        return violations
    
    def _validate_sequence_consistency(self, temporal_graph: TemporalGraph) -> List[TemporalViolation]:
        """Validate consistency of sequence numbering"""
        violations = []
        
        # Get all nodes with sequence numbers
        sequenced_nodes = [
            (node.sequence_order, node_id, node)
            for node_id, node in temporal_graph.temporal_nodes.items()
            if node.sequence_order is not None
        ]
        
        if len(sequenced_nodes) < 2:
            return violations
        
        sequenced_nodes.sort(key=lambda x: x[0])
        
        # Check for gaps in sequence
        for i in range(len(sequenced_nodes) - 1):
            current_seq, current_id, current_node = sequenced_nodes[i]
            next_seq, next_id, next_node = sequenced_nodes[i + 1]
            
            gap = next_seq - current_seq
            
            # Check for large gaps in sequence numbers
            if gap > 5:  # Configurable threshold
                violations.append(TemporalViolation(
                    violation_id=f"sequence_gap_{current_id}_{next_id}",
                    violation_type="sequence_gap",
                    nodes_involved=[current_id, next_id],
                    description=f"Large gap in sequence: {current_seq} to {next_seq} (gap: {gap})",
                    severity=0.4,
                    suggested_fix="Consider missing intermediate events or renumber sequence"
                ))
            
            # If both nodes have timestamps, check consistency
            if current_node.timestamp and next_node.timestamp:
                time_gap = next_node.timestamp - current_node.timestamp
                
                # Sequence should generally follow temporal order
                if time_gap < timedelta(0):
                    violations.append(TemporalViolation(
                        violation_id=f"sequence_time_mismatch_{current_id}_{next_id}",
                        violation_type="sequence_time_mismatch",
                        nodes_involved=[current_id, next_id],
                        description=f"Sequence order conflicts with timestamps: {current_id} (seq {current_seq}, time {current_node.timestamp}) vs {next_id} (seq {next_seq}, time {next_node.timestamp})",
                        severity=0.7,
                        suggested_fix="Align sequence numbers with temporal order"
                    ))
        
        return violations
    
    def _validate_duration_logic(self, temporal_graph: TemporalGraph) -> List[TemporalViolation]:
        """Validate logical consistency of process durations"""
        violations = []
        
        for node_id, node in temporal_graph.temporal_nodes.items():
            if not node.duration:
                continue
            
            # Check for unreasonable durations
            if node.duration < self.validation_rules['min_process_duration']:
                violations.append(TemporalViolation(
                    violation_id=f"duration_too_short_{node_id}",
                    violation_type="duration_logic",
                    nodes_involved=[node_id],
                    description=f"Unreasonably short duration: {node.duration} for '{node_id}'",
                    severity=0.4,
                    suggested_fix="Verify duration measurement or consider instantaneous event"
                ))
            
            if node.duration > self.validation_rules['max_process_duration']:
                violations.append(TemporalViolation(
                    violation_id=f"duration_too_long_{node_id}",
                    violation_type="duration_logic",
                    nodes_involved=[node_id],
                    description=f"Unreasonably long duration: {node.duration} for '{node_id}'",
                    severity=0.5,
                    suggested_fix="Verify duration measurement or break into sub-processes"
                ))
            
            # Check duration vs timestamp consistency
            if node.timestamp and node.duration:
                # For processes with both start time and duration, check reasonableness
                end_time = node.timestamp + node.duration
                now = datetime.now()
                
                if end_time > now + timedelta(days=365):  # Process extends far into future
                    violations.append(TemporalViolation(
                        violation_id=f"future_process_{node_id}",
                        violation_type="duration_logic",
                        nodes_involved=[node_id],
                        description=f"Process '{node_id}' extends far into future: ends {end_time}",
                        severity=0.3,
                        suggested_fix="Verify if process is ongoing or adjust duration"
                    ))
        
        return violations
    
    def _validate_temporal_relationships(self, temporal_graph: TemporalGraph) -> List[TemporalViolation]:
        """Validate temporal relationships between events"""
        violations = []
        
        for edge_key, edge in temporal_graph.temporal_edges.items():
            source_node = temporal_graph.temporal_nodes.get(edge.source)
            target_node = temporal_graph.temporal_nodes.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # Validate relationship consistency with timestamps
            if source_node.timestamp and target_node.timestamp:
                time_diff = target_node.timestamp - source_node.timestamp
                
                # Check if declared relationship matches actual timing
                if edge.temporal_relation == TemporalRelation.BEFORE:
                    if time_diff <= timedelta(0):
                        violations.append(TemporalViolation(
                            violation_id=f"before_violation_{edge.source}_{edge.target}",
                            violation_type="relationship_mismatch",
                            nodes_involved=[edge.source, edge.target],
                            description=f"Relationship 'before' conflicts with timestamps: {edge.source} ({source_node.timestamp}) vs {edge.target} ({target_node.timestamp})",
                            severity=0.8,
                            suggested_fix="Correct relationship type or verify timestamps"
                        ))
                
                elif edge.temporal_relation == TemporalRelation.AFTER:
                    if time_diff >= timedelta(0):
                        violations.append(TemporalViolation(
                            violation_id=f"after_violation_{edge.source}_{edge.target}",
                            violation_type="relationship_mismatch",
                            nodes_involved=[edge.source, edge.target],
                            description=f"Relationship 'after' conflicts with timestamps",
                            severity=0.8,
                            suggested_fix="Correct relationship type or verify timestamps"
                        ))
                
                elif edge.temporal_relation == TemporalRelation.CONCURRENT:
                    if abs(time_diff) > self.validation_rules['concurrent_tolerance']:
                        violations.append(TemporalViolation(
                            violation_id=f"concurrent_violation_{edge.source}_{edge.target}",
                            violation_type="relationship_mismatch",
                            nodes_involved=[edge.source, edge.target],
                            description=f"Relationship 'concurrent' conflicts with timestamps: {abs(time_diff)} gap exceeds tolerance",
                            severity=0.6,
                            suggested_fix="Adjust relationship type or verify simultaneity"
                        ))
        
        return violations
    
    def _check_uncertainty_levels(self, temporal_graph: TemporalGraph) -> List[str]:
        """Check for high uncertainty levels and generate warnings"""
        warnings = []
        
        high_uncertainty_nodes = [
            node_id for node_id, node in temporal_graph.temporal_nodes.items()
            if node.temporal_uncertainty > self.validation_rules['uncertainty_threshold']
        ]
        
        if high_uncertainty_nodes:
            warnings.append(f"High temporal uncertainty in nodes: {', '.join(high_uncertainty_nodes)}")
        
        # Check for nodes with missing temporal data
        missing_temporal_data = [
            node_id for node_id, node in temporal_graph.temporal_nodes.items()
            if not node.timestamp and node.sequence_order is None
        ]
        
        if missing_temporal_data:
            warnings.append(f"Nodes missing temporal data: {', '.join(missing_temporal_data[:5])}{'...' if len(missing_temporal_data) > 5 else ''}")
        
        return warnings
    
    def _generate_improvement_suggestions(self, temporal_graph: TemporalGraph, violations: List[TemporalViolation]) -> List[str]:
        """Generate suggestions for improving temporal consistency"""
        suggestions = []
        
        # Count violation types
        violation_types = {}
        for violation in violations:
            violation_types[violation.violation_type] = violation_types.get(violation.violation_type, 0) + 1
        
        # Generate targeted suggestions
        if violation_types.get('causal_paradox', 0) > 0:
            suggestions.append("Review causal relationships and timestamps - effects should not precede causes")
        
        if violation_types.get('sequence_paradox', 0) > 0:
            suggestions.append("Align sequence numbers with temporal order - renumber events chronologically")
        
        if violation_types.get('duration_logic', 0) > 0:
            suggestions.append("Verify process durations for reasonableness - break long processes into phases")
        
        if violation_types.get('relationship_mismatch', 0) > 0:
            suggestions.append("Ensure temporal relationship labels match actual timing data")
        
        # General suggestions based on data quality
        nodes_with_timestamps = sum(1 for node in temporal_graph.temporal_nodes.values() if node.timestamp)
        total_nodes = len(temporal_graph.temporal_nodes)
        
        if nodes_with_timestamps / total_nodes < 0.5:
            suggestions.append("Consider adding more specific timestamps to improve temporal analysis accuracy")
        
        if len(temporal_graph.temporal_constraints) == 0:
            suggestions.append("Add explicit temporal constraints to validate process timing requirements")
        
        return suggestions
    
    def _calculate_validation_confidence(self, temporal_graph: TemporalGraph, violations: List[TemporalViolation]) -> float:
        """Calculate overall confidence in temporal validation"""
        base_confidence = 1.0
        
        # Reduce confidence based on violations
        for violation in violations:
            confidence_impact = violation.severity * 0.2  # Max 20% impact per violation
            base_confidence -= confidence_impact
        
        # Adjust based on data completeness
        nodes_with_temporal_data = sum(
            1 for node in temporal_graph.temporal_nodes.values()
            if node.timestamp or node.sequence_order is not None
        )
        total_nodes = len(temporal_graph.temporal_nodes)
        
        if total_nodes > 0:
            data_completeness = nodes_with_temporal_data / total_nodes
            base_confidence *= data_completeness
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_validation_report(self, temporal_graph: TemporalGraph, violations: List[TemporalViolation], 
                                  warnings: List[str], suggestions: List[str], confidence: float) -> str:
        """Generate comprehensive validation report"""
        
        stats = temporal_graph.get_temporal_statistics()
        
        report = f"""
TEMPORAL VALIDATION REPORT
========================

OVERALL ASSESSMENT:
- Validation Status: {'PASSED' if len([v for v in violations if v.severity >= 0.8]) == 0 else 'FAILED'}
- Confidence Score: {confidence:.2f}/1.00
- Total Violations: {len(violations)}
- Critical Violations: {len([v for v in violations if v.severity >= 0.8])}

TEMPORAL DATA SUMMARY:
- Total Nodes: {stats.get('total_nodes', 0)}
- Nodes with Timestamps: {stats.get('nodes_with_timestamps', 0)}
- Nodes with Duration: {stats.get('nodes_with_duration', 0)}
- Nodes with Sequence: {stats.get('nodes_with_sequence', 0)}
- Temporal Span: {stats.get('temporal_span', 'Unknown')}

VIOLATIONS BY SEVERITY:
"""
        
        # Group violations by severity
        severity_groups = {'Critical (≥0.8)': [], 'High (0.6-0.8)': [], 'Medium (0.4-0.6)': [], 'Low (<0.4)': []}
        
        for violation in violations:
            if violation.severity >= 0.8:
                severity_groups['Critical (≥0.8)'].append(violation)
            elif violation.severity >= 0.6:
                severity_groups['High (0.6-0.8)'].append(violation)
            elif violation.severity >= 0.4:
                severity_groups['Medium (0.4-0.6)'].append(violation)
            else:
                severity_groups['Low (<0.4)'].append(violation)
        
        for severity, group_violations in severity_groups.items():
            report += f"\n{severity}: {len(group_violations)} violations\n"
            for violation in group_violations[:3]:  # Show first 3 of each type
                report += f"  - {violation.description}\n"
            if len(group_violations) > 3:
                report += f"  ... and {len(group_violations) - 3} more\n"
        
        if warnings:
            report += f"\nWARNINGS:\n"
            for warning in warnings:
                report += f"- {warning}\n"
        
        if suggestions:
            report += f"\nSUGGESTIONS FOR IMPROVEMENT:\n"
            for suggestion in suggestions:
                report += f"- {suggestion}\n"
        
        return report

def test_temporal_validator():
    """Test function for temporal validator"""
    from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge
    from core.temporal_extraction import TemporalRelation, TemporalType
    
    # Create test temporal graph
    tg = TemporalGraph()
    
    # Add nodes with temporal paradox
    node1 = TemporalNode(
        node_id="cause",
        timestamp=datetime(2020, 3, 1),  # After effect - should trigger violation
        node_type="Event"
    )
    
    node2 = TemporalNode(
        node_id="effect",
        timestamp=datetime(2020, 1, 1),  # Before cause - temporal paradox
        node_type="Event"
    )
    
    tg.add_temporal_node(node1)
    tg.add_temporal_node(node2)
    
    # Add causal edge
    edge = TemporalEdge(
        source="cause",
        target="effect",
        temporal_relation=TemporalRelation.BEFORE,
        edge_type="causes"
    )
    
    tg.add_temporal_edge(edge)
    
    # Validate
    validator = TemporalValidator()
    result = validator.validate_temporal_graph(tg)
    
    print("Validation Result:")
    print(f"Valid: {result.is_valid}")
    print(f"Violations: {len(result.violations)}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print("\nReport:")
    print(result.validation_report)

if __name__ == "__main__":
    test_temporal_validator()
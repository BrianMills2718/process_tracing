"""
Temporal Process Tracing - Temporal Graph Extension Module

Extends graph structures with temporal attributes and validates temporal
consistency in causal sequences for process tracing analysis.

Author: Claude Code Implementation  
Date: August 2025
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from core.temporal_extraction import TemporalExpression, TemporalRelation, TemporalType

class SemanticConstraintType(Enum):
    STRICT_ORDERING = "strict_ordering"  # A must come before B
    OVERLAP_ALLOWED = "overlap_allowed"  # A and B can overlap
    CONCURRENT_REQUIRED = "concurrent_required"  # A and B must be simultaneous
    DURATION_CONSTRAINT = "duration_constraint"  # A must last X time
    GAP_CONSTRAINT = "gap_constraint"  # X time between A and B

@dataclass
class TemporalNode:
    """Extended node with temporal attributes"""
    node_id: str
    timestamp: Optional[datetime] = None
    duration: Optional[timedelta] = None
    semantic_uncertainty: float = 0.0
    sequence_order: Optional[int] = None
    semantic_type: TemporalType = TemporalType.UNCERTAIN
    temporal_expressions: List[TemporalExpression] = field(default_factory=list)
    
    # Original node attributes
    node_type: str = ""
    attr_props: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalEdge:
    """Extended edge with temporal relationships"""
    source: str
    target: str
    temporal_relation: TemporalRelation
    temporal_gap: Optional[timedelta] = None
    confidence: float = 0.8
    evidence_text: str = ""
    
    # Original edge attributes
    edge_type: str = "causes"
    attr_props: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalConstraint:
    """Represents a temporal constraint between nodes"""
    constraint_id: str
    constraint_type: SemanticConstraintType
    nodes: List[str]
    constraint_value: Any  # Duration, gap time, etc.
    confidence: float
    description: str

@dataclass
class TemporalViolation:
    """Represents a violation of temporal logic"""
    violation_id: str
    violation_type: str
    nodes_involved: List[str]
    description: str
    severity: float  # 0.0 = minor, 1.0 = critical
    suggested_fix: str

class TemporalGraph:
    """
    Extended graph class that handles temporal attributes and constraints
    for process tracing analysis.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.temporal_nodes: Dict[str, TemporalNode] = {}
        self.temporal_edges: Dict[Tuple[str, str], TemporalEdge] = {}
        self.semantic_constraints: List[TemporalConstraint] = []
        self.temporal_violations: List[TemporalViolation] = []
    
    def add_temporal_node(self, node: TemporalNode):
        """Add a node with temporal attributes"""
        self.temporal_nodes[node.node_id] = node
        
        # Add to NetworkX graph with temporal attributes
        self.graph.add_node(
            node.node_id,
            timestamp=node.timestamp,
            duration=node.duration,
            temporal_uncertainty=node.semantic_uncertainty,
            sequence_order=node.sequence_order,
            temporal_type=node.semantic_type.value,
            node_type=node.node_type,
            attr_props=node.attr_props
        )
    
    def add_temporal_edge(self, edge: TemporalEdge):
        """Add an edge with temporal relationships"""
        edge_key = (edge.source, edge.target)
        self.temporal_edges[edge_key] = edge
        
        # Add to NetworkX graph with temporal attributes
        self.graph.add_edge(
            edge.source,
            edge.target,
            temporal_relation=edge.temporal_relation.value,
            temporal_gap=edge.temporal_gap,
            confidence=edge.confidence,
            evidence_text=edge.evidence_text,
            edge_type=edge.edge_type,
            attr_props=edge.attr_props
        )
    
    def add_temporal_constraint(self, constraint: TemporalConstraint):
        """Add a temporal constraint"""
        self.semantic_constraints.append(constraint)
    
    def get_temporal_sequence(self) -> List[str]:
        """Get nodes ordered by temporal sequence"""
        nodes_with_time = []
        
        for node_id, node in self.temporal_nodes.items():
            if node.timestamp:
                nodes_with_time.append((node.timestamp, node_id))
            elif node.sequence_order is not None:
                # Use sequence order as proxy for time
                fake_time = datetime(2000, 1, 1) + timedelta(days=node.sequence_order)
                nodes_with_time.append((fake_time, node_id))
        
        # Sort by timestamp
        nodes_with_time.sort(key=lambda x: x[0])
        return [node_id for _, node_id in nodes_with_time]
    
    def validate_temporal_consistency(self) -> List[TemporalViolation]:
        """
        Validate temporal consistency of the graph and identify violations.
        """
        violations = []
        
        # Check for temporal paradoxes (effect before cause)
        violations.extend(self._check_causal_ordering())
        
        # Check constraint violations
        violations.extend(self._check_constraint_violations())
        
        # Check for impossible temporal gaps
        violations.extend(self._check_temporal_gaps())
        
        # Check for sequence consistency
        violations.extend(self._check_sequence_consistency())
        
        self.temporal_violations = violations
        return violations
    
    def _check_causal_ordering(self) -> List[TemporalViolation]:
        """Check that causes precede effects"""
        violations = []
        
        for edge_key, edge in self.temporal_edges.items():
            source_node = self.temporal_nodes.get(edge.source)
            target_node = self.temporal_nodes.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # Check if both nodes have timestamps
            if source_node.timestamp and target_node.timestamp:
                if edge.edge_type == "causes" and source_node.timestamp > target_node.timestamp:
                    violations.append(TemporalViolation(
                        violation_id=f"causal_ordering_{edge.source}_{edge.target}",
                        violation_type="causal_ordering",
                        nodes_involved=[edge.source, edge.target],
                        description=f"Cause '{edge.source}' occurs after effect '{edge.target}'",
                        severity=0.9,
                        suggested_fix="Check timestamps or causal relationship direction"
                    ))
            
            # Check sequence order consistency
            if (source_node.sequence_order is not None and 
                target_node.sequence_order is not None and
                edge.edge_type == "causes" and
                source_node.sequence_order > target_node.sequence_order):
                
                violations.append(TemporalViolation(
                    violation_id=f"sequence_ordering_{edge.source}_{edge.target}",
                    violation_type="sequence_ordering",
                    nodes_involved=[edge.source, edge.target],
                    description=f"Cause '{edge.source}' has higher sequence number than effect '{edge.target}'",
                    severity=0.7,
                    suggested_fix="Verify sequence ordering or causal direction"
                ))
        
        return violations
    
    def _check_constraint_violations(self) -> List[TemporalViolation]:
        """Check violations of explicit temporal constraints"""
        violations = []
        
        for constraint in self.semantic_constraints:
            if constraint.constraint_type == SemanticConstraintType.STRICT_ORDERING:
                violations.extend(self._check_strict_ordering_constraint(constraint))
            elif constraint.constraint_type == SemanticConstraintType.DURATION_CONSTRAINT:
                violations.extend(self._check_duration_constraint(constraint))
            elif constraint.constraint_type == SemanticConstraintType.GAP_CONSTRAINT:
                violations.extend(self._check_gap_constraint(constraint))
            elif constraint.constraint_type == SemanticConstraintType.CONCURRENT_REQUIRED:
                violations.extend(self._check_concurrency_constraint(constraint))
        
        return violations
    
    def _check_strict_ordering_constraint(self, constraint: TemporalConstraint) -> List[TemporalViolation]:
        """Check strict ordering constraints"""
        violations = []
        
        if len(constraint.nodes) < 2:
            return violations
        
        for i in range(len(constraint.nodes) - 1):
            node1 = self.temporal_nodes.get(constraint.nodes[i])
            node2 = self.temporal_nodes.get(constraint.nodes[i + 1])
            
            if node1 and node2 and node1.timestamp and node2.timestamp:
                if node1.timestamp >= node2.timestamp:
                    violations.append(TemporalViolation(
                        violation_id=f"strict_order_{constraint.constraint_id}_{i}",
                        violation_type="strict_ordering",
                        nodes_involved=[constraint.nodes[i], constraint.nodes[i + 1]],
                        description=f"Strict ordering violation: {constraint.nodes[i]} should precede {constraint.nodes[i + 1]}",
                        severity=0.8,
                        suggested_fix="Adjust timestamps to maintain ordering"
                    ))
        
        return violations
    
    def _check_duration_constraint(self, constraint: TemporalConstraint) -> List[TemporalViolation]:
        """Check duration constraints"""
        violations = []
        
        for node_id in constraint.nodes:
            node = self.temporal_nodes.get(node_id)
            if node and node.duration:
                expected_duration = constraint.constraint_value
                if isinstance(expected_duration, timedelta):
                    # Allow 20% tolerance
                    tolerance = expected_duration * 0.2
                    if abs(node.duration - expected_duration) > tolerance:
                        violations.append(TemporalViolation(
                            violation_id=f"duration_{constraint.constraint_id}_{node_id}",
                            violation_type="duration_constraint",
                            nodes_involved=[node_id],
                            description=f"Duration constraint violation: {node_id} expected {expected_duration}, got {node.duration}",
                            severity=0.6,
                            suggested_fix="Verify duration data or adjust constraint"
                        ))
        
        return violations
    
    def _check_gap_constraint(self, constraint: TemporalConstraint) -> List[TemporalViolation]:
        """Check temporal gap constraints"""
        violations = []
        
        if len(constraint.nodes) < 2:
            return violations
        
        for i in range(len(constraint.nodes) - 1):
            node1 = self.temporal_nodes.get(constraint.nodes[i])
            node2 = self.temporal_nodes.get(constraint.nodes[i + 1])
            
            if node1 and node2 and node1.timestamp and node2.timestamp:
                actual_gap = node2.timestamp - node1.timestamp
                expected_gap = constraint.constraint_value
                
                if isinstance(expected_gap, timedelta):
                    # Allow 20% tolerance
                    tolerance = expected_gap * 0.2
                    if abs(actual_gap - expected_gap) > tolerance:
                        violations.append(TemporalViolation(
                            violation_id=f"gap_{constraint.constraint_id}_{i}",
                            violation_type="gap_constraint",
                            nodes_involved=[constraint.nodes[i], constraint.nodes[i + 1]],
                            description=f"Gap constraint violation: expected {expected_gap}, got {actual_gap}",
                            severity=0.7,
                            suggested_fix="Verify timing data or adjust gap constraint"
                        ))
        
        return violations
    
    def _check_concurrency_constraint(self, constraint: TemporalConstraint) -> List[TemporalViolation]:
        """Check concurrency constraints"""
        violations = []
        
        timestamps = []
        for node_id in constraint.nodes:
            node = self.temporal_nodes.get(node_id)
            if node and node.timestamp:
                timestamps.append((node.timestamp, node_id))
        
        if len(timestamps) > 1:
            # Check if all timestamps are within a reasonable window (e.g., 1 day)
            timestamps.sort()
            max_gap = timestamps[-1][0] - timestamps[0][0]
            allowed_gap = timedelta(days=1)  # Configurable tolerance
            
            if max_gap > allowed_gap:
                violations.append(TemporalViolation(
                    violation_id=f"concurrency_{constraint.constraint_id}",
                    violation_type="concurrent_required",
                    nodes_involved=constraint.nodes,
                    description=f"Concurrency constraint violation: events span {max_gap}, exceeding allowed {allowed_gap}",
                    severity=0.8,
                    suggested_fix="Verify that events were truly concurrent or adjust constraint"
                ))
        
        return violations
    
    def _check_temporal_gaps(self) -> List[TemporalViolation]:
        """Check for unreasonable temporal gaps"""
        violations = []
        
        for edge_key, edge in self.temporal_edges.items():
            source_node = self.temporal_nodes.get(edge.source)
            target_node = self.temporal_nodes.get(edge.target)
            
            if source_node and target_node and source_node.timestamp and target_node.timestamp:
                gap = target_node.timestamp - source_node.timestamp
                
                # Check for unreasonably long gaps (configurable)
                max_reasonable_gap = timedelta(days=365 * 10)  # 10 years
                if gap > max_reasonable_gap:
                    violations.append(TemporalViolation(
                        violation_id=f"long_gap_{edge.source}_{edge.target}",
                        violation_type="unreasonable_gap",
                        nodes_involved=[edge.source, edge.target],
                        description=f"Unreasonably long temporal gap: {gap}",
                        severity=0.5,
                        suggested_fix="Verify timestamps or consider intermediate events"
                    ))
                
                # Check for negative gaps (already covered in causal ordering but different severity)
                if gap < timedelta(0) and edge.edge_type != "causes":
                    violations.append(TemporalViolation(
                        violation_id=f"negative_gap_{edge.source}_{edge.target}",
                        violation_type="negative_gap",
                        nodes_involved=[edge.source, edge.target],
                        description=f"Negative temporal gap: {gap}",
                        severity=0.6,
                        suggested_fix="Check edge direction or timestamps"
                    ))
        
        return violations
    
    def _check_sequence_consistency(self) -> List[TemporalViolation]:
        """Check consistency of sequence ordering"""
        violations = []
        
        # Get nodes with sequence orders
        sequenced_nodes = [
            (node.sequence_order, node_id) 
            for node_id, node in self.temporal_nodes.items() 
            if node.sequence_order is not None
        ]
        
        if len(sequenced_nodes) < 2:
            return violations
        
        sequenced_nodes.sort()
        
        # Check for duplicate sequence numbers
        sequence_numbers = [seq for seq, _ in sequenced_nodes]
        if len(sequence_numbers) != len(set(sequence_numbers)):
            # Find duplicates
            seen = set()
            duplicates = set()
            for seq in sequence_numbers:
                if seq in seen:
                    duplicates.add(seq)
                seen.add(seq)
            
            for dup_seq in duplicates:
                dup_nodes = [node_id for seq, node_id in sequenced_nodes if seq == dup_seq]
                violations.append(TemporalViolation(
                    violation_id=f"duplicate_sequence_{dup_seq}",
                    violation_type="sequence_consistency",
                    nodes_involved=dup_nodes,
                    description=f"Duplicate sequence number {dup_seq}",
                    severity=0.7,
                    suggested_fix="Assign unique sequence numbers"
                ))
        
        return violations
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Generate temporal statistics for the graph"""
        stats = {
            'total_nodes': len(self.temporal_nodes),
            'nodes_with_timestamps': sum(1 for node in self.temporal_nodes.values() if node.timestamp),
            'nodes_with_duration': sum(1 for node in self.temporal_nodes.values() if node.duration),
            'nodes_with_sequence': sum(1 for node in self.temporal_nodes.values() if node.sequence_order is not None),
            'total_edges': len(self.temporal_edges),
            'temporal_violations': len(self.temporal_violations),
            'temporal_constraints': len(self.semantic_constraints),
        }
        
        # Calculate temporal span
        timestamps = [
            node.timestamp for node in self.temporal_nodes.values() 
            if node.timestamp
        ]
        if timestamps:
            stats['temporal_span'] = max(timestamps) - min(timestamps)
            stats['earliest_event'] = min(timestamps)
            stats['latest_event'] = max(timestamps)
        
        # Calculate average uncertainty
        uncertainties = [
            node.semantic_uncertainty for node in self.temporal_nodes.values()
            if node.semantic_uncertainty > 0
        ]
        if uncertainties:
            stats['average_uncertainty'] = sum(uncertainties) / len(uncertainties)
        
        return stats
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to standard NetworkX graph"""
        return self.graph.copy()
    
    def from_standard_graph(self, graph: nx.DiGraph, temporal_expressions: List[TemporalExpression] = None):
        """
        Convert from standard graph format, optionally incorporating temporal expressions.
        """
        # Add nodes
        for node_id, data in graph.nodes(data=True):
            temporal_node = TemporalNode(
                node_id=node_id,
                node_type=data.get('node_type', ''),
                attr_props=data.get('attr_props', {})
            )
            self.add_temporal_node(temporal_node)
        
        # Add edges
        for source, target, data in graph.edges(data=True):
            temporal_edge = TemporalEdge(
                source=source,
                target=target,
                temporal_relation=TemporalRelation.PRECEDES,  # Default
                edge_type=data.get('edge_type', 'causes'),
                attr_props=data.get('attr_props', {})
            )
            self.add_temporal_edge(temporal_edge)
        
        # Incorporate temporal expressions if provided
        if temporal_expressions:
            self._incorporate_temporal_expressions(temporal_expressions)
    
    def _incorporate_temporal_expressions(self, expressions: List[TemporalExpression]):
        """Incorporate temporal expressions into node attributes"""
        # This is a simplified approach - in practice, would need more sophisticated
        # matching between expressions and nodes based on context
        
        for expr in expressions:
            # Try to match expression to nodes based on context
            matching_nodes = self._find_matching_nodes(expr)
            
            for node_id in matching_nodes:
                if node_id in self.temporal_nodes:
                    node = self.temporal_nodes[node_id]
                    node.temporal_expressions.append(expr)
                    
                    # Update node temporal attributes based on expression
                    if expr.normalized_value and not node.timestamp:
                        node.timestamp = expr.normalized_value
                        node.semantic_uncertainty = expr.uncertainty
                        node.semantic_type = expr.temporal_type
                    
                    if expr.duration and not node.duration:
                        node.duration = expr.duration
    
    def _find_matching_nodes(self, expression: TemporalExpression) -> List[str]:
        """Find nodes that might be associated with a temporal expression"""
        # Simple keyword matching - could be enhanced with NLP
        matches = []
        
        for node_id, node in self.temporal_nodes.items():
            node_desc = node.attr_props.get('description', '')
            # Use semantic analysis to match temporal expression context
            from core.semantic_analysis_service import get_semantic_service
            semantic_service = get_semantic_service()
            
            assessment = semantic_service.assess_probative_value(
                evidence_description=node_desc,
                hypothesis_description=f"Node matches temporal expression context: {expression.context}",
                context="Matching nodes to temporal expression patterns"
            )
            if assessment.confidence_score > 0.6:
                matches.append(node_id)
        
        return matches

def test_temporal_graph():
    """Test function for temporal graph functionality"""
    tg = TemporalGraph()
    
    # Add temporal nodes
    node1 = TemporalNode(
        node_id="event1",
        timestamp=datetime(2020, 1, 15),
        duration=timedelta(days=30),
        sequence_order=1,
        semantic_type=TemporalType.ABSOLUTE,
        node_type="Event",
        attr_props={"description": "Initial policy announcement"}
    )
    
    node2 = TemporalNode(
        node_id="event2", 
        timestamp=datetime(2020, 3, 15),
        sequence_order=2,
        semantic_type=TemporalType.ABSOLUTE,
        node_type="Event",
        attr_props={"description": "Policy reversal"}
    )
    
    tg.add_temporal_node(node1)
    tg.add_temporal_node(node2)
    
    # Add temporal edge
    edge = TemporalEdge(
        source="event1",
        target="event2",
        temporal_relation=TemporalRelation.BEFORE,
        confidence=0.9,
        evidence_text="Policy reversal came after initial announcement"
    )
    
    tg.add_temporal_edge(edge)
    
    # Validate temporal consistency
    violations = tg.validate_temporal_consistency()
    print(f"Temporal violations found: {len(violations)}")
    
    # Get temporal sequence
    sequence = tg.get_temporal_sequence()
    print(f"Temporal sequence: {sequence}")
    
    # Get statistics
    stats = tg.get_temporal_statistics()
    print(f"Temporal statistics: {stats}")

if __name__ == "__main__":
    test_temporal_graph()
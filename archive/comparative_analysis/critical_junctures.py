"""
Temporal Process Tracing - Critical Juncture Analysis Module

Identifies key decision points and temporal branching moments in causal
processes where timing significantly affected outcomes.

Author: Claude Code Implementation  
Date: August 2025
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge
from core.temporal_extraction import TemporalRelation

class JunctureType(Enum):
    DECISION_POINT = "decision_point"      # Deliberate choice between alternatives
    BRANCHING_POINT = "branching_point"    # Natural divergence of pathways
    CONVERGENCE_POINT = "convergence_point" # Multiple pathways merge
    TIMING_CRITICAL = "timing_critical"    # Outcome sensitive to precise timing
    THRESHOLD_CROSSING = "threshold_crossing" # System state change

@dataclass
class AlternativePathway:
    """Represents an alternative causal pathway from a juncture"""
    pathway_id: str
    description: str
    probability: float  # Estimated likelihood this path could have occurred
    outcome_difference: str  # How outcome would differ
    evidence_requirements: List[str]  # What evidence would support this path
    plausibility_score: float

@dataclass
class CriticalJuncture:
    """Represents a critical decision or branching point in time"""
    juncture_id: str
    timestamp: Optional[datetime]
    sequence_position: Optional[int]
    juncture_type: JunctureType
    description: str
    decision_point: str
    
    # Key nodes involved in the juncture
    key_nodes: List[str]
    preceding_events: List[str]
    following_events: List[str]
    
    # Alternative analysis
    alternative_pathways: List[AlternativePathway]
    actual_pathway: str
    
    # Impact assessment
    timing_sensitivity: float    # 0.0 = timing irrelevant, 1.0 = timing critical
    counterfactual_impact: float # 0.0 = no impact, 1.0 = completely different outcome
    confidence: float           # Confidence in juncture identification
    
    # Supporting evidence
    evidence_support: List[str]
    temporal_window: Optional[Tuple[datetime, datetime]]  # Window when juncture was active

@dataclass
class JunctureAnalysisResult:
    """Results of critical juncture analysis"""
    total_junctures: int
    junctures_by_type: Dict[JunctureType, int]
    high_impact_junctures: List[CriticalJuncture]
    timing_critical_junctures: List[CriticalJuncture]
    temporal_distribution: Dict[str, int]  # Junctures by time period
    overall_timing_sensitivity: float

class CriticalJunctureAnalyzer:
    """
    Analyzes temporal graphs to identify critical junctures where timing
    significantly affected causal outcomes.
    """
    
    def __init__(self):
        self.juncture_detection_rules = self._initialize_detection_rules()
    
    def _initialize_detection_rules(self) -> Dict[str, Any]:
        """Initialize rules for detecting different types of junctures"""
        return {
            'min_alternative_pathways': 2,           # Minimum alternatives for decision point
            'timing_sensitivity_threshold': 0.6,     # Threshold for timing-critical junctures
            'convergence_node_threshold': 2,         # Minimum incoming edges for convergence
            'branching_node_threshold': 2,           # Minimum outgoing edges for branching
            'temporal_window_hours': 24,             # Hours around juncture for analysis
            'confidence_threshold': 0.7,             # Minimum confidence for valid juncture
        }
    
    def identify_junctures(self, temporal_graph: TemporalGraph) -> List[CriticalJuncture]:
        """
        Identify all critical junctures in the temporal graph.
        """
        junctures = []
        
        # Detect different types of junctures
        junctures.extend(self._detect_decision_points(temporal_graph))
        junctures.extend(self._detect_branching_points(temporal_graph))
        junctures.extend(self._detect_convergence_points(temporal_graph))
        junctures.extend(self._detect_timing_critical_points(temporal_graph))
        junctures.extend(self._detect_threshold_crossings(temporal_graph))
        
        # Remove duplicates and low-confidence junctures
        junctures = self._deduplicate_junctures(junctures)
        junctures = [j for j in junctures if j.confidence >= self.juncture_detection_rules['confidence_threshold']]
        
        # Sort by impact and timing sensitivity
        junctures.sort(key=lambda j: (j.counterfactual_impact, j.timing_sensitivity), reverse=True)
        
        return junctures
    
    def _detect_decision_points(self, temporal_graph: TemporalGraph) -> List[CriticalJuncture]:
        """Detect deliberate decision points where choices were made"""
        junctures = []
        
        # Look for nodes that represent decisions or policy choices
        decision_keywords = [
            'decision', 'choice', 'policy', 'strategy', 'plan', 'resolution',
            'announcement', 'declaration', 'vote', 'approval', 'rejection'
        ]
        
        for node_id, node in temporal_graph.temporal_nodes.items():
            node_desc = node.attr_props.get('description', '').lower()
            
            # Check if node represents a decision
            if any(keyword in node_desc for keyword in decision_keywords):
                
                # Analyze outgoing paths to identify alternatives
                graph = temporal_graph.to_networkx()
                successors = list(graph.successors(node_id))
                
                if len(successors) >= self.juncture_detection_rules['min_alternative_pathways']:
                    
                    # Generate alternative pathways
                    alternatives = self._generate_alternative_pathways(
                        temporal_graph, node_id, successors
                    )
                    
                    # Calculate timing sensitivity
                    timing_sensitivity = self._calculate_decision_timing_sensitivity(
                        temporal_graph, node_id
                    )
                    
                    # Calculate counterfactual impact
                    counterfactual_impact = self._assess_decision_impact(
                        temporal_graph, node_id, alternatives
                    )
                    
                    juncture = CriticalJuncture(
                        juncture_id=f"decision_{node_id}",
                        timestamp=node.timestamp,
                        sequence_position=node.sequence_order,
                        juncture_type=JunctureType.DECISION_POINT,
                        description=f"Decision point at {node_desc}",
                        decision_point=node_desc,
                        key_nodes=[node_id],
                        preceding_events=self._get_preceding_events(temporal_graph, node_id),
                        following_events=successors,
                        alternative_pathways=alternatives,
                        actual_pathway=self._identify_actual_pathway(temporal_graph, node_id),
                        timing_sensitivity=timing_sensitivity,
                        counterfactual_impact=counterfactual_impact,
                        confidence=0.8,  # High confidence for explicit decisions
                        evidence_support=[node_desc],
                        temporal_window=self._calculate_temporal_window(node)
                    )
                    
                    junctures.append(juncture)
        
        return junctures
    
    def _detect_branching_points(self, temporal_graph: TemporalGraph) -> List[CriticalJuncture]:
        """Detect natural branching points where multiple pathways diverge"""
        junctures = []
        
        graph = temporal_graph.to_networkx()
        
        for node_id in graph.nodes():
            successors = list(graph.successors(node_id))
            
            # Look for nodes with multiple outgoing edges (branching)
            if len(successors) >= self.juncture_detection_rules['branching_node_threshold']:
                
                node = temporal_graph.temporal_nodes.get(node_id)
                if not node:
                    continue
                
                # Check if this is a natural branching (not a decision)
                node_desc = node.attr_props.get('description', '').lower()
                decision_keywords = ['decision', 'choice', 'policy', 'vote']
                
                if not any(keyword in node_desc for keyword in decision_keywords):
                    
                    # Analyze divergent pathways
                    alternatives = self._analyze_divergent_pathways(
                        temporal_graph, node_id, successors
                    )
                    
                    timing_sensitivity = self._calculate_branching_timing_sensitivity(
                        temporal_graph, node_id
                    )
                    
                    counterfactual_impact = self._assess_branching_impact(
                        temporal_graph, node_id, alternatives
                    )
                    
                    juncture = CriticalJuncture(
                        juncture_id=f"branching_{node_id}",
                        timestamp=node.timestamp,
                        sequence_position=node.sequence_order,
                        juncture_type=JunctureType.BRANCHING_POINT,
                        description=f"Natural branching at {node_desc}",
                        decision_point="Natural divergence of causal pathways",
                        key_nodes=[node_id],
                        preceding_events=self._get_preceding_events(temporal_graph, node_id),
                        following_events=successors,
                        alternative_pathways=alternatives,
                        actual_pathway="Multiple pathways activated",
                        timing_sensitivity=timing_sensitivity,
                        counterfactual_impact=counterfactual_impact,
                        confidence=0.6,  # Lower confidence for natural branching
                        evidence_support=[f"Multiple pathways from {node_desc}"],
                        temporal_window=self._calculate_temporal_window(node)
                    )
                    
                    junctures.append(juncture)
        
        return junctures
    
    def _detect_convergence_points(self, temporal_graph: TemporalGraph) -> List[CriticalJuncture]:
        """Detect convergence points where multiple pathways merge"""
        junctures = []
        
        graph = temporal_graph.to_networkx()
        
        for node_id in graph.nodes():
            predecessors = list(graph.predecessors(node_id))
            
            # Look for nodes with multiple incoming edges (convergence)
            if len(predecessors) >= self.juncture_detection_rules['convergence_node_threshold']:
                
                node = temporal_graph.temporal_nodes.get(node_id)
                if not node:
                    continue
                
                # Analyze converging pathways
                timing_sensitivity = self._calculate_convergence_timing_sensitivity(
                    temporal_graph, node_id, predecessors
                )
                
                counterfactual_impact = self._assess_convergence_impact(
                    temporal_graph, node_id, predecessors
                )
                
                # Generate alternative scenarios where pathways didn't converge
                alternatives = self._generate_convergence_alternatives(
                    temporal_graph, node_id, predecessors
                )
                
                juncture = CriticalJuncture(
                    juncture_id=f"convergence_{node_id}",
                    timestamp=node.timestamp,
                    sequence_position=node.sequence_order,
                    juncture_type=JunctureType.CONVERGENCE_POINT,
                    description=f"Convergence point at {node.attr_props.get('description', node_id)}",
                    decision_point="Multiple pathways converge to single outcome",
                    key_nodes=[node_id] + predecessors,
                    preceding_events=predecessors,
                    following_events=list(graph.successors(node_id)),
                    alternative_pathways=alternatives,
                    actual_pathway="All pathways converged",
                    timing_sensitivity=timing_sensitivity,
                    counterfactual_impact=counterfactual_impact,
                    confidence=0.7,
                    evidence_support=[f"Multiple inputs to {node.attr_props.get('description', node_id)}"],
                    temporal_window=self._calculate_temporal_window(node)
                )
                
                junctures.append(juncture)
        
        return junctures
    
    def _detect_timing_critical_points(self, temporal_graph: TemporalGraph) -> List[CriticalJuncture]:
        """Detect points where precise timing was critical for outcomes"""
        junctures = []
        
        # Look for temporal relationships that suggest timing criticality
        timing_keywords = [
            'deadline', 'window', 'opportunity', 'crisis', 'emergency',
            'rapid', 'immediate', 'urgent', 'time-sensitive', 'critical moment'
        ]
        
        for node_id, node in temporal_graph.temporal_nodes.items():
            node_desc = node.attr_props.get('description', '').lower()
            
            # Check for timing-critical language
            if any(keyword in node_desc for keyword in timing_keywords):
                
                timing_sensitivity = self._calculate_timing_criticality(
                    temporal_graph, node_id
                )
                
                if timing_sensitivity >= self.juncture_detection_rules['timing_sensitivity_threshold']:
                    
                    # Analyze what would happen with different timing
                    alternatives = self._generate_timing_alternatives(
                        temporal_graph, node_id
                    )
                    
                    counterfactual_impact = self._assess_timing_impact(
                        temporal_graph, node_id
                    )
                    
                    juncture = CriticalJuncture(
                        juncture_id=f"timing_{node_id}",
                        timestamp=node.timestamp,
                        sequence_position=node.sequence_order,
                        juncture_type=JunctureType.TIMING_CRITICAL,
                        description=f"Timing-critical moment: {node_desc}",
                        decision_point="Precise timing critical for outcome",
                        key_nodes=[node_id],
                        preceding_events=self._get_preceding_events(temporal_graph, node_id),
                        following_events=list(temporal_graph.to_networkx().successors(node_id)),
                        alternative_pathways=alternatives,
                        actual_pathway="Optimal timing achieved",
                        timing_sensitivity=timing_sensitivity,
                        counterfactual_impact=counterfactual_impact,
                        confidence=0.8,
                        evidence_support=[f"Timing-critical: {node_desc}"],
                        temporal_window=self._calculate_temporal_window(node)
                    )
                    
                    junctures.append(juncture)
        
        return junctures
    
    def _detect_threshold_crossings(self, temporal_graph: TemporalGraph) -> List[CriticalJuncture]:
        """Detect threshold crossings where system state changed qualitatively"""
        junctures = []
        
        threshold_keywords = [
            'threshold', 'tipping point', 'breaking point', 'critical mass',
            'phase transition', 'regime change', 'transformation', 'shift'
        ]
        
        for node_id, node in temporal_graph.temporal_nodes.items():
            node_desc = node.attr_props.get('description', '').lower()
            
            if any(keyword in node_desc for keyword in threshold_keywords):
                
                # Analyze system state change
                timing_sensitivity = self._calculate_threshold_timing_sensitivity(
                    temporal_graph, node_id
                )
                
                counterfactual_impact = self._assess_threshold_impact(
                    temporal_graph, node_id
                )
                
                alternatives = self._generate_threshold_alternatives(
                    temporal_graph, node_id
                )
                
                juncture = CriticalJuncture(
                    juncture_id=f"threshold_{node_id}",
                    timestamp=node.timestamp,
                    sequence_position=node.sequence_order,
                    juncture_type=JunctureType.THRESHOLD_CROSSING,
                    description=f"Threshold crossing: {node_desc}",
                    decision_point="Critical threshold or tipping point crossed",
                    key_nodes=[node_id],
                    preceding_events=self._get_preceding_events(temporal_graph, node_id),
                    following_events=list(temporal_graph.to_networkx().successors(node_id)),
                    alternative_pathways=alternatives,
                    actual_pathway="Threshold successfully crossed",
                    timing_sensitivity=timing_sensitivity,
                    counterfactual_impact=counterfactual_impact,
                    confidence=0.7,
                    evidence_support=[f"Threshold event: {node_desc}"],
                    temporal_window=self._calculate_temporal_window(node)
                )
                
                junctures.append(juncture)
        
        return junctures
    
    def _generate_alternative_pathways(self, temporal_graph: TemporalGraph, 
                                     node_id: str, successors: List[str]) -> List[AlternativePathway]:
        """Generate alternative pathways from a decision point"""
        alternatives = []
        
        for i, successor in enumerate(successors):
            successor_node = temporal_graph.temporal_nodes.get(successor)
            if successor_node:
                alternative = AlternativePathway(
                    pathway_id=f"alt_{node_id}_{i}",
                    description=f"Alternative pathway through {successor_node.attr_props.get('description', successor)}",
                    probability=0.5,  # Equal probability by default
                    outcome_difference="Different causal outcome",
                    evidence_requirements=[f"Evidence supporting {successor} pathway"],
                    plausibility_score=0.7
                )
                alternatives.append(alternative)
        
        return alternatives
    
    def _analyze_divergent_pathways(self, temporal_graph: TemporalGraph,
                                  node_id: str, successors: List[str]) -> List[AlternativePathway]:
        """Analyze naturally divergent pathways from a branching point"""
        alternatives = []
        
        for successor in successors:
            successor_node = temporal_graph.temporal_nodes.get(successor)
            if successor_node:
                # Analyze pathway characteristics
                pathway_strength = self._calculate_pathway_strength(temporal_graph, successor)
                
                alternative = AlternativePathway(
                    pathway_id=f"divergent_{node_id}_{successor}",
                    description=f"Divergent pathway: {successor_node.attr_props.get('description', successor)}",
                    probability=pathway_strength,
                    outcome_difference="Different causal development",
                    evidence_requirements=[f"Evidence for {successor} pathway strength"],
                    plausibility_score=pathway_strength
                )
                alternatives.append(alternative)
        
        return alternatives
    
    def _generate_convergence_alternatives(self, temporal_graph: TemporalGraph,
                                         node_id: str, predecessors: List[str]) -> List[AlternativePathway]:
        """Generate alternatives where convergence didn't occur"""
        alternatives = []
        
        # Alternative where only some pathways converged
        if len(predecessors) > 2:
            alternative = AlternativePathway(
                pathway_id=f"partial_convergence_{node_id}",
                description="Partial convergence - only some pathways merged",
                probability=0.4,
                outcome_difference="Weaker or incomplete outcome",
                evidence_requirements=["Evidence that all pathways were necessary"],
                plausibility_score=0.6
            )
            alternatives.append(alternative)
        
        # Alternative where convergence was delayed
        alternative = AlternativePathway(
            pathway_id=f"delayed_convergence_{node_id}",
            description="Delayed convergence - pathways merged later",
            probability=0.3,
            outcome_difference="Same outcome but different timing",
            evidence_requirements=["Evidence of timing requirements for convergence"],
            plausibility_score=0.7
        )
        alternatives.append(alternative)
        
        return alternatives
    
    def _generate_timing_alternatives(self, temporal_graph: TemporalGraph, 
                                    node_id: str) -> List[AlternativePathway]:
        """Generate alternatives with different timing"""
        alternatives = []
        
        # Earlier timing
        alternative_early = AlternativePathway(
            pathway_id=f"early_timing_{node_id}",
            description="Earlier timing of critical event",
            probability=0.3,
            outcome_difference="Different context and conditions",
            evidence_requirements=["Evidence that earlier timing was possible"],
            plausibility_score=0.6
        )
        alternatives.append(alternative_early)
        
        # Later timing
        alternative_late = AlternativePathway(
            pathway_id=f"late_timing_{node_id}",
            description="Later timing of critical event",
            probability=0.3,
            outcome_difference="Missed opportunity or changed context",
            evidence_requirements=["Evidence that delay was possible"],
            plausibility_score=0.6
        )
        alternatives.append(alternative_late)
        
        return alternatives
    
    def _generate_threshold_alternatives(self, temporal_graph: TemporalGraph,
                                       node_id: str) -> List[AlternativePathway]:
        """Generate alternatives where threshold wasn't crossed"""
        alternatives = []
        
        alternative = AlternativePathway(
            pathway_id=f"no_threshold_{node_id}",
            description="Threshold not crossed - system remained in previous state",
            probability=0.4,
            outcome_difference="Qualitatively different system state",
            evidence_requirements=["Evidence threshold crossing was necessary"],
            plausibility_score=0.6
        )
        alternatives.append(alternative)
        
        return alternatives
    
    def _calculate_decision_timing_sensitivity(self, temporal_graph: TemporalGraph, 
                                             node_id: str) -> float:
        """Calculate how sensitive a decision was to timing"""
        # Base sensitivity score
        base_score = 0.5
        
        # Look for timing pressure indicators
        node = temporal_graph.temporal_nodes.get(node_id)
        if node:
            node_desc = node.attr_props.get('description', '').lower()
            urgency_keywords = ['urgent', 'immediate', 'deadline', 'crisis', 'emergency']
            
            if any(keyword in node_desc for keyword in urgency_keywords):
                base_score += 0.3
        
        # Check temporal constraints
        if node and node.duration:
            # Short duration suggests time pressure
            if node.duration < timedelta(days=1):
                base_score += 0.2
        
        return min(1.0, base_score)
    
    def _calculate_branching_timing_sensitivity(self, temporal_graph: TemporalGraph,
                                              node_id: str) -> float:
        """Calculate timing sensitivity for natural branching"""
        # Natural branching typically less timing-sensitive than decisions
        return 0.4
    
    def _calculate_convergence_timing_sensitivity(self, temporal_graph: TemporalGraph,
                                                node_id: str, predecessors: List[str]) -> float:
        """Calculate timing sensitivity for convergence points"""
        base_score = 0.3
        
        # More predecessors = higher timing sensitivity for convergence
        if len(predecessors) > 3:
            base_score += 0.2
        
        # Check if predecessors have similar timing
        predecessor_nodes = [temporal_graph.temporal_nodes.get(p) for p in predecessors]
        timestamps = [n.timestamp for n in predecessor_nodes if n and n.timestamp]
        
        if len(timestamps) > 1:
            time_range = max(timestamps) - min(timestamps)
            if time_range < timedelta(days=7):  # Close in time
                base_score += 0.3
        
        return min(1.0, base_score)
    
    def _calculate_timing_criticality(self, temporal_graph: TemporalGraph, 
                                    node_id: str) -> float:
        """Calculate overall timing criticality for a node"""
        return 0.8  # High by definition for timing-critical points
    
    def _calculate_threshold_timing_sensitivity(self, temporal_graph: TemporalGraph,
                                              node_id: str) -> float:
        """Calculate timing sensitivity for threshold crossings"""
        return 0.7  # Threshold crossings are typically timing-sensitive
    
    def _assess_decision_impact(self, temporal_graph: TemporalGraph, node_id: str,
                              alternatives: List[AlternativePathway]) -> float:
        """Assess counterfactual impact of a decision"""
        # More alternatives = higher potential impact
        base_impact = min(0.8, len(alternatives) * 0.2)
        
        # Check downstream effects
        graph = temporal_graph.to_networkx()
        descendants = len(nx.descendants(graph, node_id))
        
        # More downstream effects = higher impact
        if descendants > 5:
            base_impact += 0.2
        
        return min(1.0, base_impact)
    
    def _assess_branching_impact(self, temporal_graph: TemporalGraph, node_id: str,
                               alternatives: List[AlternativePathway]) -> float:
        """Assess impact of natural branching"""
        return 0.6  # Moderate impact for natural branching
    
    def _assess_convergence_impact(self, temporal_graph: TemporalGraph, node_id: str,
                                 predecessors: List[str]) -> float:
        """Assess impact of convergence points"""
        # More converging pathways = higher impact
        return min(0.8, len(predecessors) * 0.2)
    
    def _assess_timing_impact(self, temporal_graph: TemporalGraph, node_id: str) -> float:
        """Assess impact of timing-critical events"""
        return 0.8  # High impact by definition
    
    def _assess_threshold_impact(self, temporal_graph: TemporalGraph, node_id: str) -> float:
        """Assess impact of threshold crossings"""
        return 0.9  # Very high impact for system state changes
    
    def _calculate_pathway_strength(self, temporal_graph: TemporalGraph, node_id: str) -> float:
        """Calculate strength/probability of a pathway"""
        # Simple heuristic based on downstream connections
        graph = temporal_graph.to_networkx()
        descendants = len(nx.descendants(graph, node_id))
        
        # More connections = stronger pathway
        return min(1.0, descendants * 0.1 + 0.3)
    
    def _get_preceding_events(self, temporal_graph: TemporalGraph, node_id: str) -> List[str]:
        """Get events that precede this node"""
        graph = temporal_graph.to_networkx()
        return list(graph.predecessors(node_id))
    
    def _identify_actual_pathway(self, temporal_graph: TemporalGraph, node_id: str) -> str:
        """Identify the actual pathway taken from a decision point"""
        graph = temporal_graph.to_networkx()
        successors = list(graph.successors(node_id))
        
        if successors:
            return f"Pathway through {successors[0]}"
        return "No clear pathway identified"
    
    def _calculate_temporal_window(self, node: TemporalNode) -> Optional[Tuple[datetime, datetime]]:
        """Calculate the temporal window when a juncture was active"""
        if not node.timestamp:
            return None
        
        window_size = timedelta(hours=self.juncture_detection_rules['temporal_window_hours'])
        return (node.timestamp - window_size, node.timestamp + window_size)
    
    def _deduplicate_junctures(self, junctures: List[CriticalJuncture]) -> List[CriticalJuncture]:
        """Remove duplicate junctures based on key nodes and timing"""
        seen = set()
        unique_junctures = []
        
        for juncture in junctures:
            # Create key based on key nodes and approximate timing
            key_nodes = tuple(sorted(juncture.key_nodes))
            time_key = juncture.timestamp.date() if juncture.timestamp else None
            key = (key_nodes, time_key)
            
            if key not in seen:
                seen.add(key)
                unique_junctures.append(juncture)
        
        return unique_junctures
    
    def analyze_timing_sensitivity(self, juncture: CriticalJuncture) -> float:
        """Analyze how sensitive a juncture is to timing variations"""
        return juncture.timing_sensitivity
    
    def assess_counterfactual_impact(self, juncture: CriticalJuncture) -> float:
        """Assess the counterfactual impact of a juncture"""
        return juncture.counterfactual_impact
    
    def generate_juncture_report(self, junctures: List[CriticalJuncture]) -> str:
        """Generate comprehensive report of critical junctures"""
        if not junctures:
            return "No critical junctures identified in the temporal analysis."
        
        # Analyze juncture distribution
        type_counts = {}
        for juncture in junctures:
            type_counts[juncture.juncture_type] = type_counts.get(juncture.juncture_type, 0) + 1
        
        # Identify highest impact junctures
        high_impact = [j for j in junctures if j.counterfactual_impact >= 0.7]
        timing_critical = [j for j in junctures if j.timing_sensitivity >= 0.7]
        
        # Calculate overall metrics
        avg_timing_sensitivity = sum(j.timing_sensitivity for j in junctures) / len(junctures)
        avg_impact = sum(j.counterfactual_impact for j in junctures) / len(junctures)
        
        report = f"""
CRITICAL JUNCTURE ANALYSIS REPORT
================================

OVERVIEW:
- Total Critical Junctures: {len(junctures)}
- High Impact Junctures (≥0.7): {len(high_impact)}
- Timing Critical Junctures (≥0.7): {len(timing_critical)}
- Average Timing Sensitivity: {avg_timing_sensitivity:.2f}
- Average Counterfactual Impact: {avg_impact:.2f}

JUNCTURE TYPES:
"""
        
        for juncture_type, count in type_counts.items():
            report += f"- {juncture_type.value.title()}: {count}\n"
        
        report += f"\nHIGH IMPACT JUNCTURES:\n"
        for juncture in high_impact[:5]:  # Top 5
            report += f"- {juncture.description}\n"
            report += f"  Impact: {juncture.counterfactual_impact:.2f}, Timing Sensitivity: {juncture.timing_sensitivity:.2f}\n"
            report += f"  Alternatives: {len(juncture.alternative_pathways)}\n\n"
        
        report += f"TIMING CRITICAL JUNCTURES:\n"
        for juncture in timing_critical[:5]:  # Top 5
            report += f"- {juncture.description}\n"
            report += f"  Timing Sensitivity: {juncture.timing_sensitivity:.2f}\n"
            if juncture.temporal_window:
                start, end = juncture.temporal_window
                report += f"  Critical Window: {start} to {end}\n"
            report += "\n"
        
        return report
    
    def analyze_juncture_distribution(self, junctures: List[CriticalJuncture]) -> JunctureAnalysisResult:
        """Analyze the overall distribution and characteristics of junctures"""
        
        # Count by type
        junctures_by_type = {}
        for juncture in junctures:
            junctures_by_type[juncture.juncture_type] = junctures_by_type.get(juncture.juncture_type, 0) + 1
        
        # Identify high impact and timing critical
        high_impact = [j for j in junctures if j.counterfactual_impact >= 0.7]
        timing_critical = [j for j in junctures if j.timing_sensitivity >= 0.7]
        
        # Temporal distribution
        temporal_distribution = {}
        for juncture in junctures:
            if juncture.timestamp:
                year = juncture.timestamp.year
                temporal_distribution[str(year)] = temporal_distribution.get(str(year), 0) + 1
        
        # Overall timing sensitivity
        overall_timing_sensitivity = 0.0
        if junctures:
            overall_timing_sensitivity = sum(j.timing_sensitivity for j in junctures) / len(junctures)
        
        return JunctureAnalysisResult(
            total_junctures=len(junctures),
            junctures_by_type=junctures_by_type,
            high_impact_junctures=high_impact,
            timing_critical_junctures=timing_critical,
            temporal_distribution=temporal_distribution,
            overall_timing_sensitivity=overall_timing_sensitivity
        )

def test_critical_juncture_analyzer():
    """Test function for critical juncture analyzer"""
    from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge
    from core.temporal_extraction import TemporalRelation, TemporalType
    
    # Create test temporal graph with decision points
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
        node_id="stimulus_package",
        timestamp=datetime(2020, 3, 15),
        node_type="Event",
        attr_props={"description": "Large stimulus package implementation"}
    )
    
    outcome2 = TemporalNode(
        node_id="austerity_measures",
        timestamp=datetime(2020, 3, 15),
        node_type="Event",
        attr_props={"description": "Austerity measures implementation"}
    )
    
    tg.add_temporal_node(decision_node)
    tg.add_temporal_node(outcome1)
    tg.add_temporal_node(outcome2)
    
    # Add edges showing decision paths
    edge1 = TemporalEdge(
        source="policy_decision",
        target="stimulus_package",
        temporal_relation=TemporalRelation.BEFORE,
        edge_type="causes"
    )
    
    edge2 = TemporalEdge(
        source="policy_decision", 
        target="austerity_measures",
        temporal_relation=TemporalRelation.BEFORE,
        edge_type="causes"
    )
    
    tg.add_temporal_edge(edge1)
    tg.add_temporal_edge(edge2)
    
    # Test juncture analysis
    analyzer = CriticalJunctureAnalyzer()
    junctures = analyzer.identify_junctures(tg)
    
    print("Critical Juncture Analysis:")
    print(f"Junctures found: {len(junctures)}")
    
    for juncture in junctures:
        print(f"\nJuncture: {juncture.description}")
        print(f"Type: {juncture.juncture_type.value}")
        print(f"Timing Sensitivity: {juncture.timing_sensitivity:.2f}")
        print(f"Counterfactual Impact: {juncture.counterfactual_impact:.2f}")
        print(f"Alternatives: {len(juncture.alternative_pathways)}")
    
    # Generate report
    report = analyzer.generate_juncture_report(junctures)
    print("\n" + report)

if __name__ == "__main__":
    test_critical_juncture_analyzer()
"""
Temporal Process Tracing - Duration Analysis Module

Analyzes timing patterns, process durations, and temporal performance
metrics in causal processes for process tracing analysis.

Author: Claude Code Implementation  
Date: August 2025
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge

class ProcessSpeed(Enum):
    INSTANTANEOUS = "instantaneous"  # < 1 hour
    RAPID = "rapid"                 # 1 hour - 1 day
    FAST = "fast"                   # 1 day - 1 week
    MODERATE = "moderate"           # 1 week - 1 month
    SLOW = "slow"                   # 1 month - 1 year
    VERY_SLOW = "very_slow"         # > 1 year

class TemporalPhase(Enum):
    INITIATION = "initiation"       # Process start
    DEVELOPMENT = "development"     # Process progression
    CLIMAX = "climax"              # Peak or critical point
    RESOLUTION = "resolution"       # Process conclusion

@dataclass
class ProcessDuration:
    """Represents duration analysis for a single process or node"""
    process_id: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    process_speed: ProcessSpeed
    temporal_phase: TemporalPhase
    duration_confidence: float
    
    # Comparative metrics
    relative_duration: float  # Compared to other processes
    duration_percentile: float  # Percentile in overall distribution
    
    # Performance metrics
    efficiency_score: float   # How quickly process achieved outcome
    timing_optimality: float  # How optimal the timing was

@dataclass
class PathwayDuration:
    """Represents duration analysis for a causal pathway"""
    pathway_id: str
    pathway_nodes: List[str]
    total_duration: Optional[timedelta]
    average_step_duration: Optional[timedelta]
    longest_step: Optional[Tuple[str, str, timedelta]]
    shortest_step: Optional[Tuple[str, str, timedelta]]
    pathway_speed: ProcessSpeed
    bottlenecks: List[str]  # Nodes that slow down the pathway
    critical_path: bool     # Whether this is a critical path

@dataclass
class TemporalPattern:
    """Represents a recurring temporal pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    average_duration: timedelta
    duration_variance: float
    examples: List[str]
    confidence: float

@dataclass
class DurationAnalysisResult:
    """Complete duration analysis results"""
    total_processes: int
    process_durations: List[ProcessDuration]
    pathway_durations: List[PathwayDuration]
    temporal_patterns: List[TemporalPattern]
    
    # Summary statistics
    average_process_duration: Optional[timedelta]
    median_process_duration: Optional[timedelta]
    duration_distribution: Dict[ProcessSpeed, int]
    
    # Performance metrics
    fast_processes: List[ProcessDuration]
    slow_processes: List[ProcessDuration]
    efficient_processes: List[ProcessDuration]
    bottleneck_nodes: List[str]
    
    # Temporal insights
    timing_insights: List[str]
    performance_recommendations: List[str]

class DurationAnalyzer:
    """
    Analyzes duration patterns and temporal performance in process tracing graphs.
    """
    
    def __init__(self):
        self.speed_thresholds = self._initialize_speed_thresholds()
        self.performance_benchmarks = self._initialize_performance_benchmarks()
    
    def _initialize_speed_thresholds(self) -> Dict[ProcessSpeed, Tuple[timedelta, timedelta]]:
        """Initialize thresholds for process speed classification"""
        return {
            ProcessSpeed.INSTANTANEOUS: (timedelta(0), timedelta(hours=1)),
            ProcessSpeed.RAPID: (timedelta(hours=1), timedelta(days=1)),
            ProcessSpeed.FAST: (timedelta(days=1), timedelta(weeks=1)),
            ProcessSpeed.MODERATE: (timedelta(weeks=1), timedelta(days=30)),
            ProcessSpeed.SLOW: (timedelta(days=30), timedelta(days=365)),
            ProcessSpeed.VERY_SLOW: (timedelta(days=365), timedelta(days=36500))  # 100 years max
        }
    
    def _initialize_performance_benchmarks(self) -> Dict[str, Any]:
        """Initialize performance benchmarks for efficiency assessment"""
        return {
            'optimal_decision_time': timedelta(days=7),      # 1 week for decisions
            'optimal_implementation_time': timedelta(days=30), # 1 month for implementation
            'crisis_response_time': timedelta(hours=24),      # 24 hours for crisis response
            'policy_development_time': timedelta(days=90),    # 3 months for policy development
        }
    
    def analyze_durations(self, temporal_graph: TemporalGraph) -> DurationAnalysisResult:
        """
        Perform comprehensive duration analysis of the temporal graph.
        """
        # Analyze individual process durations
        process_durations = self._analyze_process_durations(temporal_graph)
        
        # Analyze pathway durations
        pathway_durations = self._analyze_pathway_durations(temporal_graph)
        
        # Identify temporal patterns
        temporal_patterns = self._identify_temporal_patterns(temporal_graph, process_durations)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(process_durations)
        
        # Identify performance characteristics
        performance_analysis = self._analyze_performance_characteristics(
            process_durations, pathway_durations
        )
        
        # Generate insights and recommendations
        insights = self._generate_temporal_insights(process_durations, pathway_durations)
        recommendations = self._generate_performance_recommendations(
            process_durations, pathway_durations
        )
        
        return DurationAnalysisResult(
            total_processes=len(process_durations),
            process_durations=process_durations,
            pathway_durations=pathway_durations,
            temporal_patterns=temporal_patterns,
            **summary_stats,
            **performance_analysis,
            timing_insights=insights,
            performance_recommendations=recommendations
        )
    
    def _analyze_process_durations(self, temporal_graph: TemporalGraph) -> List[ProcessDuration]:
        """Analyze duration characteristics of individual processes"""
        durations = []
        
        for node_id, node in temporal_graph.temporal_nodes.items():
            duration_analysis = self._analyze_single_process_duration(
                temporal_graph, node_id, node
            )
            if duration_analysis:
                durations.append(duration_analysis)
        
        # Calculate relative metrics after all durations are analyzed
        self._calculate_relative_duration_metrics(durations)
        
        return durations
    
    def _analyze_single_process_duration(self, temporal_graph: TemporalGraph, 
                                       node_id: str, node: TemporalNode) -> Optional[ProcessDuration]:
        """Analyze duration characteristics of a single process"""
        
        # Determine process duration
        duration = None
        start_time = None
        end_time = None
        
        if node.duration:
            duration = node.duration
            start_time = node.timestamp
            end_time = start_time + duration if start_time else None
        elif node.timestamp:
            # Try to infer duration from graph structure
            duration, start_time, end_time = self._infer_process_duration(
                temporal_graph, node_id, node
            )
        
        if not duration:
            return None
        
        # Classify process speed
        process_speed = self._classify_process_speed(duration)
        
        # Determine temporal phase
        temporal_phase = self._determine_temporal_phase(temporal_graph, node_id)
        
        # Calculate confidence in duration measurement
        duration_confidence = self._calculate_duration_confidence(node, duration)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(temporal_graph, node_id, duration)
        
        # Calculate timing optimality
        timing_optimality = self._calculate_timing_optimality(temporal_graph, node_id, node)
        
        return ProcessDuration(
            process_id=node_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            process_speed=process_speed,
            temporal_phase=temporal_phase,
            duration_confidence=duration_confidence,
            relative_duration=0.0,  # Will be calculated later
            duration_percentile=0.0,  # Will be calculated later
            efficiency_score=efficiency_score,
            timing_optimality=timing_optimality
        )
    
    def _infer_process_duration(self, temporal_graph: TemporalGraph, 
                              node_id: str, node: TemporalNode) -> Tuple[Optional[timedelta], Optional[datetime], Optional[datetime]]:
        """Infer process duration from graph structure and timing"""
        graph = temporal_graph.to_networkx()
        
        # Look for successor nodes to estimate end time
        successors = list(graph.successors(node_id))
        if successors and node.timestamp:
            successor_times = []
            for successor in successors:
                successor_node = temporal_graph.temporal_nodes.get(successor)
                if successor_node and successor_node.timestamp:
                    successor_times.append(successor_node.timestamp)
            
            if successor_times:
                earliest_successor = min(successor_times)
                duration = earliest_successor - node.timestamp
                return duration, node.timestamp, earliest_successor
        
        # If no successors, look for process-specific duration patterns
        node_desc = node.attr_props.get('description', '').lower()
        duration = self._estimate_duration_from_description(node_desc)
        
        if duration and node.timestamp:
            return duration, node.timestamp, node.timestamp + duration
        
        return None, None, None
    
    def _estimate_duration_from_description(self, description: str) -> Optional[timedelta]:
        """Estimate duration based on process description keywords"""
        duration_keywords = {
            'meeting': timedelta(hours=2),
            'conference': timedelta(days=3),
            'negotiation': timedelta(weeks=2),
            'policy': timedelta(days=90),
            'crisis': timedelta(days=7),
            'decision': timedelta(days=14),
            'implementation': timedelta(days=60),
            'announcement': timedelta(hours=1),
            'vote': timedelta(hours=4),
            'debate': timedelta(hours=8),
        }
        
        for keyword, duration in duration_keywords.items():
            if keyword in description:
                return duration
        
        return None
    
    def _classify_process_speed(self, duration: timedelta) -> ProcessSpeed:
        """Classify process speed based on duration"""
        for speed, (min_duration, max_duration) in self.speed_thresholds.items():
            if min_duration <= duration < max_duration:
                return speed
        
        return ProcessSpeed.VERY_SLOW  # Default for very long processes
    
    def _determine_temporal_phase(self, temporal_graph: TemporalGraph, node_id: str) -> TemporalPhase:
        """Determine what temporal phase this process represents"""
        graph = temporal_graph.to_networkx()
        
        # Analyze graph position
        predecessors = list(graph.predecessors(node_id))
        successors = list(graph.successors(node_id))
        
        if not predecessors and successors:
            return TemporalPhase.INITIATION
        elif predecessors and successors:
            # Check if this is a convergence point (climax)
            if len(predecessors) > 1:
                return TemporalPhase.CLIMAX
            else:
                return TemporalPhase.DEVELOPMENT
        elif predecessors and not successors:
            return TemporalPhase.RESOLUTION
        else:
            return TemporalPhase.DEVELOPMENT  # Default
    
    def _calculate_duration_confidence(self, node: TemporalNode, duration: timedelta) -> float:
        """Calculate confidence in duration measurement"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if duration is explicitly provided
        if node.duration:
            confidence += 0.3
        
        # Higher confidence if timestamp is precise
        if node.timestamp and node.temporal_uncertainty < 0.2:
            confidence += 0.2
        
        # Lower confidence for very uncertain temporal data
        if node.temporal_uncertainty > 0.5:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_efficiency_score(self, temporal_graph: TemporalGraph, 
                                  node_id: str, duration: timedelta) -> float:
        """Calculate efficiency score for a process"""
        # Base efficiency score
        efficiency = 0.5
        
        # Compare to benchmarks
        node = temporal_graph.temporal_nodes.get(node_id)
        if node:
            node_desc = node.attr_props.get('description', '').lower()
            
            # Check against relevant benchmarks
            if 'decision' in node_desc:
                benchmark = self.performance_benchmarks['optimal_decision_time']
                if duration <= benchmark:
                    efficiency += 0.3
                elif duration <= benchmark * 2:
                    efficiency += 0.1
                else:
                    efficiency -= 0.2
            
            elif 'crisis' in node_desc or 'emergency' in node_desc:
                benchmark = self.performance_benchmarks['crisis_response_time']
                if duration <= benchmark:
                    efficiency += 0.4
                elif duration <= benchmark * 3:
                    efficiency += 0.1
                else:
                    efficiency -= 0.3
            
            elif 'policy' in node_desc:
                benchmark = self.performance_benchmarks['policy_development_time']
                if duration <= benchmark:
                    efficiency += 0.2
                elif duration <= benchmark * 1.5:
                    efficiency += 0.1
                else:
                    efficiency -= 0.1
        
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_timing_optimality(self, temporal_graph: TemporalGraph, 
                                   node_id: str, node: TemporalNode) -> float:
        """Calculate how optimal the timing was for this process"""
        optimality = 0.5  # Base optimality
        
        # Check for timing pressure indicators
        node_desc = node.attr_props.get('description', '').lower()
        
        # Positive timing indicators
        if any(keyword in node_desc for keyword in ['timely', 'opportune', 'optimal']):
            optimality += 0.3
        
        # Negative timing indicators
        if any(keyword in node_desc for keyword in ['delayed', 'late', 'rushed', 'premature']):
            optimality -= 0.3
        
        # Check temporal context
        if node.temporal_uncertainty < 0.2:  # Low uncertainty suggests good timing
            optimality += 0.1
        
        return max(0.0, min(1.0, optimality))
    
    def _calculate_relative_duration_metrics(self, durations: List[ProcessDuration]):
        """Calculate relative duration metrics across all processes"""
        if not durations:
            return
        
        # Get all duration values
        duration_values = [d.duration.total_seconds() for d in durations if d.duration]
        
        if not duration_values:
            return
        
        # Calculate statistics
        mean_duration = statistics.mean(duration_values)
        duration_values_sorted = sorted(duration_values)
        
        # Update relative metrics
        for duration in durations:
            if duration.duration:
                seconds = duration.duration.total_seconds()
                
                # Relative duration (compared to mean)
                duration.relative_duration = seconds / mean_duration if mean_duration > 0 else 1.0
                
                # Duration percentile
                percentile = (duration_values_sorted.index(seconds) + 1) / len(duration_values_sorted)
                duration.duration_percentile = percentile
    
    def _analyze_pathway_durations(self, temporal_graph: TemporalGraph) -> List[PathwayDuration]:
        """Analyze duration characteristics of causal pathways"""
        pathway_durations = []
        
        # Find all paths in the graph
        graph = temporal_graph.to_networkx()
        
        # Get all simple paths up to reasonable length
        all_paths = []
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        # Issue #18 Fix: Add max paths limit to prevent hangs
                        import itertools
                        paths_iterator = nx.all_simple_paths(graph, source, target, cutoff=8)
                        paths = list(itertools.islice(paths_iterator, 100))
                        all_paths.extend(paths)
                    except:
                        continue
        
        # Analyze each pathway
        for i, path in enumerate(all_paths):
            if len(path) > 1:  # Only analyze paths with multiple nodes
                pathway_analysis = self._analyze_single_pathway_duration(
                    temporal_graph, f"pathway_{i}", path
                )
                if pathway_analysis:
                    pathway_durations.append(pathway_analysis)
        
        return pathway_durations
    
    def _analyze_single_pathway_duration(self, temporal_graph: TemporalGraph, 
                                       pathway_id: str, path: List[str]) -> Optional[PathwayDuration]:
        """Analyze duration characteristics of a single pathway"""
        
        # Calculate pathway timing
        pathway_times = []
        step_durations = []
        
        for i, node_id in enumerate(path):
            node = temporal_graph.temporal_nodes.get(node_id)
            if node and node.timestamp:
                pathway_times.append((node.timestamp, node_id))
                
                # Calculate step duration to next node
                if i < len(path) - 1:
                    next_node = temporal_graph.temporal_nodes.get(path[i + 1])
                    if next_node and next_node.timestamp:
                        step_duration = next_node.timestamp - node.timestamp
                        step_durations.append((node_id, path[i + 1], step_duration))
        
        if len(pathway_times) < 2:
            return None
        
        # Sort by time
        pathway_times.sort()
        
        # Calculate total pathway duration
        total_duration = pathway_times[-1][0] - pathway_times[0][0]
        
        # Calculate average step duration
        average_step_duration = None
        if step_durations:
            total_step_time = sum(duration for _, _, duration in step_durations)
            average_step_duration = total_step_time / len(step_durations)
        
        # Find longest and shortest steps
        longest_step = None
        shortest_step = None
        if step_durations:
            longest_step = max(step_durations, key=lambda x: x[2])
            shortest_step = min(step_durations, key=lambda x: x[2])
        
        # Classify pathway speed
        pathway_speed = self._classify_process_speed(total_duration)
        
        # Identify bottlenecks (longest steps)
        bottlenecks = []
        if step_durations and average_step_duration:
            for source, target, duration in step_durations:
                if duration > average_step_duration * 1.5:  # 50% longer than average
                    bottlenecks.append(target)
        
        # Determine if this is a critical path (simplified heuristic)
        critical_path = len(path) >= 5 and total_duration > timedelta(days=30)
        
        return PathwayDuration(
            pathway_id=pathway_id,
            pathway_nodes=path,
            total_duration=total_duration,
            average_step_duration=average_step_duration,
            longest_step=longest_step,
            shortest_step=shortest_step,
            pathway_speed=pathway_speed,
            bottlenecks=bottlenecks,
            critical_path=critical_path
        )
    
    def _identify_temporal_patterns(self, temporal_graph: TemporalGraph, 
                                  durations: List[ProcessDuration]) -> List[TemporalPattern]:
        """Identify recurring temporal patterns in the processes"""
        patterns = []
        
        # Group processes by speed
        speed_groups = {}
        for duration in durations:
            speed = duration.process_speed
            if speed not in speed_groups:
                speed_groups[speed] = []
            speed_groups[speed].append(duration)
        
        # Create patterns for common speeds
        for speed, group in speed_groups.items():
            if len(group) >= 3:  # Minimum 3 instances for a pattern
                durations_list = [d.duration for d in group if d.duration]
                if durations_list:
                    avg_duration = sum(durations_list, timedelta()) / len(durations_list)
                    
                    # Calculate variance
                    variance = sum((d - avg_duration).total_seconds() ** 2 for d in durations_list) / len(durations_list)
                    
                    pattern = TemporalPattern(
                        pattern_id=f"speed_pattern_{speed.value}",
                        pattern_type="speed_consistency",
                        description=f"Processes with {speed.value} speed",
                        frequency=len(group),
                        average_duration=avg_duration,
                        duration_variance=variance,
                        examples=[d.process_id for d in group[:3]],
                        confidence=0.8 if len(group) >= 5 else 0.6
                    )
                    patterns.append(pattern)
        
        # Identify phase-based patterns
        phase_groups = {}
        for duration in durations:
            phase = duration.temporal_phase
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(duration)
        
        for phase, group in phase_groups.items():
            if len(group) >= 3:
                durations_list = [d.duration for d in group if d.duration]
                if durations_list:
                    avg_duration = sum(durations_list, timedelta()) / len(durations_list)
                    
                    pattern = TemporalPattern(
                        pattern_id=f"phase_pattern_{phase.value}",
                        pattern_type="phase_consistency",
                        description=f"Processes in {phase.value} phase",
                        frequency=len(group),
                        average_duration=avg_duration,
                        duration_variance=0.0,  # Simplified
                        examples=[d.process_id for d in group[:3]],
                        confidence=0.7
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_summary_statistics(self, durations: List[ProcessDuration]) -> Dict[str, Any]:
        """Calculate summary statistics for process durations"""
        if not durations:
            return {
                'average_process_duration': None,
                'median_process_duration': None,
                'duration_distribution': {}
            }
        
        # Filter out None durations
        valid_durations = [d.duration for d in durations if d.duration]
        
        if not valid_durations:
            return {
                'average_process_duration': None,
                'median_process_duration': None,
                'duration_distribution': {}
            }
        
        # Calculate average
        total_seconds = sum(d.total_seconds() for d in valid_durations)
        average_duration = timedelta(seconds=total_seconds / len(valid_durations))
        
        # Calculate median
        sorted_durations = sorted(valid_durations)
        median_duration = sorted_durations[len(sorted_durations) // 2]
        
        # Calculate distribution by speed
        distribution = {}
        for duration in durations:
            speed = duration.process_speed
            distribution[speed] = distribution.get(speed, 0) + 1
        
        return {
            'average_process_duration': average_duration,
            'median_process_duration': median_duration,
            'duration_distribution': distribution
        }
    
    def _analyze_performance_characteristics(self, process_durations: List[ProcessDuration], 
                                           pathway_durations: List[PathwayDuration]) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        
        # Fast processes (top 25% by efficiency)
        sorted_by_efficiency = sorted(process_durations, key=lambda x: x.efficiency_score, reverse=True)
        fast_processes = sorted_by_efficiency[:len(sorted_by_efficiency) // 4]
        
        # Slow processes (bottom 25% by efficiency)
        slow_processes = sorted_by_efficiency[-len(sorted_by_efficiency) // 4:]
        
        # Efficient processes (efficiency > 0.7)
        efficient_processes = [d for d in process_durations if d.efficiency_score > 0.7]
        
        # Identify bottleneck nodes from pathways
        bottleneck_nodes = []
        for pathway in pathway_durations:
            bottleneck_nodes.extend(pathway.bottlenecks)
        
        # Count frequency of bottlenecks
        bottleneck_frequency = {}
        for node in bottleneck_nodes:
            bottleneck_frequency[node] = bottleneck_frequency.get(node, 0) + 1
        
        # Get most frequent bottlenecks
        frequent_bottlenecks = sorted(bottleneck_frequency.items(), key=lambda x: x[1], reverse=True)
        top_bottlenecks = [node for node, count in frequent_bottlenecks[:5]]
        
        return {
            'fast_processes': fast_processes,
            'slow_processes': slow_processes,
            'efficient_processes': efficient_processes,
            'bottleneck_nodes': top_bottlenecks
        }
    
    def _generate_temporal_insights(self, process_durations: List[ProcessDuration], 
                                  pathway_durations: List[PathwayDuration]) -> List[str]:
        """Generate insights about temporal patterns"""
        insights = []
        
        if not process_durations:
            return ["No temporal data available for analysis."]
        
        # Overall timing insights
        speed_distribution = {}
        for duration in process_durations:
            speed = duration.process_speed
            speed_distribution[speed] = speed_distribution.get(speed, 0) + 1
        
        most_common_speed = max(speed_distribution, key=speed_distribution.get)
        insights.append(f"Most processes operate at {most_common_speed.value} speed ({speed_distribution[most_common_speed]} processes)")
        
        # Efficiency insights
        efficient_count = len([d for d in process_durations if d.efficiency_score > 0.7])
        total_count = len(process_durations)
        efficiency_rate = efficient_count / total_count
        
        if efficiency_rate > 0.8:
            insights.append(f"High process efficiency: {efficiency_rate:.1%} of processes are highly efficient")
        elif efficiency_rate < 0.3:
            insights.append(f"Low process efficiency: Only {efficiency_rate:.1%} of processes are highly efficient")
        
        # Pathway insights
        if pathway_durations:
            critical_paths = [p for p in pathway_durations if p.critical_path]
            if critical_paths:
                insights.append(f"Identified {len(critical_paths)} critical pathways requiring extended timeframes")
            
            # Bottleneck insights
            all_bottlenecks = []
            for pathway in pathway_durations:
                all_bottlenecks.extend(pathway.bottlenecks)
            
            if all_bottlenecks:
                bottleneck_frequency = {}
                for node in all_bottlenecks:
                    bottleneck_frequency[node] = bottleneck_frequency.get(node, 0) + 1
                
                most_frequent_bottleneck = max(bottleneck_frequency, key=bottleneck_frequency.get)
                frequency = bottleneck_frequency[most_frequent_bottleneck]
                insights.append(f"'{most_frequent_bottleneck}' is the most frequent bottleneck (appears in {frequency} pathways)")
        
        # Timing optimality insights
        optimal_count = len([d for d in process_durations if d.timing_optimality > 0.7])
        optimality_rate = optimal_count / total_count
        
        if optimality_rate > 0.8:
            insights.append(f"Excellent timing optimization: {optimality_rate:.1%} of processes have optimal timing")
        elif optimality_rate < 0.3:
            insights.append(f"Timing improvement needed: Only {optimality_rate:.1%} of processes have optimal timing")
        
        return insights
    
    def _generate_performance_recommendations(self, process_durations: List[ProcessDuration], 
                                            pathway_durations: List[PathwayDuration]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if not process_durations:
            return ["No data available for performance recommendations."]
        
        # Efficiency recommendations
        inefficient_processes = [d for d in process_durations if d.efficiency_score < 0.4]
        if inefficient_processes:
            recommendations.append(f"Improve efficiency of {len(inefficient_processes)} underperforming processes")
            
            # Specific recommendations for slow processes
            very_slow_processes = [d for d in inefficient_processes if d.process_speed == ProcessSpeed.VERY_SLOW]
            if very_slow_processes:
                recommendations.append("Consider breaking down very slow processes into smaller, manageable phases")
        
        # Bottleneck recommendations
        if pathway_durations:
            all_bottlenecks = []
            for pathway in pathway_durations:
                all_bottlenecks.extend(pathway.bottlenecks)
            
            if all_bottlenecks:
                unique_bottlenecks = set(all_bottlenecks)
                recommendations.append(f"Address {len(unique_bottlenecks)} identified bottleneck nodes to improve pathway flow")
        
        # Timing optimization recommendations
        poor_timing_processes = [d for d in process_durations if d.timing_optimality < 0.4]
        if poor_timing_processes:
            recommendations.append(f"Improve timing optimization for {len(poor_timing_processes)} processes with suboptimal timing")
        
        # Speed consistency recommendations
        speed_distribution = {}
        for duration in process_durations:
            speed = duration.process_speed
            speed_distribution[speed] = speed_distribution.get(speed, 0) + 1
        
        if len(speed_distribution) > 4:  # High variability in speeds
            recommendations.append("Standardize process speeds to reduce temporal variability and improve predictability")
        
        # Phase-specific recommendations
        phase_distribution = {}
        for duration in process_durations:
            phase = duration.temporal_phase
            phase_distribution[phase] = phase_distribution.get(phase, 0) + 1
        
        if phase_distribution.get(TemporalPhase.INITIATION, 0) > phase_distribution.get(TemporalPhase.RESOLUTION, 0):
            recommendations.append("Focus on completing initiated processes to improve resolution rates")
        
        return recommendations

def test_duration_analyzer():
    """Test function for duration analyzer"""
    from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge
    from core.temporal_extraction import TemporalRelation, TemporalType
    
    # Create test temporal graph
    tg = TemporalGraph()
    
    # Add nodes with different durations
    fast_process = TemporalNode(
        node_id="fast_decision",
        timestamp=datetime(2020, 1, 1),
        duration=timedelta(hours=6),
        node_type="Event",
        attr_props={"description": "Rapid crisis decision"}
    )
    
    slow_process = TemporalNode(
        node_id="policy_development",
        timestamp=datetime(2020, 1, 15),
        duration=timedelta(days=120),
        node_type="Event",
        attr_props={"description": "Complex policy development process"}
    )
    
    moderate_process = TemporalNode(
        node_id="implementation",
        timestamp=datetime(2020, 6, 1),
        duration=timedelta(days=30),
        node_type="Event",
        attr_props={"description": "Policy implementation phase"}
    )
    
    tg.add_temporal_node(fast_process)
    tg.add_temporal_node(slow_process)
    tg.add_temporal_node(moderate_process)
    
    # Add edges to create pathways
    edge1 = TemporalEdge(
        source="fast_decision",
        target="policy_development",
        temporal_relation=TemporalRelation.BEFORE,
        edge_type="causes"
    )
    
    edge2 = TemporalEdge(
        source="policy_development",
        target="implementation",
        temporal_relation=TemporalRelation.BEFORE,
        edge_type="causes"
    )
    
    tg.add_temporal_edge(edge1)
    tg.add_temporal_edge(edge2)
    
    # Test duration analysis
    analyzer = DurationAnalyzer()
    result = analyzer.analyze_durations(tg)
    
    print("Duration Analysis Results:")
    print(f"Total Processes: {result.total_processes}")
    print(f"Average Duration: {result.average_process_duration}")
    print(f"Median Duration: {result.median_process_duration}")
    
    print("\nProcess Speed Distribution:")
    for speed, count in result.duration_distribution.items():
        print(f"  {speed.value}: {count}")
    
    print(f"\nFast Processes: {len(result.fast_processes)}")
    print(f"Slow Processes: {len(result.slow_processes)}")
    print(f"Efficient Processes: {len(result.efficient_processes)}")
    
    print("\nTiming Insights:")
    for insight in result.timing_insights:
        print(f"  - {insight}")
    
    print("\nPerformance Recommendations:")
    for recommendation in result.performance_recommendations:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    test_duration_analyzer()
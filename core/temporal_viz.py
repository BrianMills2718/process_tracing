"""
Temporal Process Tracing - Temporal Visualization Module

Creates interactive temporal visualizations including timelines, temporal networks,
and critical juncture highlighting for process tracing analysis.

Author: Claude Code Implementation  
Date: August 2025
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge
# Critical junctures and duration analysis moved to archive - using fallback
class CriticalJuncture:
    pass

class JunctureType:
    pass

class ProcessDuration:
    pass

class PathwayDuration:
    pass

class ProcessSpeed:
    pass

@dataclass
class TimelineEvent:
    """Represents an event on the timeline"""
    id: str
    title: str
    description: str
    start: datetime
    end: Optional[datetime]
    type: str  # event, juncture, process
    category: str  # node_type from graph
    className: str  # CSS class for styling
    content: str  # HTML content for tooltip

@dataclass
class NetworkNodeViz:
    """Represents a node in temporal network visualization"""
    id: str
    label: str
    x: Optional[float]
    y: Optional[float]
    size: float
    color: str
    shape: str
    title: str  # Tooltip
    timestamp: Optional[str]
    duration: Optional[str]
    temporal_phase: Optional[str]

@dataclass
class NetworkEdgeViz:
    """Represents an edge in temporal network visualization"""
    id: str
    from_node: str
    to_node: str
    label: str
    color: str
    width: float
    arrows: str
    title: str  # Tooltip
    temporal_relation: str

@dataclass
class TemporalVisualizationData:
    """Complete data structure for temporal visualizations"""
    timeline_events: List[TimelineEvent]
    network_nodes: List[NetworkNodeViz]
    network_edges: List[NetworkEdgeViz]
    critical_junctures: List[Dict[str, Any]]
    timeline_config: Dict[str, Any]
    network_config: Dict[str, Any]

class TemporalVisualizer:
    """
    Creates interactive temporal visualizations for process tracing analysis.
    Generates data structures that can be consumed by web-based visualization libraries.
    """
    
    def __init__(self):
        self.color_schemes = self._initialize_color_schemes()
        self.timeline_templates = self._initialize_timeline_templates()
        self.network_layouts = self._initialize_network_layouts()
    
    def _initialize_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Initialize color schemes for different visualization elements"""
        return {
            'node_types': {
                'Event': '#3498db',           # Blue
                'Evidence': '#2ecc71',        # Green
                'Hypothesis': '#f39c12',      # Orange
                'Causal_Mechanism': '#9b59b6', # Purple
                'Condition': '#e74c3c',       # Red
                'Outcome': '#1abc9c'          # Teal
            },
            'juncture_types': {
                'decision_point': '#e74c3c',     # Red
                'branching_point': '#f39c12',    # Orange
                'convergence_point': '#3498db',  # Blue
                'timing_critical': '#e91e63',    # Pink
                'threshold_crossing': '#9c27b0'  # Purple
            },
            'process_speeds': {
                'instantaneous': '#ff5722',   # Deep Orange
                'rapid': '#ff9800',          # Orange
                'fast': '#ffc107',           # Amber
                'moderate': '#4caf50',       # Green
                'slow': '#2196f3',           # Blue
                'very_slow': '#673ab7'       # Deep Purple
            },
            'temporal_phases': {
                'initiation': '#4caf50',     # Green
                'development': '#2196f3',    # Blue
                'climax': '#ff5722',         # Deep Orange
                'resolution': '#9c27b0'      # Purple
            }
        }
    
    def _initialize_timeline_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for timeline visualization"""
        return {
            'default': {
                'orientation': 'horizontal',
                'showMajorLabels': True,
                'showMinorLabels': True,
                'stack': True,
                'height': '400px',
                'min': None,
                'max': None,
                'zoomMin': 1000 * 60 * 60 * 24,  # 1 day in milliseconds
                'zoomMax': 1000 * 60 * 60 * 24 * 365 * 10  # 10 years
            },
            'compact': {
                'orientation': 'horizontal',
                'showMajorLabels': True,
                'showMinorLabels': False,
                'stack': False,
                'height': '200px'
            }
        }
    
    def _initialize_network_layouts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize network layout configurations"""
        return {
            'temporal_hierarchical': {
                'layout': {
                    'hierarchical': {
                        'enabled': True,
                        'direction': 'LR',  # Left to Right
                        'sortMethod': 'directed',
                        'levelSeparation': 150,
                        'nodeSpacing': 100
                    }
                },
                'physics': {
                    'enabled': False
                }
            },
            'temporal_force': {
                'layout': {
                    'improvedLayout': True
                },
                'physics': {
                    'enabled': True,
                    'stabilization': {'iterations': 100},
                    'barnesHut': {
                        'gravitationalConstant': -2000,
                        'centralGravity': 0.3,
                        'springLength': 95,
                        'springConstant': 0.04
                    }
                }
            }
        }
    
    def create_temporal_visualization(self, temporal_graph: TemporalGraph, 
                                    critical_junctures: List[CriticalJuncture] = None,
                                    process_durations: List[ProcessDuration] = None,
                                    pathway_durations: List[PathwayDuration] = None) -> TemporalVisualizationData:
        """
        Create comprehensive temporal visualization data.
        """
        # Create timeline events
        timeline_events = self._create_timeline_events(
            temporal_graph, critical_junctures, process_durations
        )
        
        # Create network visualization data
        network_nodes, network_edges = self._create_network_visualization(
            temporal_graph, critical_junctures, process_durations
        )
        
        # Process critical junctures for visualization
        juncture_viz_data = self._process_junctures_for_viz(critical_junctures)
        
        # Configure timeline
        timeline_config = self._configure_timeline(timeline_events)
        
        # Configure network
        network_config = self._configure_network(network_nodes, network_edges)
        
        return TemporalVisualizationData(
            timeline_events=timeline_events,
            network_nodes=network_nodes,
            network_edges=network_edges,
            critical_junctures=juncture_viz_data,
            timeline_config=timeline_config,
            network_config=network_config
        )
    
    def _create_timeline_events(self, temporal_graph: TemporalGraph,
                               critical_junctures: List[CriticalJuncture] = None,
                               process_durations: List[ProcessDuration] = None) -> List[TimelineEvent]:
        """Create timeline events from temporal graph data"""
        events = []
        
        # Add nodes as timeline events
        for node_id, node in temporal_graph.temporal_nodes.items():
            if node.timestamp:
                # Determine end time
                end_time = None
                if node.duration:
                    end_time = node.timestamp + node.duration
                
                # Get process duration info if available
                process_info = None
                if process_durations:
                    process_info = next((p for p in process_durations if p.process_id == node_id), None)
                
                # Determine styling
                node_type = node.node_type or 'Event'
                color = self.color_schemes['node_types'].get(node_type, '#95a5a6')
                
                if process_info:
                    # Use process speed for additional styling
                    speed_color = self.color_schemes['process_speeds'].get(
                        process_info.process_speed.value, color
                    )
                    className = f"timeline-event {node_type.lower()} {process_info.process_speed.value}"
                else:
                    speed_color = color
                    className = f"timeline-event {node_type.lower()}"
                
                # Create content with detailed information
                content = self._create_event_content(node, process_info)
                
                event = TimelineEvent(
                    id=node_id,
                    title=node.attr_props.get('description', node_id)[:50],
                    description=node.attr_props.get('description', ''),
                    start=node.timestamp,
                    end=end_time,
                    type='process' if node.duration else 'event',
                    category=node_type,
                    className=className,
                    content=content
                )
                events.append(event)
        
        # Add critical junctures as special timeline events
        if critical_junctures:
            for juncture in critical_junctures:
                if juncture.timestamp:
                    color = self.color_schemes['juncture_types'].get(
                        juncture.juncture_type.value, '#e74c3c'
                    )
                    
                    content = self._create_juncture_content(juncture)
                    
                    event = TimelineEvent(
                        id=f"juncture_{juncture.juncture_id}",
                        title=f"🔥 {juncture.description[:50]}",
                        description=juncture.description,
                        start=juncture.timestamp,
                        end=None,
                        type='juncture',
                        category=juncture.juncture_type.value,
                        className=f"timeline-juncture {juncture.juncture_type.value}",
                        content=content
                    )
                    events.append(event)
        
        # Sort events by timestamp
        events.sort(key=lambda e: e.start)
        
        return events
    
    def _create_network_visualization(self, temporal_graph: TemporalGraph,
                                    critical_junctures: List[CriticalJuncture] = None,
                                    process_durations: List[ProcessDuration] = None) -> Tuple[List[NetworkNodeViz], List[NetworkEdgeViz]]:
        """Create network visualization data with temporal attributes"""
        
        nodes = []
        edges = []
        
        # Create juncture lookup for enhanced node visualization
        juncture_nodes = set()
        if critical_junctures:
            for juncture in critical_junctures:
                juncture_nodes.update(juncture.key_nodes)
        
        # Create process duration lookup
        duration_lookup = {}
        if process_durations:
            duration_lookup = {p.process_id: p for p in process_durations}
        
        # Create nodes
        for node_id, node in temporal_graph.temporal_nodes.items():
            # Determine node characteristics
            node_type = node.node_type or 'Event'
            base_color = self.color_schemes['node_types'].get(node_type, '#95a5a6')
            
            # Enhance styling for critical junctures
            if node_id in juncture_nodes:
                color = '#e74c3c'  # Red for critical junctures
                shape = 'star'
                size = 25
            else:
                color = base_color
                shape = 'dot'
                size = 15
            
            # Adjust size based on duration/importance
            process_info = duration_lookup.get(node_id)
            if process_info:
                # Size based on efficiency score
                size = 10 + (process_info.efficiency_score * 20)
                
                # Color based on process speed if not a juncture
                if node_id not in juncture_nodes:
                    color = self.color_schemes['process_speeds'].get(
                        process_info.process_speed.value, base_color
                    )
            
            # Create detailed tooltip
            title = self._create_node_tooltip(node, process_info, node_id in juncture_nodes)
            
            # Format temporal information
            timestamp_str = node.timestamp.strftime('%Y-%m-%d %H:%M') if node.timestamp else 'Unknown'
            duration_str = str(node.duration) if node.duration else 'Instantaneous'
            
            # Determine temporal phase
            temporal_phase = None
            if process_info:
                temporal_phase = process_info.temporal_phase.value
            
            network_node = NetworkNodeViz(
                id=node_id,
                label=node.attr_props.get('description', node_id)[:30],
                x=None,  # Let layout algorithm decide
                y=None,
                size=size,
                color=color,
                shape=shape,
                title=title,
                timestamp=timestamp_str,
                duration=duration_str,
                temporal_phase=temporal_phase
            )
            nodes.append(network_node)
        
        # Create edges
        for edge_key, edge in temporal_graph.temporal_edges.items():
            source, target = edge_key
            
            # Determine edge styling based on temporal relationship
            if edge.temporal_relation:
                relation = edge.temporal_relation.value
                if relation == 'before':
                    color = '#2ecc71'  # Green
                    width = 2
                elif relation == 'after':
                    color = '#e74c3c'  # Red
                    width = 2
                elif relation == 'concurrent':
                    color = '#f39c12'  # Orange
                    width = 3
                else:
                    color = '#95a5a6'  # Gray
                    width = 1
            else:
                color = '#95a5a6'
                width = 1
            
            # Create tooltip for edge
            edge_title = self._create_edge_tooltip(edge)
            
            network_edge = NetworkEdgeViz(
                id=f"{source}_{target}",
                from_node=source,
                to_node=target,
                label=edge.temporal_relation.value if edge.temporal_relation else '',
                color=color,
                width=width,
                arrows='to',
                title=edge_title,
                temporal_relation=edge.temporal_relation.value if edge.temporal_relation else 'unknown'
            )
            edges.append(network_edge)
        
        return nodes, edges
    
    def _process_junctures_for_viz(self, critical_junctures: List[CriticalJuncture] = None) -> List[Dict[str, Any]]:
        """Process critical junctures for visualization"""
        if not critical_junctures:
            return []
        
        juncture_data = []
        for juncture in critical_junctures:
            data = {
                'id': juncture.juncture_id,
                'type': juncture.juncture_type.value,
                'description': juncture.description,
                'timestamp': juncture.timestamp.isoformat() if juncture.timestamp else None,
                'timing_sensitivity': juncture.timing_sensitivity,
                'counterfactual_impact': juncture.counterfactual_impact,
                'confidence': juncture.confidence,
                'key_nodes': juncture.key_nodes,
                'alternatives': [
                    {
                        'id': alt.pathway_id,
                        'description': alt.description,
                        'probability': alt.probability,
                        'outcome_difference': alt.outcome_difference,
                        'plausibility': alt.plausibility_score
                    }
                    for alt in juncture.alternative_pathways
                ],
                'color': self.color_schemes['juncture_types'].get(juncture.juncture_type.value, '#e74c3c')
            }
            juncture_data.append(data)
        
        return juncture_data
    
    def _create_event_content(self, node: TemporalNode, process_info: ProcessDuration = None) -> str:
        """Create detailed HTML content for timeline event"""
        content = f"<div class='timeline-event-content'>"
        content += f"<h4>{node.attr_props.get('description', node.node_id)}</h4>"
        content += f"<p><strong>Type:</strong> {node.node_type}</p>"
        
        if node.timestamp:
            content += f"<p><strong>Time:</strong> {node.timestamp.strftime('%Y-%m-%d %H:%M')}</p>"
        
        if node.duration:
            content += f"<p><strong>Duration:</strong> {node.duration}</p>"
        
        if process_info:
            content += f"<p><strong>Speed:</strong> {process_info.process_speed.value}</p>"
            content += f"<p><strong>Phase:</strong> {process_info.temporal_phase.value}</p>"
            content += f"<p><strong>Efficiency:</strong> {process_info.efficiency_score:.2f}</p>"
            content += f"<p><strong>Timing Optimality:</strong> {process_info.timing_optimality:.2f}</p>"
        
        if node.temporal_uncertainty > 0:
            content += f"<p><strong>Uncertainty:</strong> {node.temporal_uncertainty:.2f}</p>"
        
        content += "</div>"
        return content
    
    def _create_juncture_content(self, juncture: CriticalJuncture) -> str:
        """Create detailed HTML content for critical juncture"""
        content = f"<div class='juncture-content'>"
        content += f"<h4>🔥 Critical Juncture</h4>"
        content += f"<p><strong>Type:</strong> {juncture.juncture_type.value}</p>"
        content += f"<p><strong>Description:</strong> {juncture.description}</p>"
        content += f"<p><strong>Timing Sensitivity:</strong> {juncture.timing_sensitivity:.2f}</p>"
        content += f"<p><strong>Counterfactual Impact:</strong> {juncture.counterfactual_impact:.2f}</p>"
        content += f"<p><strong>Alternatives:</strong> {len(juncture.alternative_pathways)}</p>"
        content += f"<p><strong>Confidence:</strong> {juncture.confidence:.2f}</p>"
        content += "</div>"
        return content
    
    def _create_node_tooltip(self, node: TemporalNode, process_info: ProcessDuration = None, 
                           is_juncture: bool = False) -> str:
        """Create detailed tooltip for network node"""
        tooltip = f"<div><strong>{node.attr_props.get('description', node.node_id)}</strong><br/>"
        tooltip += f"Type: {node.node_type}<br/>"
        
        if node.timestamp:
            tooltip += f"Time: {node.timestamp.strftime('%Y-%m-%d %H:%M')}<br/>"
        
        if node.duration:
            tooltip += f"Duration: {node.duration}<br/>"
        
        if process_info:
            tooltip += f"Speed: {process_info.process_speed.value}<br/>"
            tooltip += f"Efficiency: {process_info.efficiency_score:.2f}<br/>"
        
        if is_juncture:
            tooltip += f"<em>Critical Juncture</em><br/>"
        
        tooltip += "</div>"
        return tooltip
    
    def _create_edge_tooltip(self, edge: TemporalEdge) -> str:
        """Create detailed tooltip for network edge"""
        tooltip = f"<div><strong>Temporal Relationship</strong><br/>"
        tooltip += f"Type: {edge.temporal_relation.value if edge.temporal_relation else 'Unknown'}<br/>"
        tooltip += f"Confidence: {edge.confidence:.2f}<br/>"
        
        if edge.temporal_gap:
            tooltip += f"Time Gap: {edge.temporal_gap}<br/>"
        
        if edge.evidence_text:
            tooltip += f"Evidence: {edge.evidence_text[:100]}...<br/>"
        
        tooltip += "</div>"
        return tooltip
    
    def _configure_timeline(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Configure timeline visualization settings"""
        if not events:
            return self.timeline_templates['default'].copy()
        
        # Calculate time range
        timestamps = [e.start for e in events]
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            
            # Add padding
            time_range = max_time - min_time
            padding = time_range * 0.1 if time_range.total_seconds() > 0 else timedelta(days=30)
            
            config = self.timeline_templates['default'].copy()
            config['min'] = (min_time - padding).isoformat()
            config['max'] = (max_time + padding).isoformat()
            
            return config
        
        return self.timeline_templates['default'].copy()
    
    def _configure_network(self, nodes: List[NetworkNodeViz], 
                         edges: List[NetworkEdgeViz]) -> Dict[str, Any]:
        """Configure network visualization settings"""
        # Use temporal hierarchical layout for temporal data
        config = self.network_layouts['temporal_hierarchical'].copy()
        
        # Add temporal-specific options
        config['nodes'] = {
            'borderWidth': 2,
            'font': {
                'size': 12,
                'color': '#2c3e50'
            },
            'shadow': True
        }
        
        config['edges'] = {
            'smooth': {
                'type': 'continuous'
            },
            'font': {
                'size': 10,
                'color': '#7f8c8d',
                'strokeWidth': 2,
                'strokeColor': '#ffffff'
            },
            'shadow': True
        }
        
        config['interaction'] = {
            'hover': True,
            'tooltipDelay': 200,
            'hideEdgesOnDrag': True
        }
        
        return config
    
    def generate_html_visualization(self, viz_data: TemporalVisualizationData, 
                                  output_path: str = None) -> str:
        """
        Generate complete HTML page with temporal visualizations.
        """
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Temporal Process Tracing Analysis</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- vis.js Timeline -->
    <script src="https://unpkg.com/vis-timeline@7.7.3/standalone/umd/vis-timeline-graph2d.min.js"></script>
    <link href="https://unpkg.com/vis-timeline@7.7.3/styles/vis-timeline-graph2d.min.css" rel="stylesheet">
    
    <!-- vis.js Network -->
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    
    <style>
        .timeline-container { height: 400px; border: 1px solid #ddd; margin: 20px 0; }
        .network-container { height: 600px; border: 1px solid #ddd; margin: 20px 0; }
        .vis-item.timeline-juncture { background-color: #e74c3c; border-color: #c0392b; }
        .vis-item.timeline-event { background-color: #3498db; border-color: #2980b9; }
        .vis-item.timeline-process { background-color: #2ecc71; border-color: #27ae60; }
        .juncture-panel { max-height: 400px; overflow-y: auto; }
        .juncture-item { border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; background: #f8f9fa; }
        .stats-card { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mb-4">Temporal Process Tracing Analysis</h1>
        
        <!-- Statistics Overview -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Timeline Events</h5>
                        <h3 class="text-primary">{timeline_events_count}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Network Nodes</h5>
                        <h3 class="text-success">{network_nodes_count}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Critical Junctures</h5>
                        <h3 class="text-danger">{critical_junctures_count}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Causal Connections</h5>
                        <h3 class="text-warning">{network_edges_count}</h3>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Timeline Visualization -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Temporal Timeline</h3>
                <p class="text-muted mb-0">Interactive timeline showing process events and critical junctures</p>
            </div>
            <div class="card-body">
                <div id="timeline" class="timeline-container"></div>
            </div>
        </div>
        
        <!-- Network Visualization -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Temporal Network</h3>
                <p class="text-muted mb-0">Interactive network showing causal relationships with temporal attributes</p>
            </div>
            <div class="card-body">
                <div id="network" class="network-container"></div>
            </div>
        </div>
        
        <!-- Critical Junctures Panel -->
        <div class="card mb-4">
            <div class="card-header">
                <h3>Critical Junctures Analysis</h3>
                <p class="text-muted mb-0">Key decision points and temporal branching moments</p>
            </div>
            <div class="card-body">
                <div id="junctures" class="juncture-panel">
                    {junctures_html}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Timeline Data
        const timelineData = {timeline_data};
        const timelineOptions = {timeline_options};
        
        // Network Data
        const networkNodes = new vis.DataSet({network_nodes});
        const networkEdges = new vis.DataSet({network_edges});
        const networkData = {{ nodes: networkNodes, edges: networkEdges }};
        const networkOptions = {network_options};
        
        // Critical Junctures Data
        const juncturesData = {junctures_data};
        
        // Initialize Timeline
        const timelineContainer = document.getElementById('timeline');
        const timeline = new vis.Timeline(timelineContainer, timelineData, timelineOptions);
        
        // Initialize Network
        const networkContainer = document.getElementById('network');
        const network = new vis.Network(networkContainer, networkData, networkOptions);
        
        // Timeline selection handler
        timeline.on('select', function(event) {{
            const selection = event.items;
            if (selection.length > 0) {{
                const selectedId = selection[0];
                const item = timelineData.get(selectedId);
                if (item) {{
                    console.log('Selected timeline item:', item);
                    // Could highlight corresponding network node
                }}
            }}
        }});
        
        // Network selection handler
        network.on('selectNode', function(event) {{
            const nodeIds = event.nodes;
            if (nodeIds.length > 0) {{
                const nodeId = nodeIds[0];
                console.log('Selected network node:', nodeId);
                // Could highlight corresponding timeline event
            }}
        }});
        
        console.log('Temporal visualization initialized');
        console.log('Timeline events:', timelineData.length);
        console.log('Network nodes:', networkNodes.length);
        console.log('Critical junctures:', juncturesData.length);
    </script>
</body>
</html>
"""
        
        # Convert data to vis.js format
        timeline_data_vis = self._convert_timeline_to_vis(viz_data.timeline_events)
        network_nodes_vis = self._convert_network_nodes_to_vis(viz_data.network_nodes)
        network_edges_vis = self._convert_network_edges_to_vis(viz_data.network_edges)
        
        # Generate junctures HTML
        junctures_html = self._generate_junctures_html(viz_data.critical_junctures)
        
        # Fill template
        html_content = html_template.format(
            timeline_events_count=len(viz_data.timeline_events),
            network_nodes_count=len(viz_data.network_nodes),
            critical_junctures_count=len(viz_data.critical_junctures),
            network_edges_count=len(viz_data.network_edges),
            timeline_data=json.dumps(timeline_data_vis),
            timeline_options=json.dumps(viz_data.timeline_config),
            network_nodes=json.dumps(network_nodes_vis),
            network_edges=json.dumps(network_edges_vis),
            network_options=json.dumps(viz_data.network_config),
            junctures_data=json.dumps(viz_data.critical_junctures),
            junctures_html=junctures_html
        )
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    def _convert_timeline_to_vis(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Convert timeline events to vis.js format"""
        vis_items = []
        for event in events:
            item = {
                'id': event.id,
                'content': event.title,
                'start': event.start.isoformat(),
                'type': 'range' if event.end else 'point',
                'className': event.className,
                'title': event.content
            }
            
            if event.end:
                item['end'] = event.end.isoformat()
            
            vis_items.append(item)
        
        return vis_items
    
    def _convert_network_nodes_to_vis(self, nodes: List[NetworkNodeViz]) -> List[Dict[str, Any]]:
        """Convert network nodes to vis.js format"""
        vis_nodes = []
        for node in nodes:
            vis_node = {
                'id': node.id,
                'label': node.label,
                'size': node.size,
                'color': node.color,
                'shape': node.shape,
                'title': node.title
            }
            
            if node.x is not None:
                vis_node['x'] = node.x
            if node.y is not None:
                vis_node['y'] = node.y
            
            vis_nodes.append(vis_node)
        
        return vis_nodes
    
    def _convert_network_edges_to_vis(self, edges: List[NetworkEdgeViz]) -> List[Dict[str, Any]]:
        """Convert network edges to vis.js format"""
        vis_edges = []
        for edge in edges:
            vis_edge = {
                'id': edge.id,
                'from': edge.from_node,
                'to': edge.to_node,
                'label': edge.label,
                'color': edge.color,
                'width': edge.width,
                'arrows': edge.arrows,
                'title': edge.title
            }
            vis_edges.append(vis_edge)
        
        return vis_edges
    
    def _generate_junctures_html(self, junctures: List[Dict[str, Any]]) -> str:
        """Generate HTML for critical junctures panel"""
        if not junctures:
            return "<p class='text-muted'>No critical junctures identified.</p>"
        
        html = ""
        for juncture in junctures:
            html += f"""
            <div class="juncture-item">
                <h5><span style="color: {juncture['color']}">🔥</span> {juncture['type'].replace('_', ' ').title()}</h5>
                <p><strong>Description:</strong> {juncture['description']}</p>
                <div class="row">
                    <div class="col-md-6">
                        <small><strong>Timing Sensitivity:</strong> {juncture['timing_sensitivity']:.2f}</small>
                    </div>
                    <div class="col-md-6">
                        <small><strong>Counterfactual Impact:</strong> {juncture['counterfactual_impact']:.2f}</small>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6">
                        <small><strong>Confidence:</strong> {juncture['confidence']:.2f}</small>
                    </div>
                    <div class="col-md-6">
                        <small><strong>Alternatives:</strong> {len(juncture['alternatives'])}</small>
                    </div>
                </div>
            </div>
            """
        
        return html

def test_temporal_visualizer():
    """Test function for temporal visualizer"""
    from core.temporal_graph import TemporalGraph, TemporalNode, TemporalEdge
    from core.temporal_extraction import TemporalRelation, TemporalType
    from core.critical_junctures import CriticalJuncture, JunctureType, AlternativePathway
    from core.duration_analysis import ProcessDuration, ProcessSpeed, TemporalPhase
    
    # Create test temporal graph
    tg = TemporalGraph()
    
    # Add nodes
    node1 = TemporalNode(
        node_id="policy_announcement",
        timestamp=datetime(2020, 1, 15),
        duration=timedelta(hours=2),
        node_type="Event",
        attr_props={"description": "Government policy announcement"}
    )
    
    node2 = TemporalNode(
        node_id="public_reaction",
        timestamp=datetime(2020, 1, 16),
        duration=timedelta(days=3),
        node_type="Event",
        attr_props={"description": "Public reaction and protests"}
    )
    
    tg.add_temporal_node(node1)
    tg.add_temporal_node(node2)
    
    # Add edge with LLM-based confidence assessment
    evidence_text = "Announcement triggered public reaction"
    
    try:
        from core.semantic_analysis_service import get_semantic_service
        from core.llm_required import LLMRequiredError
        
        semantic_service = get_semantic_service()
        confidence_result = semantic_service.assess_probative_value(
            evidence_description=evidence_text,
            hypothesis_description="Temporal relationship demonstrates causal sequence between policy announcement and public reaction",
            context="Temporal edge confidence assessment for policy announcement → public reaction"
        )
        
        if not hasattr(confidence_result, 'probative_value'):
            raise LLMRequiredError("Confidence assessment missing probative_value - invalid LLM response")
            
        temporal_confidence = confidence_result.probative_value
    except Exception as e:
        raise LLMRequiredError(f"Cannot assess temporal confidence without LLM: {e}")
    
    edge = TemporalEdge(
        source="policy_announcement",
        target="public_reaction",
        temporal_relation=TemporalRelation.BEFORE,
        confidence=temporal_confidence,
        evidence_text=evidence_text
    )
    
    tg.add_temporal_edge(edge)
    
    # Create test critical juncture
    alternative = AlternativePathway(
        pathway_id="alt1",
        description="Alternative policy approach",
        probability=0.4,
        outcome_difference="Different public reaction",
        evidence_requirements=["Evidence of alternative consideration"],
        plausibility_score=0.6
    )
    
    # Assess juncture confidence using LLM
    try:
        from core.semantic_analysis_service import get_semantic_service
        from core.llm_required import LLMRequiredError
        
        semantic_service = get_semantic_service()
        juncture_evidence = "Decision meeting minutes show critical policy decision juncture with immediate announcement pathway"
        
        juncture_confidence_result = semantic_service.assess_probative_value(
            evidence_description=juncture_evidence,
            hypothesis_description="Critical juncture represents decisive decision point with significant counterfactual impact",
            context="Critical juncture confidence assessment for policy decision point"
        )
        
        if not hasattr(juncture_confidence_result, 'probative_value'):
            raise LLMRequiredError("Juncture confidence assessment missing probative_value - invalid LLM response")
            
        juncture_confidence = juncture_confidence_result.probative_value
    except Exception as e:
        raise LLMRequiredError(f"Cannot assess juncture confidence without LLM: {e}")
    
    juncture = CriticalJuncture(
        juncture_id="policy_decision_point",
        timestamp=datetime(2020, 1, 14),
        juncture_type=JunctureType.DECISION_POINT,
        description="Critical policy decision juncture",
        decision_point="Whether to announce policy immediately",
        key_nodes=["policy_announcement"],
        preceding_events=[],
        following_events=["public_reaction"],
        alternative_pathways=[alternative],
        actual_pathway="Immediate announcement",
        timing_sensitivity=0.8,
        counterfactual_impact=0.7,
        confidence=juncture_confidence,
        evidence_support=["Decision meeting minutes"],
        temporal_window=(datetime(2020, 1, 13), datetime(2020, 1, 15))
    )
    
    # Create test process durations with LLM-based duration confidence
    try:
        from core.semantic_analysis_service import get_semantic_service
        from core.llm_required import LLMRequiredError
        
        semantic_service = get_semantic_service()
        duration_evidence = "Policy announcement process completed in 2 hours during initiation phase with rapid speed"
        
        duration_confidence_result = semantic_service.assess_probative_value(
            evidence_description=duration_evidence,
            hypothesis_description="Process duration estimation demonstrates reliable temporal measurement and timing accuracy",
            context="Process duration confidence assessment for policy announcement timing"
        )
        
        if not hasattr(duration_confidence_result, 'probative_value'):
            raise LLMRequiredError("Duration confidence assessment missing probative_value - invalid LLM response")
            
        assessed_duration_confidence = duration_confidence_result.probative_value
    except Exception as e:
        raise LLMRequiredError(f"Cannot assess duration confidence without LLM: {e}")
    
    duration1 = ProcessDuration(
        process_id="policy_announcement",
        start_time=datetime(2020, 1, 15),
        end_time=datetime(2020, 1, 15, 2),
        duration=timedelta(hours=2),
        process_speed=ProcessSpeed.RAPID,
        temporal_phase=TemporalPhase.INITIATION,
        duration_confidence=assessed_duration_confidence,
        relative_duration=0.5,
        duration_percentile=0.3,
        efficiency_score=0.8,
        timing_optimality=0.7
    )
    
    # Test visualization
    visualizer = TemporalVisualizer()
    viz_data = visualizer.create_temporal_visualization(
        tg, [juncture], [duration1]
    )
    
    print("Temporal Visualization Test:")
    print(f"Timeline events: {len(viz_data.timeline_events)}")
    print(f"Network nodes: {len(viz_data.network_nodes)}")
    print(f"Network edges: {len(viz_data.network_edges)}")
    print(f"Critical junctures: {len(viz_data.critical_junctures)}")
    
    # Generate HTML (without saving)
    html_content = visualizer.generate_html_visualization(viz_data)
    print(f"Generated HTML length: {len(html_content)} characters")

if __name__ == "__main__":
    test_temporal_visualizer()
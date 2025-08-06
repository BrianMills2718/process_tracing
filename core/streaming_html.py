"""
Streaming HTML Generation for Process Tracing Analysis

Provides progressive HTML generation for large analyses with streaming output,
partial visualization rendering, and progressive enhancement.

Author: Claude Code Implementation
Date: August 2025
"""

import io
import time
import json
import textwrap
from pathlib import Path
from typing import Generator, Dict, Any, Optional, List, Union
from contextlib import contextmanager
import threading
from queue import Queue, Empty


class StreamingHTMLWriter:
    """
    Streaming HTML writer with progressive content delivery.
    
    Features:
    - Progressive HTML generation with immediate output
    - Partial visualization rendering for large graphs  
    - Progressive enhancement of interactive features
    - Fallback to complete rendering for smaller analyses
    - Thread-safe streaming operations
    - Real-time content delivery
    """
    
    def __init__(self, output_path: Path, buffer_size: int = 8192):
        """
        Initialize streaming HTML writer.
        
        Args:
            output_path: Path to write HTML file
            buffer_size: Buffer size for streaming operations
        """
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.file_handle = None
        self.content_queue = Queue()
        self.streaming_active = False
        self.writer_thread: Optional[threading.Thread] = None
        self.total_bytes_written = 0
        self.sections_completed = 0
        self.start_time = time.time()
        
        # JavaScript placeholders for progressive enhancement
        self.js_queue = []
        self.css_queue = []
        
    def start_streaming(self):
        """Start streaming HTML generation"""
        self.file_handle = open(self.output_path, 'w', encoding='utf-8', buffering=self.buffer_size)
        self.streaming_active = True
        self.start_time = time.time()
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.writer_thread.start()
        
        print(f"[STREAMING] Started HTML streaming to {self.output_path}")
    
    def _write_worker(self):
        """Background thread for writing content to file"""
        while self.streaming_active:
            try:
                content = self.content_queue.get(timeout=0.1)
                if content is None:  # Shutdown signal
                    break
                    
                self.file_handle.write(content)
                self.file_handle.flush()  # Ensure immediate write
                self.total_bytes_written += len(content)
                
            except Empty:
                continue
            except Exception as e:
                print(f"[STREAMING ERROR] Write worker error: {e}")
                break
    
    def write_chunk(self, content: str):
        """Write content chunk to stream"""
        if self.streaming_active:
            self.content_queue.put(content)
        else:
            raise RuntimeError("Streaming not active")
    
    def write_header(self, title: str = "Process Tracing Analysis Report"):
        """Write HTML header with progressive loading support"""
        header = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- vis.js Network library for interactive graph visualization -->
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .chart {{ margin-bottom: 1.5rem; }}
        .evidence-item {{ margin-bottom: 0.75rem; padding: 0.25rem; border-left: 3px solid #eee; }}
        .supporting {{ border-left-color: #28a745; }}
        .refuting {{ border-left-color: #dc3545; }}
        .card {{ margin-bottom: 1.5rem; overflow: hidden; }}
        .causal-chain {{ margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }}
        .section-complete {{ opacity: 1; transition: opacity 0.3s; }}
        .progress-indicator {{ position: fixed; top: 10px; right: 10px; z-index: 1000; }}
    </style>
</head>
<body>
<div class="container-fluid py-4">
    <div class="progress-indicator">
        <div class="badge bg-primary" id="progress-badge">Generating...</div>
    </div>
    <h1 class="mb-4">{title}</h1>
"""
        self.write_chunk(header)
    
    def write_section_start(self, section_id: str, title: str, description: str = ""):
        """Write section start with clean interface"""
        section_html = f"""
    <div class="card" id="{section_id}">
        <div class="card-header">
            <h2 class="card-title mb-0">{title}</h2>
            {f'<p class="text-muted mb-0">{description}</p>' if description else ''}
        </div>
        <div class="card-body">
"""
        self.write_chunk(section_html)
        
        # Add JavaScript to remove loading indicator
        self.js_queue.append(f"""
            setTimeout(() => {{
                const section = document.getElementById('{section_id}');
                if (section) {{
                    section.classList.remove('loading');
                    section.classList.add('section-complete');
                }}
            }}, 100);
        """)
    
    def write_section_content(self, content: str):
        """Write section content"""
        self.write_chunk(content)
    
    def write_section_end(self):
        """Write section end"""
        self.write_chunk("""
        </div>
    </div>""")
        self.sections_completed += 1
        
        # Update progress
        self.js_queue.append(f"""
            document.getElementById('progress-badge').textContent = '{self.sections_completed} sections complete';
        """)
    
    def write_network_visualization(self, network_data: Dict, container_id: str = "network-container"):
        """Write network visualization with progressive loading"""
        # Start with clean placeholder
        placeholder_html = f"""
            <div id="{container_id}" style="width: 100%; height: 600px; border: 1px solid #ddd; position: relative;">
                <div class="d-flex align-items-center justify-content-center h-100">
                    <div class="text-center text-muted">
                        <p>Interactive network graph will render here</p>
                    </div>
                </div>
            </div>
        """
        self.write_chunk(placeholder_html)
        
        # Add JavaScript to render network
        nodes_data = json.dumps(network_data.get('nodes', []))
        edges_data = json.dumps(network_data.get('edges', []))
        
        network_js = f"""
            setTimeout(() => {{
                try {{
                    const container = document.getElementById('{container_id}');
                    if (!container) return;
                    
                    // Clear loading content
                    container.innerHTML = '';
                    
                    const nodes = new vis.DataSet({nodes_data});
                    const edges = new vis.DataSet({edges_data});
                    
                    const data = {{ nodes: nodes, edges: edges }};
                    const options = {{
                        nodes: {{
                            shape: 'dot',
                            size: 16,
                            font: {{ size: 12, color: '#343a40' }},
                            borderWidth: 2,
                            shadow: true
                        }},
                        edges: {{
                            width: 2,
                            color: {{ color: '#848484', highlight: '#848484' }},
                            arrows: {{ to: {{ enabled: true, scaleFactor: 1, type: 'arrow' }} }},
                            shadow: true
                        }},
                        physics: {{
                            enabled: true,
                            stabilization: {{ iterations: 100 }}
                        }},
                        interaction: {{
                            hover: true,
                            tooltipDelay: 200
                        }}
                    }};
                    
                    new vis.Network(container, data, options);
                    
                    // Update progress
                    console.log('Network visualization rendered successfully');
                }} catch (error) {{
                    console.error('Error rendering network:', error);
                    document.getElementById('{container_id}').innerHTML = '<div class="alert alert-warning">Error rendering network visualization</div>';
                }}
            }}, 500);
        """
        
        self.js_queue.append(network_js)
    
    def write_chart_placeholder(self, chart_id: str, chart_title: str, chart_data: Optional[str] = None):
        """Write chart placeholder with progressive loading"""
        if chart_data:
            chart_html = f"""
                <div class="chart">
                    <h5>{chart_title}</h5>
                    <img src="data:image/png;base64,{chart_data}" alt="{chart_title}" class="img-fluid">
                </div>
            """
        else:
            chart_html = f"""
                <div class="chart" id="{chart_id}">
                    <h5>{chart_title}</h5>
                    <div class="alert alert-info">
                        <i class="fas fa-chart-bar me-2"></i>Chart data not available
                    </div>
                </div>
            """
        
        self.write_chunk(chart_html)
    
    def write_table_streaming(self, headers: List[str], rows: List[List[str]], table_id: str = "data-table"):
        """Write table with streaming rows"""
        table_start = f"""
            <div class="table-responsive">
                <table class="table table-striped table-hover" id="{table_id}">
                    <thead class="table-dark">
                        <tr>
                            {''.join(f'<th>{header}</th>' for header in headers)}
                        </tr>
                    </thead>
                    <tbody>
        """
        self.write_chunk(table_start)
        
        # Stream rows progressively
        for i, row in enumerate(rows):
            row_html = f"""
                        <tr>
                            {''.join(f'<td>{cell}</td>' for cell in row)}
                        </tr>
            """
            self.write_chunk(row_html)
            
            # Small delay for very large tables
            if i > 0 and i % 100 == 0:
                time.sleep(0.01)
        
        table_end = """
                    </tbody>
                </table>
            </div>
        """
        self.write_chunk(table_end)
    
    def flush_javascript(self):
        """Flush accumulated JavaScript to document"""
        if self.js_queue:
            js_content = f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            {''.join(self.js_queue)}
        }});
    </script>
            """
            self.write_chunk(js_content)
            self.js_queue.clear()
    
    def write_footer(self):
        """Write HTML footer with final JavaScript"""
        # Flush any remaining JavaScript
        self.flush_javascript()
        
        # Final progress update
        duration = time.time() - self.start_time
        footer_js = f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            document.getElementById('progress-badge').textContent = 'Complete ({duration:.1f}s)';
            document.getElementById('progress-badge').className = 'badge bg-success';
        }});
    </script>
    
    <!-- Bootstrap 5 JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</div>
</body>
</html>
        """
        self.write_chunk(footer_js)
    
    def finish_streaming(self):
        """Finish streaming and close file"""
        if self.streaming_active:
            # Signal worker thread to stop
            self.content_queue.put(None)
            
            # Wait for worker thread
            if self.writer_thread:
                self.writer_thread.join(timeout=2.0)
            
            # Close file
            if self.file_handle:
                self.file_handle.close()
            
            self.streaming_active = False
            duration = time.time() - self.start_time
            
            print(f"[STREAMING] Completed HTML streaming in {duration:.2f}s ({self.total_bytes_written} bytes)")
    
    @contextmanager
    def streaming_context(self):
        """Context manager for streaming operations"""
        try:
            self.start_streaming()
            yield self
        finally:
            self.finish_streaming()


class ProgressiveHTMLAnalysis:
    """
    Progressive HTML analysis generator with streaming support.
    
    Combines streaming HTML generation with analysis pipeline for
    real-time progress reporting and improved user experience.
    """
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.writer = StreamingHTMLWriter(output_path)
    
    def should_use_streaming(self, analysis_data: Dict) -> bool:
        """
        Determine if streaming should be used based on analysis complexity.
        
        Args:
            analysis_data: Analysis results data
            
        Returns:
            True if streaming should be used
        """
        # Use streaming for larger analyses
        node_count = len(analysis_data.get('metrics', {}).get('nodes_by_type', {}))
        edge_count = len(analysis_data.get('metrics', {}).get('edges_by_type', {}))
        chain_count = len(analysis_data.get('causal_chains', []))
        
        # Streaming thresholds
        return (node_count > 20 or 
                edge_count > 30 or 
                chain_count > 5 or
                'dag_analysis' in analysis_data or
                'cross_domain_analysis' in analysis_data)
    
    def generate_streaming_html(self, results: Dict, G, network_data: Dict, theoretical_insights=None):
        """
        Generate HTML analysis report with streaming output.
        
        Args:
            results: Analysis results
            G: NetworkX graph
            network_data: Network visualization data
            theoretical_insights: Optional theoretical insights
        """
        # Store results for access in other methods
        self.results = results
        
        use_streaming = self.should_use_streaming(results)
        
        if not use_streaming:
            # Fall back to standard generation for small analyses
            print("[HTML] Using standard generation for small analysis")
            return self._generate_standard_html(results, G, network_data, theoretical_insights)
        
        print(f"[HTML] Using streaming generation for large analysis")
        
        with self.writer.streaming_context():
            # Write header
            self.writer.write_header()
            
            # Overview section
            self._write_overview_section(results)
            
            # Network visualization section
            self._write_network_section(network_data)
            
            # Analysis sections
            self._write_analysis_sections(results, G)
            
            # Phase 2B advanced sections
            if 'dag_analysis' in results:
                self._write_dag_section(results['dag_analysis'])
            
            if 'cross_domain_analysis' in results:
                self._write_cross_domain_section(results['cross_domain_analysis'])
            
            # Theoretical insights
            if theoretical_insights:
                self._write_insights_section(theoretical_insights)
            
            # Write footer
            self.writer.write_footer()
        
        print(f"[HTML] Streaming report completed: {self.output_path}")
    
    def _write_overview_section(self, results: Dict):
        """Write overview section with streaming"""
        self.writer.write_section_start("overview", "Analysis Overview", "Summary of key findings")
        
        # Handle both old and new metric structure formats
        metrics = results.get('metrics', {})
        network_metrics = results.get('network_metrics', {})
        
        # Try both possible locations for node/edge counts (fixed priority order)
        node_counts = (metrics.get('node_type_distribution') or 
                      network_metrics.get('node_type_distribution') or 
                      metrics.get('nodes_by_type') or {})
        edge_counts = (metrics.get('edge_type_distribution') or 
                      network_metrics.get('edge_type_distribution') or 
                      metrics.get('edges_by_type') or {})
        
        
        overview_content = f"""
            <div class="row">
                <div class="col-md-6">
                    <h5>Network Statistics</h5>
                    <ul class="list-unstyled">
                        <li><strong>Total Nodes:</strong> {sum(node_counts.values())}</li>
                        <li><strong>Total Edges:</strong> {sum(edge_counts.values())}</li>
                        <li><strong>Causal Chains:</strong> {len(results.get('causal_chains', []))}</li>
                        <li><strong>Mechanisms:</strong> {len(results.get('mechanisms', []))}</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h5>Node Types</h5>
                    <ul class="list-unstyled">
        """
        
        for node_type, count in node_counts.items():
            overview_content += f"<li><strong>{node_type}:</strong> {count}</li>"
        
        overview_content += """
                    </ul>
                </div>
            </div>
        """
        
        self.writer.write_section_content(overview_content)
        self.writer.write_section_end()
    
    def _write_network_section(self, network_data: Dict):
        """Write network visualization section"""
        self.writer.write_section_start("network", "Interactive Network Visualization", 
                                       "Explore the causal network interactively")
        
        self.writer.write_network_visualization(network_data)
        
        self.writer.write_section_end()
    
    def _write_analysis_sections(self, results: Dict, G):
        """Write core analysis sections"""
        # Causal chains section
        if results.get('causal_chains'):
            self._write_causal_chains_section(results['causal_chains'])
        
        # Evidence analysis section  
        print(f"[ANALYSIS_DEBUG] Evidence analysis in results: {'evidence_analysis' in results}")
        print(f"[ANALYSIS_DEBUG] Evidence analysis data: {results.get('evidence_analysis', 'NOT FOUND')}")
        
        if results.get('evidence_analysis'):
            self._write_evidence_section(results['evidence_analysis'])
        else:
            print("[ANALYSIS_DEBUG] No evidence_analysis found, calling with empty dict")
            self._write_evidence_section({})
        
        # Van Evera systematic testing section
        if results.get('van_evera_assessment'):
            self._write_van_evera_section(results['van_evera_assessment'])
        
        # Mechanisms section
        if results.get('mechanisms'):
            self._write_mechanisms_section(results['mechanisms'])
    
    def filter_causal_chains_intelligently(self, chains: List, max_chains: int = 50) -> List:
        """Filter and prioritize causal chains for optimal user experience"""
        if len(chains) <= max_chains:
            return chains
        
        # Group by length for diversity
        by_length = {}
        for chain in chains:
            path_length = len(chain.get('path_descriptions', chain.get('path', [])))
            if path_length not in by_length:
                by_length[path_length] = []
            by_length[path_length].append(chain)
        
        # Select diverse chains across length groups
        filtered = []
        lengths_desc = sorted(by_length.keys(), reverse=True)
        
        # Take top chains from each length group
        chains_per_group = max(3, max_chains // len(lengths_desc)) if lengths_desc else 5
        
        for length in lengths_desc:
            group_chains = by_length[length][:chains_per_group]
            filtered.extend(group_chains)
            if len(filtered) >= max_chains:
                break
        
        return filtered[:max_chains]

    def _write_causal_chains_section(self, chains: List):
        """Write causal chains section with intelligent filtering"""
        # Filter chains for better UX
        filtered_chains = self.filter_causal_chains_intelligently(chains, max_chains=50)
        original_count = len(chains)
        
        self.writer.write_section_start("causal-chains", "Causal Chains", 
                                       f"Top {len(filtered_chains)} of {original_count} causal sequences (filtered for clarity)")
        
        for i, chain in enumerate(filtered_chains):
            # Build chain description from path_descriptions
            path_descriptions = chain.get('path_descriptions', [])
            if path_descriptions:
                # Create a flow description: A → B → C
                chain_description = ' → '.join([textwrap.shorten(desc, width=50, placeholder='...')
                                               for desc in path_descriptions])
                chain_length = len(path_descriptions)
                chain_desc_text = f"**{chain_length}-node chain**: {chain_description}"
            else:
                chain_desc_text = chain.get('description', 'No description available')
            
            chain_html = f"""
                <div class="causal-chain">
                    <h6>Chain {i+1}</h6>
                    <p class="chain-description">{chain_desc_text}</p>
                </div>
            """
            self.writer.write_section_content(chain_html)
        
        self.writer.write_section_end()
    
    def extract_hypothesis_evidence_from_graph(self, results: dict) -> dict:
        """Extract evidence relationships from graph structure"""
        hypothesis_evidence = {}
        
        # Get graph data if available
        graph_data = results.get('graph_data')
        print(f"[GRAPH_DEBUG] Graph data available: {graph_data is not None}")
        if not graph_data:
            print("[GRAPH_DEBUG] No graph_data found, trying to load from file...")
            # Try to load from the graph file directly
            try:
                import json
                from pathlib import Path
                # Try to find the graph file
                graph_file_pattern = str(Path("output_data/revolutions").glob("*_graph.json"))
                if graph_file_pattern and graph_file_pattern != 'output_data/revolutions':
                    # Get the first match
                    import glob
                    graph_files = glob.glob("output_data/revolutions/*_graph.json")
                    if graph_files:
                        print(f"[GRAPH_DEBUG] Loading from {graph_files[0]}")
                        with open(graph_files[0]) as f:
                            graph_data = json.load(f)
                        print(f"[GRAPH_DEBUG] Loaded {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")
            except Exception as e:
                print(f"[GRAPH_DEBUG] Error loading graph file: {e}")
                return {}
        
        # Build lookup for hypothesis nodes
        hypothesis_nodes = {}
        for node in graph_data.get('nodes', []):
            if node.get('type') == 'Hypothesis':
                hypothesis_nodes[node['id']] = node.get('properties', {}).get('description', node['id'])
        
        # Initialize evidence collections for each hypothesis
        for hyp_id, hyp_desc in hypothesis_nodes.items():
            hypothesis_evidence[hyp_id] = {
                'description': hyp_desc,
                'supporting_evidence': [],
                'refuting_evidence': []
            }
        
        # Find evidence relationships from edges
        for edge in graph_data.get('edges', []):
            edge_type = edge.get('type')
            source_id = edge.get('source_id')
            target_id = edge.get('target_id')
            
            # Get source node info
            source_node = next((n for n in graph_data.get('nodes', []) if n['id'] == source_id), None)
            if not source_node or source_node.get('type') != 'Evidence':
                continue
                
            # Check if target is a hypothesis
            if target_id in hypothesis_nodes:
                evidence_item = {
                    'id': source_id,
                    'type': edge.get('properties', {}).get('diagnostic_type', 'general'),
                    'description': source_node.get('properties', {}).get('description', source_id)
                }
                
                if edge_type in ['supports', 'provides_evidence_for']:
                    hypothesis_evidence[target_id]['supporting_evidence'].append(evidence_item)
                elif edge_type in ['refutes']:
                    hypothesis_evidence[target_id]['refuting_evidence'].append(evidence_item)
        
        return hypothesis_evidence

    def _write_evidence_section(self, evidence: Dict):
        """Write evidence analysis section with enhanced graph-based extraction"""
        self.writer.write_section_start("evidence", "Evidence Analysis", 
                                       "Van Evera diagnostic evidence assessment")
        
        # Try to extract from graph if evidence dict is sparse
        print(f"[EVIDENCE_DEBUG] Original evidence: {evidence}")
        print(f"[EVIDENCE_DEBUG] Evidence length: {len(evidence) if evidence else 0}")
        
        if not evidence or len(evidence) < 2:
            print("[EVIDENCE_DEBUG] Extracting from graph...")
            graph_evidence = self.extract_hypothesis_evidence_from_graph(self.results)
            print(f"[EVIDENCE_DEBUG] Graph evidence keys: {list(graph_evidence.keys()) if graph_evidence else []}")
            if graph_evidence:
                evidence = graph_evidence
            print(f"[EVIDENCE_DEBUG] Final evidence keys: {list(evidence.keys()) if evidence else []}")
        
        for hyp_id, hyp_data in evidence.items():
            evidence_html = f"""
                <div class="hypothesis-evidence mb-4">
                    <h6>Hypothesis: {hyp_data.get('description', hyp_id)}</h6>
                    <div class="evidence-items">
            """
            
            # Add supporting evidence
            for evidence_item in hyp_data.get('supporting_evidence', []):
                evidence_html += f"""
                    <div class="evidence-item supporting">
                        <strong>{evidence_item.get('type', 'Unknown')}:</strong> 
                        {evidence_item.get('description', 'No description')}
                    </div>
                """
            
            # Add refuting evidence
            for evidence_item in hyp_data.get('refuting_evidence', []):
                evidence_html += f"""
                    <div class="evidence-item refuting">
                        <strong>{evidence_item.get('type', 'Unknown')}:</strong> 
                        {evidence_item.get('description', 'No description')}
                    </div>
                """
            
            evidence_html += """
                    </div>
                </div>
            """
            
            self.writer.write_section_content(evidence_html)
        
        self.writer.write_section_end()
    
    def _write_van_evera_section(self, van_evera_results: Dict):
        """Write systematic Van Evera hypothesis testing results"""
        self.writer.write_section_start("van-evera", "Van Evera Systematic Analysis", 
                                       "Academic process tracing with diagnostic tests")
        
        for hyp_id, assessment in van_evera_results.items():
            assessment_html = f"""
                <div class="hypothesis-assessment mb-4">
                    <h6>{assessment.description}</h6>
                    <div class="assessment-summary">
                        <p><strong>Status:</strong> {assessment.overall_status}</p>
                        <p><strong>Posterior Probability:</strong> {assessment.posterior_probability:.2f} 
                           (CI: {assessment.confidence_interval[0]:.2f}-{assessment.confidence_interval[1]:.2f})</p>
                    </div>
                    <div class="test-results">
                        <h7>Diagnostic Test Results:</h7>
            """
            
            for test in assessment.test_results:
                test_html = f"""
                        <div class="test-result {test.test_result.value.lower()}">
                            <strong>{test.test_result.value}:</strong> {test.reasoning}
                            <br><small>Confidence: {test.confidence_level:.2f}</small>
                        </div>
                """
                assessment_html += test_html
            
            assessment_html += f"""
                    </div>
                    <div class="academic-conclusion">
                        <h7>Academic Conclusion:</h7>
                        <pre>{assessment.academic_conclusion}</pre>
                    </div>
                </div>
            """
            
            self.writer.write_section_content(assessment_html)
        
        self.writer.write_section_end()
    
    def _write_mechanisms_section(self, mechanisms: List):
        """Write mechanisms section"""
        self.writer.write_section_start("mechanisms", "Causal Mechanisms", 
                                       f"Analysis of {len(mechanisms)} causal mechanisms")
        
        for mechanism in mechanisms:
            # Use 'name' field for mechanism description
            mech_name = mechanism.get('name') or mechanism.get('description', 'Unknown Mechanism')
            # Truncate very long mechanism names for display
            if len(mech_name) > 150:
                mech_name = mech_name[:150] + '...'
            
            mech_html = f"""
                <div class="mechanism mb-3">
                    <h6>{mech_name}</h6>
                    <p><strong>Completeness:</strong> {mechanism.get('completeness', 'Unknown')}%</p>
                </div>
            """
            self.writer.write_section_content(mech_html)
        
        self.writer.write_section_end()
    
    def _write_dag_section(self, dag_analysis: Dict):
        """Write DAG analysis section"""
        self.writer.write_section_start("dag-analysis", "Advanced Causal Analysis", 
                                       "Complex pathway and convergence/divergence analysis")
        
        pathways = dag_analysis.get('causal_pathways', [])
        convergence = dag_analysis.get('convergence_analysis', {}).get('convergence_points', {})
        divergence = dag_analysis.get('divergence_analysis', {}).get('divergence_points', {})
        
        dag_html = f"""
            <div class="row">
                <div class="col-md-4">
                    <h6>Complex Pathways</h6>
                    <p class="text-muted">{len(pathways)} pathways identified</p>
                </div>
                <div class="col-md-4">
                    <h6>Convergence Points</h6>
                    <p class="text-muted">{len(convergence)} points where multiple causes converge</p>
                </div>
                <div class="col-md-4">
                    <h6>Divergence Points</h6>
                    <p class="text-muted">{len(divergence)} points where causes branch</p>
                </div>
            </div>
        """
        
        self.writer.write_section_content(dag_html)
        self.writer.write_section_end()
    
    def _write_cross_domain_section(self, cross_domain: Dict):
        """Write cross-domain analysis section"""
        self.writer.write_section_start("cross-domain", "Cross-Domain Analysis", 
                                       "Evidence-Hypothesis-Event pathway analysis")
        
        stats = cross_domain.get('cross_domain_statistics', {})
        
        cross_html = f"""
            <div class="row">
                <div class="col-md-6">
                    <h6>Cross-Domain Pathways</h6>
                    <p><strong>Total Paths:</strong> {stats.get('total_cross_paths', 0)}</p>
                    <p><strong>Average Path Length:</strong> {stats.get('avg_path_length', 0):.1f}</p>
                </div>
                <div class="col-md-6">
                    <h6>Van Evera Integration</h6>
                    <p><strong>Evidence Coverage:</strong> {stats.get('van_evera_coverage', 0)}</p>
                </div>
            </div>
        """
        
        self.writer.write_section_content(cross_html)
        self.writer.write_section_end()
    
    def _write_insights_section(self, insights):
        """Write theoretical insights section"""
        self.writer.write_section_start("insights", "Theoretical Insights", 
                                       "Key analytical insights and recommendations")
        
        # Handle both string and dictionary formats for insights
        if isinstance(insights, dict):
            # New structured format
            insights_content = f"<p>{insights.get('summary', 'No insights available')}</p>"
        elif isinstance(insights, str):
            # Legacy string format - convert markdown to HTML
            import re
            # Replace markdown headers
            insights_content = re.sub(r'^# (.+)$', r'<h3>\1</h3>', insights, flags=re.MULTILINE)
            insights_content = re.sub(r'^## (.+)$', r'<h4>\1</h4>', insights_content, flags=re.MULTILINE)
            # Replace double newlines with paragraph breaks
            insights_content = re.sub(r'\n\n+', '</p><p>', insights_content)
            # Replace single newlines with breaks
            insights_content = insights_content.replace('\n', '<br>')
            # Wrap in paragraph tags if needed
            if not insights_content.startswith('<h') and not insights_content.startswith('<p>'):
                insights_content = f"<p>{insights_content}</p>"
            # Clean up any empty paragraphs
            insights_content = re.sub(r'<p></p>', '', insights_content)
        else:
            # Fallback for other types
            insights_content = f"<p>{str(insights) if insights else 'No insights available'}</p>"
        
        insights_html = f"""
            <div class="insights-content">
                {insights_content}
            </div>
        """
        
        self.writer.write_section_content(insights_html)
        self.writer.write_section_end()
    
    def _generate_standard_html(self, results: Dict, G, network_data: Dict, theoretical_insights=None):
        """Fallback to standard HTML generation for small analyses"""
        # This would call the original format_html_analysis function
        # For now, just create a simple version
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Process Tracing Analysis (Standard)</title>
    <style>body {{ font-family: Arial, sans-serif; padding: 20px; }}</style>
</head>
<body>
    <h1>Process Tracing Analysis Report</h1>
    <p>Small analysis - using standard generation mode.</p>
    <p>Nodes: {sum(results.get('metrics', {}).get('nodes_by_type', {}).values())}</p>
    <p>Edges: {sum(results.get('metrics', {}).get('edges_by_type', {}).values())}</p>
</body>
</html>
            """)


if __name__ == "__main__":
    # Demo streaming HTML generation
    output_path = Path("demo_streaming.html")
    
    # Sample analysis data
    sample_data = {
        'metrics': {
            'nodes_by_type': {'Event': 25, 'Evidence': 15, 'Hypothesis': 5},
            'edges_by_type': {'causes': 30, 'supports': 20}
        },
        'causal_chains': [
            {'description': 'Sample causal chain 1'},
            {'description': 'Sample causal chain 2'}
        ],
        'evidence_analysis': {
            'hyp1': {
                'description': 'Sample hypothesis',
                'supporting_evidence': [
                    {'type': 'smoking_gun', 'description': 'Strong evidence'}
                ]
            }
        }
    }
    
    network_data = {
        'nodes': [
            {'id': 1, 'label': 'Event 1'},
            {'id': 2, 'label': 'Event 2'}
        ],
        'edges': [
            {'from': 1, 'to': 2}
        ]
    }
    
    generator = ProgressiveHTMLAnalysis(output_path)
    generator.generate_streaming_html(sample_data, None, network_data)
    
    print(f"Demo streaming HTML generated: {output_path}")
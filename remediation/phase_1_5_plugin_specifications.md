# Phase 1.5: Plugin Specifications

*Detailed specifications for critical plugins addressing special considerations*

## Token Limit Plugin

**Issue Addressed**: #85 - Enforce Gemini's 1M token limit without chunking

```python
# core/plugins/validation/token_limit_plugin.py
from core.plugins.base import ProcessTracingPlugin

class TokenLimitPlugin(ProcessTracingPlugin):
    """Enforces Gemini's 1M token limit - fails loud if exceeded"""
    plugin_id = "token_limit_validator"
    
    MAX_TOKENS = 1_000_000
    CHARS_PER_TOKEN_ESTIMATE = 4  # Conservative estimate
    
    def validate_input(self, data):
        if 'text' in data:
            estimated_tokens = len(data['text']) / self.CHARS_PER_TOKEN_ESTIMATE
            if estimated_tokens > self.MAX_TOKENS:
                raise ValueError(
                    f"Text too large: ~{estimated_tokens:,.0f} tokens exceeds "
                    f"Gemini's {self.MAX_TOKENS:,} token limit. "
                    f"Maximum text length is ~{self.MAX_TOKENS * self.CHARS_PER_TOKEN_ESTIMATE:,} characters."
                )
        return True
    
    def execute(self, data):
        # Validation happens in validate_input
        return data
    
    def get_checkpoint_data(self):
        return {"validated": True, "max_tokens": self.MAX_TOKENS}
```

**Test**:
```python
def test_token_limit_plugin_fails_loud():
    plugin = TokenLimitPlugin("token_limit", mock_context)
    
    # Just under limit - should pass
    small_text = "x" * 3_999_999
    plugin.validate_input({"text": small_text})
    
    # Over limit - should fail loud
    large_text = "x" * 4_000_001
    with pytest.raises(ValueError, match="exceeds Gemini's 1,000,000 token limit"):
        plugin.validate_input({"text": large_text})
```

## Floating Point Plugin

**Issue Addressed**: #63 - Consistent floating point comparisons with ε=1e-9

```python
# core/plugins/math/floating_point_plugin.py
from core.plugins.base import ProcessTracingPlugin
from typing import List, Tuple

class FloatingPointPlugin(ProcessTracingPlugin):
    """Handles floating point comparisons consistently"""
    plugin_id = "floating_point_handler"
    
    EPSILON = 1e-9  # High precision for scientific accuracy
    
    def validate_input(self, data):
        # This plugin provides utilities, no validation needed
        return True
    
    def execute(self, data):
        # Process any floating point comparisons in the data
        if 'probabilities' in data:
            data['probabilities'] = self._clean_probabilities(data['probabilities'])
        if 'scores' in data:
            data['scores'] = self._normalize_scores(data['scores'])
        return data
    
    def are_equal(self, a: float, b: float) -> bool:
        """Compare floats with epsilon tolerance"""
        return abs(a - b) < self.EPSILON
    
    def is_zero(self, value: float) -> bool:
        """Check if effectively zero"""
        return abs(value) < self.EPSILON
    
    def compare(self, a: float, b: float) -> int:
        """Returns -1 if a<b, 0 if a==b, 1 if a>b"""
        if self.are_equal(a, b):
            return 0
        return -1 if a < b else 1
    
    def _clean_probabilities(self, probs: List[float]) -> List[float]:
        """Ensure probabilities sum to 1.0 within epsilon"""
        total = sum(probs)
        if not self.are_equal(total, 1.0):
            # Normalize to sum to 1.0
            return [p / total for p in probs]
        return probs
    
    def _normalize_scores(self, scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Handle near-zero scores consistently"""
        return [(name, 0.0 if self.is_zero(score) else score) 
                for name, score in scores]
    
    def get_checkpoint_data(self):
        return {"epsilon": self.EPSILON}
```

**Test**:
```python
def test_floating_point_precision():
    plugin = FloatingPointPlugin("fp", mock_context)
    
    # Classic floating point problem
    a = 0.1 + 0.2
    b = 0.3
    assert a != b  # Python says they're different
    assert plugin.are_equal(a, b)  # Plugin says they're equal
    
    # Near-zero detection
    assert plugin.is_zero(1e-10)
    assert not plugin.is_zero(1e-8)
    
    # Probability normalization
    probs = [0.33, 0.33, 0.33]  # Sum is 0.99
    cleaned = plugin._clean_probabilities(probs)
    assert plugin.are_equal(sum(cleaned), 1.0)
```

## Interactive Visualization Plugin

**Issues Addressed**: #59-60 - Thread-safe interactive visualization

```python
# core/plugins/visualization/interactive_viz_plugin.py
import matplotlib
matplotlib.use('Qt5Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
import networkx as nx
from threading import Lock
from core.plugins.base import ProcessTracingPlugin

class InteractiveVisualizationPlugin(ProcessTracingPlugin):
    """Thread-safe interactive graph visualization"""
    plugin_id = "interactive_visualizer"
    
    _plot_lock = Lock()  # Class-level lock for matplotlib
    
    def validate_input(self, data):
        if 'graph' not in data:
            raise ValueError("No graph provided for visualization")
        if not isinstance(data['graph'], nx.DiGraph):
            raise TypeError("Graph must be a NetworkX DiGraph")
        return True
    
    def execute(self, data):
        graph = data['graph']
        title = data.get('title', 'Process Tracing Graph')
        
        with self._plot_lock:  # Ensure thread safety
            plt.ioff()  # Disable interactive mode during setup
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.suptitle(title, fontsize=16)
            
            # Use thread-safe layout algorithm
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
            
            # Color nodes by type
            node_colors = []
            for node in graph.nodes():
                node_type = graph.nodes[node].get('type', 'default')
                color_map = {
                    'event': 'lightblue',
                    'actor': 'lightgreen',
                    'hypothesis': 'lightyellow',
                    'evidence': 'lightcoral',
                    'default': 'lightgray'
                }
                node_colors.append(color_map.get(node_type, 'lightgray'))
            
            # Draw the graph
            nx.draw(graph, pos, ax=ax, with_labels=True, 
                   node_color=node_colors, edge_color='gray',
                   node_size=1000, font_size=10, arrows=True)
            
            # Add edge labels if present
            edge_labels = nx.get_edge_attributes(graph, 'label')
            if edge_labels:
                nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)
            
            # Enable interactivity
            plt.ion()
            plt.tight_layout()
            
            # Save to file before showing
            output_path = self._get_output_path(data)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {output_path}")
            
            # Show interactive plot
            plt.show(block=False)  # Non-blocking show
            
            # Save reference for cleanup
            self._current_figure = fig
            
        return {
            'visualization': fig, 
            'layout': pos,
            'output_path': str(output_path)
        }
    
    def cleanup(self):
        """Close figures properly to prevent memory leaks"""
        with self._plot_lock:
            if hasattr(self, '_current_figure'):
                plt.close(self._current_figure)
                self.logger.info("Closed visualization figure")
    
    def _get_output_path(self, data):
        """Generate output path for saved visualization"""
        from pathlib import Path
        case_id = data.get('case_id', 'unnamed')
        timestamp = data.get('timestamp', 'undated')
        return Path('output_data') / case_id / f'graph_{timestamp}.png'
    
    def get_checkpoint_data(self):
        return {
            "backend": matplotlib.get_backend(),
            "thread_safe": True
        }
```

**Test**:
```python
def test_thread_safe_visualization():
    plugin = InteractiveVisualizationPlugin("viz", mock_context)
    
    # Create test graph
    G = nx.DiGraph()
    G.add_edge("A", "B", label="causes")
    G.add_node("A", type="event")
    G.add_node("B", type="event")
    
    # Test in multiple threads
    import threading
    errors = []
    
    def viz_thread():
        try:
            plugin.execute({"graph": G, "title": "Test"})
            plugin.cleanup()
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=viz_thread) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0  # No threading errors
```

## Cross-Platform I/O Plugin

**Issues Addressed**: #39-40 - Consistent cross-platform file I/O

```python
# core/plugins/io/cross_platform_io_plugin.py
from pathlib import Path
from core.plugins.base import ProcessTracingPlugin
import json

class CrossPlatformIOPlugin(ProcessTracingPlugin):
    """Cross-platform file I/O handling"""
    plugin_id = "cross_platform_io"
    
    def validate_input(self, data):
        # This plugin provides utilities, validation depends on operation
        return True
    
    def execute(self, data):
        operation = data.get('operation')
        
        if operation == 'read':
            content = self.read_file(data['filepath'])
            return {'content': content}
        
        elif operation == 'write':
            self.write_file(data['filepath'], data['content'])
            return {'success': True, 'path': data['filepath']}
        
        elif operation == 'read_json':
            obj = self.read_json(data['filepath'])
            return {'object': obj}
        
        elif operation == 'write_json':
            self.write_json(data['filepath'], data['object'])
            return {'success': True, 'path': data['filepath']}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def read_file(self, filepath):
        """Read file with proper encoding handling"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Always use UTF-8, handle BOM
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            content = f.read()
        
        self.logger.info(f"Read {len(content)} characters from {filepath}")
        return content
    
    def write_file(self, filepath, content):
        """Write file with consistent encoding"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Always write UTF-8 without BOM, with Unix line endings
        # This ensures consistency across platforms
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        self.logger.info(f"Wrote {len(content)} characters to {filepath}")
    
    def read_json(self, filepath):
        """Read JSON with proper encoding"""
        content = self.read_file(filepath)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")
    
    def write_json(self, filepath, obj):
        """Write JSON with consistent formatting"""
        # Ensure consistent JSON formatting across platforms
        content = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
        self.write_file(filepath, content)
    
    def get_output_path(self, case_id, filename):
        """Generate cross-platform output paths"""
        # Use pathlib for automatic OS-appropriate separators
        return Path("output_data") / case_id / filename
    
    def get_checkpoint_data(self):
        return {
            "encoding": "utf-8",
            "line_endings": "unix (\\n)",
            "json_format": "indented, sorted keys"
        }
```

**Test**:
```python
def test_cross_platform_io():
    plugin = CrossPlatformIOPlugin("io", mock_context)
    
    # Test path generation works on all platforms
    path = plugin.get_output_path("case1", "results.json")
    assert str(path).replace('\\', '/') == "output_data/case1/results.json"
    
    # Test line ending normalization
    mixed_content = "Line1\r\nLine2\rLine3\nLine4"
    plugin.write_file("test.txt", mixed_content)
    read_back = plugin.read_file("test.txt")
    assert read_back == "Line1\nLine2\nLine3\nLine4"  # All normalized to \n
    
    # Test UTF-8 handling
    unicode_content = "Hello 世界 🌍"
    plugin.write_file("unicode.txt", unicode_content)
    assert plugin.read_file("unicode.txt") == unicode_content
    
    # Test JSON consistency
    obj = {"b": 2, "a": 1, "c": {"nested": True}}
    plugin.write_json("test.json", obj)
    loaded = plugin.read_json("test.json")
    assert loaded == obj
```

## Integration into Workflow

These plugins integrate seamlessly into the analysis workflow:

```python
# Example workflow with special consideration plugins
workflow = {
    "nodes": [
        {"id": "token_check", "plugin": "token_limit_validator"},
        {"id": "import", "plugin": "cross_platform_io"},
        {"id": "extract", "plugin": "text_extraction"},
        {"id": "fp_normalize", "plugin": "floating_point_handler"},
        {"id": "analyze", "plugin": "van_evera_analysis"},
        {"id": "visualize", "plugin": "interactive_visualizer"},
        {"id": "export", "plugin": "cross_platform_io"}
    ],
    "edges": [
        {"from": "token_check", "to": "import"},
        {"from": "import", "to": "extract"},
        {"from": "extract", "to": "fp_normalize"},
        {"from": "fp_normalize", "to": "analyze"},
        {"from": "analyze", "to": "visualize"},
        {"from": "visualize", "to": "export"}
    ]
}
```

## Summary

These four plugins address the special considerations:

1. **TokenLimitPlugin**: Hard fails on >1M tokens, no chunking
2. **FloatingPointPlugin**: Consistent ε=1e-9 comparisons
3. **InteractiveVisualizationPlugin**: Thread-safe Qt5Agg backend
4. **CrossPlatformIOPlugin**: pathlib + UTF-8 for all platforms

All plugins follow the fail-fast, fully observable, checkpoint-enabled principles established in Phase 1.
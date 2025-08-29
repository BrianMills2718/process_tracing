#!/usr/bin/env python3
"""
Classify all Python files in core/ into categories for LLM migration.
"""

import ast
from pathlib import Path
import re

def classify_file(filepath):
    """Classify a file into semantic, computational, or dead categories"""
    
    filename = filepath.name
    content = filepath.read_text(encoding='utf-8', errors='ignore')
    
    # Dead files (known unused)
    if 'confidence_calculator.py' in filename:
        return 'dead', 'Known unused file (not imported anywhere)'
    
    # Check for computational patterns
    computational_patterns = [
        'networkx', 'matplotlib', 'plotly', 'scipy.stats',
        'numpy', 'pandas.DataFrame', 'graph.nodes', 'graph.edges',
        'PageRank', 'centrality', 'export_to', 'save_to_file',
        'write_csv', 'to_json', 'visualization', 'render_'
    ]
    
    computational_files = [
        'network_analysis', 'statistical_validator', 'visualization',
        'data_export', 'graph_builder', 'file_utils', 'export_utils',
        'output_formatter', 'report_generator'
    ]
    
    # Check filename
    for comp_file in computational_files:
        if comp_file in filename.lower():
            return 'computational', f'File name suggests computational: {comp_file}'
    
    # Check content
    for pattern in computational_patterns:
        if pattern in content:
            return 'computational', f'Contains computational pattern: {pattern}'
    
    # Check for semantic patterns
    semantic_patterns = [
        'semantic_service', 'require_llm', 'LLMRequiredError',
        'get_van_evera_llm', 'assess_', 'evaluate_', 'classify_',
        'enhance_', 'analyze_relationship', 'interpret_'
    ]
    
    semantic_files = [
        'semantic_analysis', 'llm_required', 'van_evera',
        'hypothesis', 'evidence', 'causal', 'temporal',
        'counterfactual', 'process_discovery', 'diagnostic'
    ]
    
    # Check filename
    for sem_file in semantic_files:
        if sem_file in filename.lower():
            # Special case: graph_builder is computational
            if 'graph_builder' not in filename.lower():
                return 'semantic', f'File name suggests semantic: {sem_file}'
    
    # Check content
    for pattern in semantic_patterns:
        if pattern in content:
            return 'semantic', f'Contains semantic pattern: {pattern}'
    
    # Check for plugin patterns
    if 'plugins/' in str(filepath):
        # Most plugins need semantic understanding
        if any(comp in filename for comp in ['export', 'visualization', 'network']):
            return 'computational', 'Plugin for computational tasks'
        else:
            return 'semantic', 'Plugin requiring semantic understanding'
    
    # Default: if has classes/functions dealing with analysis, likely semantic
    if 'def analyze' in content or 'def evaluate' in content or 'def assess' in content:
        return 'semantic', 'Contains analysis methods'
    
    # If primarily utility functions
    if 'def save_' in content or 'def load_' in content or 'def format_' in content:
        return 'computational', 'Utility functions for I/O or formatting'
    
    # Default to semantic (safer to check if needs LLM)
    return 'semantic', 'Default classification - needs review'

def main():
    """Classify all Python files in core/"""
    
    categories = {
        'semantic': [],
        'computational': [],
        'dead': []
    }
    
    # Find all Python files
    core_path = Path('core')
    py_files = sorted(core_path.rglob('*.py'))
    
    print("# File Classification for LLM Migration\n")
    print(f"Total Python files in core/: {len(py_files)}\n")
    
    # Classify each file
    for filepath in py_files:
        category, reason = classify_file(filepath)
        rel_path = filepath.relative_to(core_path)
        categories[category].append((str(rel_path), reason))
    
    # Output results by category
    print("## Category A: Semantic (MUST be LLM-first)")
    print(f"Count: {len(categories['semantic'])}\n")
    for file, reason in sorted(categories['semantic']):
        print(f"- `{file}`: {reason}")
    
    print("\n## Category B: Computational (Keep non-LLM)")
    print(f"Count: {len(categories['computational'])}\n")
    for file, reason in sorted(categories['computational']):
        print(f"- `{file}`: {reason}")
    
    print("\n## Category D: Dead (Delete these)")
    print(f"Count: {len(categories['dead'])}\n")
    for file, reason in sorted(categories['dead']):
        print(f"- `{file}`: {reason}")
    
    # Summary
    print("\n## Summary")
    print(f"- Total files classified: {len(py_files)}")
    print(f"- Semantic (need LLM): {len(categories['semantic'])} ({len(categories['semantic'])*100//len(py_files)}%)")
    print(f"- Computational (keep as-is): {len(categories['computational'])} ({len(categories['computational'])*100//len(py_files)}%)")
    print(f"- Dead (remove): {len(categories['dead'])} ({len(categories['dead'])*100//len(py_files)}%)")
    
    # Migration priority
    print("\n## Migration Priority (Semantic Files)")
    print("\nHigh Priority (core functionality):")
    high_priority = [f for f, _ in categories['semantic'] if any(x in f for x in ['van_evera', 'semantic', 'analyze', 'llm_required'])]
    for f in high_priority[:10]:
        print(f"1. `{f}`")
    
    print("\nMedium Priority (plugins):")
    medium_priority = [f for f, _ in categories['semantic'] if 'plugins/' in f and f not in high_priority]
    for f in medium_priority[:10]:
        print(f"2. `{f}`")
    
    print("\nLow Priority (utilities):")
    low_priority = [f for f, _ in categories['semantic'] if f not in high_priority and f not in medium_priority]
    for f in low_priority[:10]:
        print(f"3. `{f}`")

if __name__ == "__main__":
    main()
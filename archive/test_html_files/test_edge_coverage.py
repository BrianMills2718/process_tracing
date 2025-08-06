#!/usr/bin/env python3
"""
Edge Type Coverage Verification Tool

This tool analyzes extraction results to verify edge type coverage against the
complete ontology schema, providing detailed reports and improvement suggestions.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime


def load_ontology_schema():
    """Load the complete ontology schema from config file."""
    config_path = Path("config/ontology_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Ontology config not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config['edge_types']


def load_extraction_results(file_path):
    """Load extraction results from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Extraction results not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def analyze_edge_coverage(schema_edges, found_edges):
    """Analyze coverage and return detailed report."""
    schema_set = set(schema_edges.keys())
    found_set = set(found_edges)
    
    missing = schema_set - found_set
    extra = found_set - schema_set
    found_count = len(found_set)
    total_count = len(schema_set) 
    coverage_percent = (found_count / total_count) * 100 if total_count > 0 else 0
    
    return {
        'total_defined': total_count,
        'total_found': found_count,
        'coverage_percent': coverage_percent,
        'missing': sorted(missing),
        'extra': sorted(extra),
        'found': sorted(found_set),
        'complete': len(missing) == 0
    }


def generate_improvement_suggestions(missing_edges, schema_edges):
    """Generate specific suggestions for missing edge types."""
    suggestions = []
    
    edge_patterns = {
        'explains_mechanism': {
            'pattern': 'Hypothesis explaining how a mechanism works',
            'text_examples': [
                '"The hypothesis explains that the mechanism operates through..."',
                '"This theory explains the mechanism by..."',
                '"The hypothesis explains how the mechanism functions via..."'
            ],
            'domain_range': f"Domain: {schema_edges.get('explains_mechanism', {}).get('domain', [])} -> Range: {schema_edges.get('explains_mechanism', {}).get('range', [])}"
        },
        'part_of_mechanism': {
            'pattern': 'Event as component of a larger mechanism',
            'text_examples': [
                '"This event was a crucial part of the mechanism..."',
                '"The event served as a component of the larger mechanism..."',
                '"As part of the resistance mechanism, this event..."'
            ],
            'domain_range': f"Domain: {schema_edges.get('part_of_mechanism', {}).get('domain', [])} -> Range: {schema_edges.get('part_of_mechanism', {}).get('range', [])}"
        },
        'disproves_occurrence': {
            'pattern': 'Evidence showing an event did NOT happen',
            'text_examples': [
                '"Evidence shows this event did NOT occur because..."',
                '"Records disprove that this event happened by..."',
                '"Documentary evidence disproves the occurrence of..."'
            ],
            'domain_range': f"Domain: {schema_edges.get('disproves_occurrence', {}).get('domain', [])} -> Range: {schema_edges.get('disproves_occurrence', {}).get('range', [])}"
        },
        'initiates': {
            'pattern': 'Actor directly starting or launching an event',
            'text_examples': [
                '"Actor X initiated the event by..."',
                '"The leader launched/started the action..."',
                '"Person Y directly initiated the process through..."'
            ],
            'domain_range': f"Domain: {schema_edges.get('initiates', {}).get('domain', [])} -> Range: {schema_edges.get('initiates', {}).get('range', [])}"
        },
        'provides_evidence': {
            'pattern': 'Data source providing evidence',
            'text_examples': [
                '"Archive documents provide evidence that..."',
                '"Interview data provides evidence about..."',
                '"The source provides evidence for..."'
            ],
            'domain_range': f"Domain: {schema_edges.get('provides_evidence', {}).get('domain', [])} -> Range: {schema_edges.get('provides_evidence', {}).get('range', [])}"
        },
        'refutes': {
            'pattern': 'Evidence contradicting hypothesis/event/mechanism',
            'text_examples': [
                '"Evidence refutes the hypothesis by showing..."',
                '"Data refutes the claim that..."',
                '"Records refute the theory by demonstrating..."'
            ],
            'domain_range': f"Domain: {schema_edges.get('refutes', {}).get('domain', [])} -> Range: {schema_edges.get('refutes', {}).get('range', [])}"
        }
    }
    
    for edge_type in missing_edges:
        if edge_type in edge_patterns:
            pattern_info = edge_patterns[edge_type]
            suggestions.append({
                'edge_type': edge_type,
                'pattern': pattern_info['pattern'],
                'domain_range': pattern_info['domain_range'],
                'text_examples': pattern_info['text_examples'],
                'priority': 'high' if edge_type in ['explains_mechanism', 'part_of_mechanism'] else 'medium'
            })
    
    return suggestions


def print_coverage_report(analysis, suggestions=None):
    """Print detailed coverage report."""
    print("=" * 60)
    print("EDGE TYPE COVERAGE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("COVERAGE SUMMARY:")
    print(f"  Total edge types defined: {analysis['total_defined']}")
    print(f"  Total edge types found: {analysis['total_found']}")
    print(f"  Coverage percentage: {analysis['coverage_percent']:.1f}%")
    print(f"  Status: {'COMPLETE' if analysis['complete'] else 'INCOMPLETE'}")
    print()
    
    if analysis['missing']:
        print(f"MISSING EDGE TYPES ({len(analysis['missing'])}):")
        for edge_type in analysis['missing']:
            print(f"  - {edge_type}")
        print()
    
    if analysis['extra']:
        print(f"EXTRA EDGE TYPES ({len(analysis['extra'])}):")
        for edge_type in analysis['extra']:
            print(f"  + {edge_type}")
        print()
    
    print(f"FOUND EDGE TYPES ({len(analysis['found'])}):")
    for edge_type in analysis['found']:
        print(f"  * {edge_type}")
    print()
    
    if suggestions:
        print("IMPROVEMENT SUGGESTIONS:")
        print("-" * 40)
        for suggestion in suggestions:
            print(f"Edge Type: {suggestion['edge_type']} (Priority: {suggestion['priority']})")
            print(f"  Pattern: {suggestion['pattern']}")
            print(f"  Schema: {suggestion['domain_range']}")
            print(f"  Text Examples:")
            for example in suggestion['text_examples']:
                print(f"    - {example}")
            print()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python test_edge_coverage.py <extraction_results.json>")
        print("       python test_edge_coverage.py --latest  # Use latest extraction")
        sys.exit(1)
    
    try:
        # Load schema
        schema_edges = load_ontology_schema()
        
        # Determine input file
        if sys.argv[1] == '--latest':
            # Find latest extraction file
            output_dir = Path("output_data/test_mechanism")
            if not output_dir.exists():
                raise FileNotFoundError("No test_mechanism output directory found")
            
            json_files = list(output_dir.glob("test_mechanism_*_graph.json"))
            if not json_files:
                raise FileNotFoundError("No extraction JSON files found")
            
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            print(f"Using latest extraction: {latest_file}")
            extraction_file = str(latest_file)
        else:
            extraction_file = sys.argv[1]
        
        # Load extraction results
        extraction_data = load_extraction_results(extraction_file)
        
        # Extract edge types from results
        found_edges = [edge.get('type') for edge in extraction_data.get('edges', [])]
        found_edges = [edge for edge in found_edges if edge]  # Remove None values
        
        # Analyze coverage
        analysis = analyze_edge_coverage(schema_edges, found_edges)
        
        # Generate suggestions for missing edges
        suggestions = None
        if analysis['missing']:
            suggestions = generate_improvement_suggestions(analysis['missing'], schema_edges)
        
        # Print report
        print_coverage_report(analysis, suggestions)
        
        # Exit with appropriate code
        sys.exit(0 if analysis['complete'] else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
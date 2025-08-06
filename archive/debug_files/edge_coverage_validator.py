#!/usr/bin/env python3
"""
Rigorous Edge Type Coverage Validator
Validates claims about edge type coverage by analyzing schema vs actual extraction results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple

def load_schema_edge_types(config_path: str) -> Set[str]:
    """Load all edge types defined in the ontology schema."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    edge_types = set(config.get('edge_types', {}).keys())
    return edge_types

def analyze_extraction_file(file_path: str) -> Tuple[str, Set[str], int]:
    """Analyze a single extraction JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    edges = data.get('edges', [])
    edge_types_found = set(edge.get('type') for edge in edges if edge.get('type'))
    
    return file_path, edge_types_found, len(edges)

def validate_coverage_claims():
    """Perform comprehensive validation of edge type coverage claims."""
    
    # Paths
    base_dir = Path(__file__).parent
    config_path = base_dir / "config" / "ontology_config.json"
    output_dir = base_dir / "output_data" / "test_mechanism"
    
    print("RIGOROUS EDGE TYPE COVERAGE VALIDATION")
    print("=" * 60)
    
    # 1. Schema Analysis
    print("\n1. SCHEMA ANALYSIS")
    print("-" * 30)
    
    if not config_path.exists():
        print(f"ERROR: Schema file not found: {config_path}")
        return
        
    schema_edge_types = load_schema_edge_types(str(config_path))
    print(f"SUCCESS: Schema defines {len(schema_edge_types)} edge types:")
    for i, edge_type in enumerate(sorted(schema_edge_types), 1):
        print(f"   {i:2d}. {edge_type}")
    
    # 2. Extraction Results Analysis
    print(f"\n2. EXTRACTION RESULTS ANALYSIS")
    print("-" * 40)
    
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        return
    
    graph_files = list(output_dir.glob("*_graph.json"))
    if not graph_files:
        print(f"ERROR: No graph JSON files found in {output_dir}")
        return
    
    print(f"FOUND: {len(graph_files)} extraction files")
    
    # Analyze each file
    all_found_edge_types = set()
    file_results = []
    
    for file_path in sorted(graph_files):
        file_name, edge_types_found, edge_count = analyze_extraction_file(str(file_path))
        file_results.append((file_name, edge_types_found, edge_count))
        all_found_edge_types.update(edge_types_found)
        
        coverage_percent = (len(edge_types_found) / len(schema_edge_types)) * 100
        print(f"   FILE: {file_path.name}:")
        print(f"      Edge types: {len(edge_types_found)}/{len(schema_edge_types)} ({coverage_percent:.1f}%)")
        print(f"      Total edges: {edge_count}")
    
    # 3. Overall Coverage Analysis
    print(f"\n3. OVERALL COVERAGE ANALYSIS")
    print("-" * 35)
    
    overall_coverage_percent = (len(all_found_edge_types) / len(schema_edge_types)) * 100
    print(f"MAXIMUM ACHIEVED COVERAGE: {len(all_found_edge_types)}/{len(schema_edge_types)} ({overall_coverage_percent:.1f}%)")
    
    # 4. Missing Edge Types
    missing_edge_types = schema_edge_types - all_found_edge_types
    print(f"\n4. MISSING EDGE TYPES")
    print("-" * 25)
    
    if missing_edge_types:
        print(f"ERROR: {len(missing_edge_types)} edge types NEVER extracted:")
        for i, edge_type in enumerate(sorted(missing_edge_types), 1):
            print(f"   {i:2d}. {edge_type}")
    else:
        print("SUCCESS: ALL edge types have been extracted at least once!")
    
    # 5. Best Coverage File
    print(f"\n5. BEST COVERAGE ANALYSIS")
    print("-" * 30)
    
    best_file = max(file_results, key=lambda x: len(x[1]))
    best_coverage_percent = (len(best_file[1]) / len(schema_edge_types)) * 100
    
    print(f"BEST SINGLE-FILE COVERAGE:")
    print(f"   File: {Path(best_file[0]).name}")
    print(f"   Coverage: {len(best_file[1])}/{len(schema_edge_types)} ({best_coverage_percent:.1f}%)")
    print(f"   Total edges: {best_file[2]}")
    
    # 6. Claim Validation
    print(f"\n6. CLAIM VALIDATION")
    print("-" * 25)
    
    print("Checking claims from CLAUDE.md...")
    
    # Check for "100% coverage" claims
    if overall_coverage_percent == 100:
        print("VERIFIED: 100% edge type coverage achieved")
    else:
        print(f"DISPUTED: Coverage is {overall_coverage_percent:.1f}%, NOT 100%")
    
    # Check for "19/19" claims
    if len(schema_edge_types) == 19:
        if len(all_found_edge_types) == 19:
            print("VERIFIED: 19/19 edge types extracted")
        else:
            print(f"DISPUTED: Only {len(all_found_edge_types)}/19 edge types extracted")
    else:
        print(f"DISPUTED: Schema defines {len(schema_edge_types)} edge types, not 19")
    
    # 7. Critical Assessment
    print(f"\n7. CRITICAL ASSESSMENT")
    print("-" * 30)
    
    if overall_coverage_percent >= 95:
        print("ASSESSMENT: Excellent coverage achieved")
    elif overall_coverage_percent >= 80:
        print("ASSESSMENT: Good coverage but improvement needed")
    elif overall_coverage_percent >= 60:
        print("ASSESSMENT: Moderate coverage, significant gaps remain")
    else:
        print("ASSESSMENT: Poor coverage, major improvements required")
    
    # 8. Evidence Summary
    print(f"\n8. EVIDENCE SUMMARY")
    print("-" * 25)
    
    print(f"Schema edge types: {len(schema_edge_types)}")
    print(f"Extracted edge types: {len(all_found_edge_types)}")
    print(f"Missing edge types: {len(missing_edge_types)}")
    print(f"Coverage percentage: {overall_coverage_percent:.1f}%")
    print(f"Best single file: {best_coverage_percent:.1f}%")
    
    return {
        'schema_count': len(schema_edge_types),
        'extracted_count': len(all_found_edge_types),
        'missing_count': len(missing_edge_types),
        'coverage_percent': overall_coverage_percent,
        'best_single_coverage': best_coverage_percent,
        'missing_types': sorted(missing_edge_types)
    }

if __name__ == "__main__":
    results = validate_coverage_claims()
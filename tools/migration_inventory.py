#!/usr/bin/env python3
"""
Migration Inventory Tracker for Phase 25B

Tracks migration progress and provides systematic approach to completing
all hardcoded edge type migrations.
"""

import json
from datetime import datetime

# Migration inventory based on Phase 25B file discovery
MIGRATION_INVENTORY = {
    "metadata": {
        "created": "2025-01-11",
        "phase": "25B",
        "total_files_discovered": 27,
        "search_commands_used": [
            "grep -r \"'supports'\\|'tests_hypothesis'\\|'provides_evidence_for'\\|'updates_probability'\\|'weighs_evidence'\" --include=\"*.py\" .",
            "grep -r \"edge.*type.*in \\[\" --include=\"*.py\" .",
            "grep -r \"edge\\['type'\\].*==\" --include=\"*.py\" ."
        ]
    },
    
    "already_migrated": [
        "core/analyze.py",
        "core/disconnection_repair.py", 
        "core/html_generator.py",
        "core/streaming_html.py",
        "core/van_evera_testing_engine.py",
        "core/plugins/primary_hypothesis_identifier.py",
        "core/plugins/van_evera_testing.py",
        "core/connectivity_analysis.py",  # Added in Phase 25B
        "core/plugins/evidence_connector_enhancer.py",  # Added in Phase 25B
        "process_trace_advanced.py"  # Added in Phase 25B
    ],
    
    "residual_cleanup_needed": [
        # Files that were "migrated" but still have hardcoded patterns
        "core/disconnection_repair.py",  # 14 matches in semantic patterns
        "core/van_evera_testing_engine.py",  # 1 residual hardcoded list
        "core/analyze.py",  # 1 residual match
        "core/streaming_html.py"  # 1 residual in comment
    ],
    
    "priority_levels": {
        "P0_critical": [
            {
                "file": "core/connectivity_analysis.py",
                "matches": 2,
                "description": "Line 22 has hardcoded edge type list",
                "criticality": "Main execution path"
            }
        ],
        
        "P1_high": [
            {
                "file": "core/plugins/evidence_connector_enhancer.py", 
                "matches": 1,
                "description": "Hardcoded 'supports' edge type",
                "criticality": "Plugin module"
            },
            {
                "file": "core/plugins/content_based_diagnostic_classifier.py",
                "matches": 1, 
                "description": "Hardcoded patterns in classifier",
                "criticality": "Plugin module"
            },
            {
                "file": "process_trace_advanced.py",
                "matches": 2,
                "description": "Hardcoded edge types in advanced processing",
                "criticality": "Advanced features"
            }
        ],
        
        "P2_medium": [
            {
                "file": "tests/test_dag_analysis.py",
                "matches": 1,
                "description": "Test data with hardcoded edge types",
                "criticality": "Test file"
            },
            {
                "file": "tests/test_cross_domain.py", 
                "matches": 2,
                "description": "Test data with hardcoded edge types",
                "criticality": "Test file"
            },
            {
                "file": "tests/plugins/test_van_evera_testing.py",
                "matches": 2,
                "description": "Plugin test data", 
                "criticality": "Test file"
            },
            {
                "file": "tests/plugins/test_evidence_connector_enhancer.py",
                "matches": 5,
                "description": "Plugin test data",
                "criticality": "Test file"
            },
            {
                "file": "tests/plugins/test_alternative_hypothesis_generator.py",
                "matches": 2,
                "description": "Plugin test data",
                "criticality": "Test file"
            },
            {
                "file": "tests/plugins/test_content_based_diagnostic_classifier.py",
                "matches": 9,
                "description": "Plugin test data",
                "criticality": "Test file"
            },
            {
                "file": "tests/plugins/test_primary_hypothesis_identifier.py",
                "matches": 7,
                "description": "Plugin test data",
                "criticality": "Test file"
            },
            {
                "file": "tests/test_van_evera_bayesian_integration.py",
                "matches": 1,
                "description": "Bayesian integration test",
                "criticality": "Test file"
            }
        ],
        
        "P3_low": [
            {
                "file": "docs/testing/test_all_critical_fixes.py",
                "matches": 1,
                "description": "Documentation test file",
                "criticality": "Example/documentation"
            },
            {
                "file": "docs/testing/test_critical_bug_21.py",
                "matches": 1,
                "description": "Bug test documentation",
                "criticality": "Example/documentation"
            },
            {
                "file": "docs/testing/test_critical_bug_34.py",
                "matches": 1,
                "description": "Bug test documentation", 
                "criticality": "Example/documentation"
            },
            {
                "file": "docs/testing/test_focused_extraction.py",
                "matches": 1,
                "description": "Extraction test documentation",
                "criticality": "Example/documentation"
            },
            {
                "file": "docs/testing/test_phase2b_integration.py",
                "matches": 2,
                "description": "Integration test documentation",
                "criticality": "Example/documentation"
            },
            {
                "file": "docs/testing/manual_analysis_test.py",
                "matches": 1,
                "description": "Manual analysis test",
                "criticality": "Example/documentation"
            },
            {
                "file": "docs/testing/test_critical_bug_16.py",
                "matches": 1,
                "description": "Bug test documentation",
                "criticality": "Example/documentation"  
            },
            {
                "file": "docs/testing/test_direct_integration.py",
                "matches": 1,
                "description": "Direct integration test",
                "criticality": "Example/documentation"
            }
        ]
    },
    
    "ignore_files": [
        "core/ontology_manager.py",  # Expected to have edge types in docstrings
        "tools/migrate_ontology.py",  # Migration tool itself
        "tests/test_ontology_manager.py",  # OntologyManager test data
        "test_env/lib/python3.12/site-packages/",  # Third-party libraries
        "core/plugins/content_based_diagnostic_classifier.py"  # Semantic patterns, not edge types
    ],
    
    "migration_status": {
        "completed": 0,
        "in_progress": 0,
        "remaining": 24  # Total files needing migration
    }
}

def get_next_priority_file():
    """Get the next file to migrate based on priority."""
    for priority in ["P0_critical", "P1_high", "P2_medium", "P3_low"]:
        for file_info in MIGRATION_INVENTORY["priority_levels"][priority]:
            # Check if file has been migrated
            if file_info["file"] not in get_completed_files():
                return file_info
    return None

def get_completed_files():
    """Get list of all completed migrations."""
    return MIGRATION_INVENTORY["already_migrated"]

def mark_file_completed(filepath):
    """Mark a file as migration complete."""
    if filepath not in MIGRATION_INVENTORY["already_migrated"]:
        MIGRATION_INVENTORY["already_migrated"].append(filepath)
        MIGRATION_INVENTORY["migration_status"]["completed"] += 1
        MIGRATION_INVENTORY["migration_status"]["remaining"] -= 1

def save_inventory():
    """Save current inventory state."""
    with open('tools/migration_inventory.json', 'w') as f:
        json.dump(MIGRATION_INVENTORY, f, indent=2)

def print_status():
    """Print current migration status."""
    total = MIGRATION_INVENTORY["metadata"]["total_files_discovered"] - len(MIGRATION_INVENTORY["ignore_files"])
    completed = len(MIGRATION_INVENTORY["already_migrated"])
    remaining = total - completed
    
    print(f"Migration Progress:")
    print(f"  Completed: {completed}")
    print(f"  Remaining: {remaining}")
    print(f"  Total: {total}")
    print(f"  Completion: {(completed/total)*100:.1f}%")
    
    next_file = get_next_priority_file()
    if next_file:
        print(f"\nNext to migrate: {next_file['file']} (Priority: {next_file.get('criticality', 'Unknown')})")

if __name__ == "__main__":
    print_status()
    save_inventory()
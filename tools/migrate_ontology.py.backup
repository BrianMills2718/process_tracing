#!/usr/bin/env python3
"""
Ontology Migration Tool for Phase 25A Refactoring

This tool migrates existing JSON graph files to use the new consolidated ontology,
particularly handling the consolidation of redundant Evidence→Hypothesis edge types.

Usage:
    python migrate_ontology.py <input_file.json> <output_file.json>
    python migrate_ontology.py --directory <input_dir> <output_dir>
    python migrate_ontology.py --dry-run <input_file.json>
"""

import json
import os
import sys
import argparse
import shutil
from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ontology_manager import ontology_manager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class OntologyMigrator:
    """Handles migration of graph data to new ontology structure."""
    
    def __init__(self, consolidate_edges: bool = False):
        """
        Initialize the migrator.
        
        Args:
            consolidate_edges: If True, consolidates redundant edge types
        """
        self.consolidate_edges = consolidate_edges
        self.migration_stats = {
            'files_processed': 0,
            'edges_migrated': 0,
            'edges_consolidated': 0,
            'errors': []
        }
        
        # Define edge type consolidation mappings
        # Based on Phase 24A findings, these redundant edges should be consolidated
        self.edge_consolidation_map = {
            'provides_evidence_for': 'tests_hypothesis',
            'updates_probability': 'tests_hypothesis',
            'weighs_evidence': 'tests_hypothesis',
            'supports': 'tests_hypothesis',
            # Keep tests_hypothesis as the canonical form
            'tests_hypothesis': 'tests_hypothesis'
        }
    
    def migrate_edge_types(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate edge types in graph data to new ontology.
        
        Args:
            graph_data: Graph data dictionary with 'nodes' and 'edges'
            
        Returns:
            Migrated graph data
        """
        migrated_data = graph_data.copy()
        
        for edge in migrated_data.get('edges', []):
            old_type = edge.get('type')
            
            # Check if this edge needs migration
            if self.consolidate_edges and old_type in self.edge_consolidation_map:
                new_type = self.edge_consolidation_map[old_type]
                
                if old_type != new_type:
                    # Preserve original type for reference
                    edge['_original_type'] = old_type
                    edge['_migration_timestamp'] = datetime.now().isoformat()
                    edge['type'] = new_type
                    
                    # Ensure diagnostic_type is set if not present
                    if 'properties' not in edge:
                        edge['properties'] = {}
                    if 'diagnostic_type' not in edge['properties']:
                        # Infer diagnostic type based on original edge type
                        if old_type == 'supports':
                            edge['properties']['diagnostic_type'] = 'straw_in_wind'
                        elif old_type == 'provides_evidence_for':
                            edge['properties']['diagnostic_type'] = 'general'
                        else:
                            edge['properties']['diagnostic_type'] = 'general'
                    
                    self.migration_stats['edges_consolidated'] += 1
                    logger.debug(f"Consolidated edge type: {old_type} → {new_type}")
            
            self.migration_stats['edges_migrated'] += 1
        
        # Add migration metadata
        if '_metadata' not in migrated_data:
            migrated_data['_metadata'] = {}
        
        migrated_data['_metadata']['migration'] = {
            'timestamp': datetime.now().isoformat(),
            'tool_version': '1.0.0',
            'consolidation_applied': self.consolidate_edges,
            'edges_migrated': self.migration_stats['edges_migrated'],
            'edges_consolidated': self.migration_stats['edges_consolidated']
        }
        
        return migrated_data
    
    def validate_migration(self, original: Dict[str, Any], migrated: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that migration preserved essential structure.
        
        Args:
            original: Original graph data
            migrated: Migrated graph data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check node count
        original_nodes = len(original.get('nodes', []))
        migrated_nodes = len(migrated.get('nodes', []))
        if original_nodes != migrated_nodes:
            issues.append(f"Node count mismatch: {original_nodes} → {migrated_nodes}")
        
        # Check edge count
        original_edges = len(original.get('edges', []))
        migrated_edges = len(migrated.get('edges', []))
        if original_edges != migrated_edges:
            issues.append(f"Edge count mismatch: {original_edges} → {migrated_edges}")
        
        # Validate all edges against ontology
        for edge in migrated.get('edges', []):
            is_valid, error = ontology_manager.validate_edge(edge)
            if not is_valid:
                issues.append(f"Edge validation failed: {error}")
        
        # Check that all nodes are preserved
        original_ids = {n['id'] for n in original.get('nodes', [])}
        migrated_ids = {n['id'] for n in migrated.get('nodes', [])}
        
        missing_nodes = original_ids - migrated_ids
        if missing_nodes:
            issues.append(f"Missing nodes after migration: {missing_nodes}")
        
        extra_nodes = migrated_ids - original_ids
        if extra_nodes:
            issues.append(f"Extra nodes after migration: {extra_nodes}")
        
        return len(issues) == 0, issues
    
    def migrate_file(self, input_path: str, output_path: str, dry_run: bool = False) -> bool:
        """
        Migrate a single JSON file.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            dry_run: If True, only validate without writing
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Processing: {input_path}")
            
            # Load original data
            with open(input_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Migrate the data
            migrated_data = self.migrate_edge_types(original_data.copy())
            
            # Validate migration
            is_valid, issues = self.validate_migration(original_data, migrated_data)
            
            if not is_valid:
                logger.warning(f"Validation issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                
                if not dry_run:
                    # Ask for confirmation on validation issues
                    response = input("Continue with migration despite issues? (y/n): ")
                    if response.lower() != 'y':
                        logger.info("Migration cancelled")
                        return False
            
            if dry_run:
                logger.info(f"Dry run complete. Would migrate {self.migration_stats['edges_migrated']} edges")
                if self.consolidate_edges:
                    logger.info(f"Would consolidate {self.migration_stats['edges_consolidated']} edges")
            else:
                # Create output directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Write migrated data
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(migrated_data, f, indent=2)
                
                logger.info(f"Migration complete: {output_path}")
                logger.info(f"  - Edges migrated: {self.migration_stats['edges_migrated']}")
                if self.consolidate_edges:
                    logger.info(f"  - Edges consolidated: {self.migration_stats['edges_consolidated']}")
            
            self.migration_stats['files_processed'] += 1
            return True
            
        except Exception as e:
            error_msg = f"Error processing {input_path}: {str(e)}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def migrate_directory(self, input_dir: str, output_dir: str, dry_run: bool = False) -> bool:
        """
        Migrate all JSON files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            dry_run: If True, only validate without writing
            
        Returns:
            True if all migrations successful
        """
        success = True
        
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.json'):
                    input_path = os.path.join(root, filename)
                    
                    # Compute relative path for output
                    rel_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, rel_path)
                    
                    if not self.migrate_file(input_path, output_path, dry_run):
                        success = False
        
        return success
    
    def print_summary(self):
        """Print migration summary statistics."""
        print("\n" + "="*50)
        print("MIGRATION SUMMARY")
        print("="*50)
        print(f"Files processed: {self.migration_stats['files_processed']}")
        print(f"Total edges migrated: {self.migration_stats['edges_migrated']}")
        
        if self.consolidate_edges:
            print(f"Edges consolidated: {self.migration_stats['edges_consolidated']}")
        
        if self.migration_stats['errors']:
            print(f"\nErrors encountered: {len(self.migration_stats['errors'])}")
            for error in self.migration_stats['errors']:
                print(f"  - {error}")
        else:
            print("\nNo errors encountered")


def main():
    """Main entry point for the migration tool."""
    parser = argparse.ArgumentParser(
        description='Migrate process tracing JSON files to new ontology structure'
    )
    
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('output', nargs='?', help='Output file or directory')
    parser.add_argument('--directory', '-d', action='store_true',
                       help='Process entire directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate without writing files')
    parser.add_argument('--consolidate', '-c', action='store_true',
                       help='Consolidate redundant edge types')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize migrator
    migrator = OntologyMigrator(consolidate_edges=args.consolidate)
    
    # Determine output path
    if args.dry_run:
        output_path = None
    elif not args.output:
        # Generate output path
        if args.directory:
            output_path = args.input + '_migrated'
        else:
            base, ext = os.path.splitext(args.input)
            output_path = f"{base}_migrated{ext}"
    else:
        output_path = args.output
    
    # Perform migration
    if args.directory:
        if not os.path.isdir(args.input):
            logger.error(f"Input directory does not exist: {args.input}")
            sys.exit(1)
        
        success = migrator.migrate_directory(args.input, output_path, args.dry_run)
    else:
        if not os.path.isfile(args.input):
            logger.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        
        success = migrator.migrate_file(args.input, output_path, args.dry_run)
    
    # Print summary
    migrator.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
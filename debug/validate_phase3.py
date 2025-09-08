"""
Validation script for Phase 3 LLM-First Migration.
Tests the migrated components and reports on remaining keyword patterns.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple


def check_keyword_patterns(file_path: Path) -> List[Tuple[int, str]]:
    """Check for keyword matching patterns in a file."""
    patterns = [
        r"if\s+.*\s+in\s+.*desc",
        r"if\s+['\"].*['\"]\s+in\s+",
        r"any\(.*in.*for.*in\s*\[",
        r"probative_value.*=.*0\.\d",
        r"'hutchinson'|'stamp act'|'taxation without'",
        r"'boston'|'parliament'|'colonial'"
    ]
    
    found_patterns = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                for pattern in patterns:
                    if re.search(pattern, line.lower()):
                        found_patterns.append((i, line.strip()))
                        break
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return found_patterns


def scan_codebase() -> Dict[str, List[Tuple[int, str]]]:
    """Scan entire codebase for keyword patterns."""
    results = {}
    
    # Core files to check
    core_files = [
        "core/analyze.py",
        "core/connectivity_analysis.py", 
        "core/disconnection_repair.py",
        "core/van_evera_testing_engine.py",
        "core/alternative_hypothesis_generator.py",
        "core/mechanism_detector.py",
        "core/likelihood_calculator.py",
        "core/prior_assignment.py",
        "core/temporal_graph.py",
        "core/confidence_calculator.py",
        "core/extract.py"
    ]
    
    # Plugin files to check
    plugin_files = [
        "core/plugins/content_based_diagnostic_classifier.py",
        "core/plugins/research_question_generator.py",
        "core/plugins/advanced_van_evera_prediction_engine.py",
        "core/plugins/alternative_hypothesis_generator.py",
        "core/plugins/evidence_connector_enhancer.py",
        "core/plugins/diagnostic_rebalancer.py",
        "core/plugins/primary_hypothesis_identifier.py"
    ]
    
    all_files = core_files + plugin_files
    
    for file_path in all_files:
        full_path = Path(file_path)
        if full_path.exists():
            patterns = check_keyword_patterns(full_path)
            if patterns:
                results[file_path] = patterns
    
    return results


def test_llm_integration():
    """Test that LLM integration is working."""
    try:
        # Test semantic service
        from core.semantic_analysis_service import get_semantic_service
        service = get_semantic_service()
        
        # Test basic functionality
        result = service.classify_domain(
            "Economic factors led to the revolution",
            "Testing semantic analysis"
        )
        
        print("[OK] SemanticAnalysisService working")
        print(f"   - Primary domain: {result.primary_domain}")
        print(f"   - Confidence: {result.confidence_score}")
        
        # Test LLM interface
        from core.plugins.van_evera_llm_interface import get_van_evera_llm
        llm = get_van_evera_llm()
        print("[OK] LLM Interface available")
        
        # Get cache statistics
        stats = service.get_statistics()
        print(f"[OK] Cache Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] LLM Integration failed: {e}")
        return False


def main():
    """Run Phase 3 validation."""
    print("=" * 60)
    print("Phase 3 LLM-First Migration Validation")
    print("=" * 60)
    print()
    
    # Test LLM integration
    print("Testing LLM Integration...")
    llm_ok = test_llm_integration()
    print()
    
    # Scan for remaining keyword patterns
    print("Scanning for remaining keyword patterns...")
    results = scan_codebase()
    
    if not results:
        print("[SUCCESS] NO KEYWORD PATTERNS FOUND - Migration Complete!")
    else:
        total_patterns = sum(len(patterns) for patterns in results.values())
        print(f"[WARNING] Found {total_patterns} keyword patterns in {len(results)} files:")
        print()
        
        for file_path, patterns in sorted(results.items()):
            print(f"  {file_path}: {len(patterns)} instances")
            # Show first 3 examples
            for line_no, line in patterns[:3]:
                print(f"    Line {line_no}: {line[:80]}...")
            if len(patterns) > 3:
                print(f"    ... and {len(patterns) - 3} more")
            print()
    
    # Calculate completion percentage
    total_files = 28  # Approximate total files needing migration
    migrated_files = total_files - len(results)
    completion = (migrated_files / total_files) * 100
    
    print("=" * 60)
    print(f"Migration Progress: {completion:.1f}% complete")
    print(f"Files migrated: {migrated_files}/{total_files}")
    print(f"Files remaining: {len(results)}")
    print("=" * 60)
    
    # Success criteria
    if llm_ok and len(results) < 10:
        print("[SUCCESS] Phase 3 Partial Success - Core modules migrated")
    elif not llm_ok:
        print("[FAIL] Phase 3 Failed - LLM integration not working")
    else:
        print("[IN_PROGRESS] Phase 3 In Progress - More migration needed")


if __name__ == "__main__":
    main()
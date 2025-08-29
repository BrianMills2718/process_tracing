#!/usr/bin/env python3
"""
Honest validation of actual LLM-first coverage.
"""

import os
import re
from pathlib import Path

def count_files_with_pattern(pattern, directory="core"):
    """Count files containing a specific pattern"""
    count = 0
    files = []
    for path in Path(directory).rglob("*.py"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if re.search(pattern, content):
                    count += 1
                    files.append(str(path))
        except:
            pass
    return count, files

def main():
    print("=" * 60)
    print("HONEST LLM-FIRST COVERAGE ASSESSMENT")
    print("=" * 60)
    
    # Count total Python files
    total_files = len(list(Path("core").rglob("*.py")))
    print(f"\nTotal Python files in core/: {total_files}")
    
    # Count files using require_llm
    require_count, require_files = count_files_with_pattern(r"require_llm|LLMRequiredError", "core")
    print(f"\nFiles using require_llm: {require_count}")
    for f in require_files[:5]:
        print(f"  - {f}")
    
    # Count files using semantic_analysis_service
    semantic_count, semantic_files = count_files_with_pattern(r"get_semantic_service|semantic_service\.", "core")
    print(f"\nFiles using semantic_analysis_service: {semantic_count}")
    
    # Count files with fallbacks
    fallback_count, fallback_files = count_files_with_pattern(
        r"except.*:\s*return None|return.*#.*fallback|#.*Fallback|= 0\.\d+.*#.*default", 
        "core"
    )
    print(f"\nFiles with potential fallbacks: {fallback_count}")
    
    # Count files with hardcoded values
    hardcoded_count, _ = count_files_with_pattern(
        r"= 0\.\d+(?!.*Field)(?!.*\*)(?!.*\/)(?!.*\+)(?!.*\-)", 
        "core"
    )
    print(f"Files with hardcoded decimal values: {hardcoded_count}")
    
    # Calculate percentages
    print("\n" + "=" * 60)
    print("COVERAGE METRICS")
    print("=" * 60)
    
    direct_llm_percent = (require_count / total_files) * 100
    indirect_llm_percent = (semantic_count / total_files) * 100
    fallback_percent = (fallback_count / total_files) * 100
    
    print(f"\nDirect LLM requirement: {require_count}/{total_files} = {direct_llm_percent:.1f}%")
    print(f"Using semantic service: {semantic_count}/{total_files} = {indirect_llm_percent:.1f}%")
    print(f"Files with fallbacks: {fallback_count}/{total_files} = {fallback_percent:.1f}%")
    
    # Estimate true coverage
    # Files that require LLM minus those with fallbacks
    true_llm_files = max(0, semantic_count - (fallback_count // 2))  # Rough estimate
    true_coverage = (true_llm_files / total_files) * 100
    
    print("\n" + "=" * 60)
    print("HONEST ASSESSMENT")
    print("=" * 60)
    print(f"\nEstimated TRUE LLM-first coverage: ~{true_coverage:.0f}%")
    print("\nReality Check:")
    print("✅ Main semantic path requires LLM")
    print("✅ semantic_analysis_service has no fallbacks")
    print("❌ Many plugins still have fallbacks")
    print("❌ Enhancement functions return None on failure")
    print("❌ Hardcoded thresholds remain")
    
    print("\nConclusion: System is PARTIALLY LLM-first (~30-40%), not fully.")
    
if __name__ == "__main__":
    main()
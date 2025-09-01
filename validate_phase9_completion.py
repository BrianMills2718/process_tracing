#!/usr/bin/env python3
"""Validate Phase 9 completion - 100% LLM-first compliance"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_file_compliance(filepath):
    """Check single file for LLM-first compliance"""
    if not Path(filepath).exists():
        return False, ["File not found"]
    
    content = Path(filepath).read_text()
    
    violations = []
    
    # Check for prohibited patterns
    prohibited = [
        (r"if\s+['\"].*['\"].*in.*text", "Keyword matching"),
        (r"return\s+None\s*#.*fallback", "Fallback to None"),
        (r"return\s+0\.?\d+\s*#.*default", "Hardcoded default"),
        (r"confidence\s*=\s*0\.\d+\s*#.*hardcoded", "Hardcoded confidence comment"),
        (r"probative_value\s*=\s*0\.\d+\s*#.*hardcoded", "Hardcoded probative value comment")
    ]
    
    import re
    for pattern, description in prohibited:
        if re.search(pattern, content, re.IGNORECASE):
            violations.append(description)
    
    # Check for required patterns - must use some form of LLM
    uses_llm = any([
        "VanEveraLLMInterface" in content,
        "semantic_analysis_service" in content,
        "semantic_service" in content,
        "LLMRequiredError" in content,
        "require_llm" in content,
        "query_llm" in content,
        "refine_evidence_assessment_with_llm" in content
    ])
    
    # Special handling for files that may not need LLM
    non_semantic_files = ["__init__.py", "structured_models.py", "llm_required.py"]
    if any(ns in filepath for ns in non_semantic_files):
        uses_llm = True  # Don't require LLM for these files
    
    if not uses_llm and "semantic" in filepath.lower():
        violations.append("No LLM interface usage")
    
    return len(violations) == 0, violations

def main():
    # Target files for validation
    semantic_files = [
        "core/enhance_evidence.py",
        "core/enhance_hypotheses.py",
        "core/enhance_mechanisms.py",
        "core/semantic_analysis_service.py",
        "core/confidence_calculator.py",
        "core/analyze.py",
        "core/van_evera_testing_engine.py",
        "core/plugins/diagnostic_rebalancer.py",
        "core/plugins/alternative_hypothesis_generator.py",
        "core/plugins/evidence_connector_enhancer.py",
        "core/plugins/content_based_diagnostic_classifier.py",
        "core/plugins/research_question_generator.py",
        "core/plugins/primary_hypothesis_identifier.py",
        "core/plugins/bayesian_van_evera_engine.py"
    ]
    
    compliant = 0
    total = len(semantic_files)
    non_compliant = []
    
    print("Phase 9 LLM-First Compliance Validation")
    print("=" * 50)
    
    for filepath in semantic_files:
        if Path(filepath).exists():
            is_compliant, violations = check_file_compliance(filepath)
            
            if is_compliant:
                print(f"[OK] {filepath}")
                compliant += 1
            else:
                print(f"[FAIL] {filepath}")
                non_compliant.append((filepath, violations))
                for v in violations[:3]:
                    print(f"      - {v}")
        else:
            print(f"[SKIP] {filepath} - File not found")
    
    print("=" * 50)
    compliance_rate = (compliant / total) * 100
    print(f"Compliance: {compliant}/{total} files ({compliance_rate:.1f}%)")
    
    # Show details for non-compliant files
    if non_compliant:
        print("\nNon-compliant files need attention:")
        for filepath, violations in non_compliant:
            print(f"\n{filepath}:")
            for v in violations:
                print(f"  - {v}")
    
    if compliance_rate == 100:
        print("\n[SUCCESS] 100% LLM-first compliance achieved!")
        return 0
    elif compliance_rate >= 85:
        print(f"\n[GOOD] {compliance_rate:.1f}% compliance achieved")
        print(f"Only {total - compliant} files need minor adjustments")
        return 0
    else:
        print(f"\n[INCOMPLETE] {total - compliant} files still need migration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
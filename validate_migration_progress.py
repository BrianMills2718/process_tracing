#!/usr/bin/env python3
"""
Validate Phase 8 Week 2 Migration Progress
Checks for LLM-first compliance across the codebase
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_file_for_fallbacks(file_path: Path) -> Tuple[bool, List[str]]:
    """Check a file for non-LLM fallback patterns"""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Pattern checks
    patterns = [
        (r'return\s+None\s*#.*fallback', 'Returns None as fallback'),
        (r'return\s+0\.?\d*\s*#.*default', 'Returns hardcoded default value'),
        (r'if.*keyword.*in.*text', 'Keyword matching for semantic decisions'),
        (r'if\s+["\']before["\']\s+in', 'Temporal keyword matching'),
        (r'probative_value\s*=\s*0\.\d+', 'Hardcoded probative value'),
        (r'confidence\s*=\s*0\.\d+\s*#.*hardcoded', 'Hardcoded confidence'),
        (r'except.*:\s*return\s+None', 'Returns None on exception'),
        (r'except.*:\s*pass', 'Silent exception handling'),
        (r'semantic.*=.*if.*else.*0', 'Conditional fallback to 0'),
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern, desc in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(f"Line {i}: {desc}")
    
    # Check for LLM Gateway usage (positive signal)
    uses_gateway = 'LLMGateway' in content or 'llm_gateway' in content
    uses_llm_required = 'LLMRequiredError' in content or 'require_llm' in content
    
    is_compliant = len(issues) == 0 and (uses_gateway or uses_llm_required)
    
    return is_compliant, issues

def check_semantic_files() -> Dict[str, Dict]:
    """Check all semantic files for LLM-first compliance"""
    
    # Semantic files that should use LLM
    semantic_files = [
        'core/enhance_evidence.py',
        'core/enhance_hypotheses.py',
        'core/enhance_mechanisms.py',
        'core/semantic_analysis_service.py',
        'core/confidence_calculator.py',
        'core/analyze.py',
        'core/plugins/van_evera_testing_engine.py',
        'core/plugins/diagnostic_rebalancer.py',
        'core/plugins/alternative_hypothesis_generator.py',
        'core/plugins/evidence_connector_enhancer.py',
        'core/plugins/content_based_diagnostic_classifier.py',
        'core/plugins/research_question_generator.py',
        'core/plugins/primary_hypothesis_identifier.py',
        'core/plugins/bayesian_van_evera_engine.py',
    ]
    
    results = {}
    
    for file_path in semantic_files:
        path = Path(file_path)
        if path.exists():
            is_compliant, issues = check_file_for_fallbacks(path)
            results[file_path] = {
                'exists': True,
                'compliant': is_compliant,
                'issues': issues,
                'uses_gateway': 'LLMGateway' in open(path).read()
            }
        else:
            results[file_path] = {
                'exists': False,
                'compliant': False,
                'issues': ['File not found'],
                'uses_gateway': False
            }
    
    return results

def calculate_metrics(results: Dict) -> Dict:
    """Calculate migration metrics"""
    
    total_files = len(results)
    compliant_files = sum(1 for r in results.values() if r['compliant'])
    gateway_files = sum(1 for r in results.values() if r['uses_gateway'])
    files_with_issues = sum(1 for r in results.values() if len(r['issues']) > 0)
    
    return {
        'total_files': total_files,
        'compliant_files': compliant_files,
        'gateway_files': gateway_files,
        'files_with_issues': files_with_issues,
        'compliance_rate': (compliant_files / total_files * 100) if total_files > 0 else 0,
        'gateway_adoption': (gateway_files / total_files * 100) if total_files > 0 else 0
    }

def main():
    print("=" * 60)
    print("PHASE 8 MIGRATION VALIDATION")
    print("=" * 60)
    
    # Check semantic files
    print("\n[1] Checking semantic files for LLM-first compliance...")
    results = check_semantic_files()
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Display results
    print(f"\n[2] Results Summary:")
    print(f"   Total files checked: {metrics['total_files']}")
    print(f"   Compliant files: {metrics['compliant_files']}")
    print(f"   Files using gateway: {metrics['gateway_files']}")
    print(f"   Files with issues: {metrics['files_with_issues']}")
    print(f"   Compliance rate: {metrics['compliance_rate']:.1f}%")
    print(f"   Gateway adoption: {metrics['gateway_adoption']:.1f}%")
    
    # Show details for non-compliant files
    print(f"\n[3] Non-compliant files:")
    for file_path, result in results.items():
        if not result['compliant'] and result['exists']:
            print(f"\n   {file_path}:")
            for issue in result['issues'][:3]:  # Show first 3 issues
                print(f"      - {issue}")
            if len(result['issues']) > 3:
                print(f"      ... and {len(result['issues']) - 3} more issues")
    
    # Files successfully migrated
    print(f"\n[4] Successfully migrated files:")
    for file_path, result in results.items():
        if result['compliant']:
            status = "[GATEWAY]" if result['uses_gateway'] else "[LLM-FIRST]"
            print(f"   [OK] {status} {file_path}")
    
    # Overall verdict
    print("\n" + "=" * 60)
    if metrics['compliance_rate'] >= 80:
        print("[SUCCESS] Migration is on track!")
        print(f"Achieved {metrics['compliance_rate']:.1f}% LLM-first compliance")
    elif metrics['compliance_rate'] >= 50:
        print("[PROGRESS] Migration is making progress")
        print(f"Currently at {metrics['compliance_rate']:.1f}% compliance")
    else:
        print("[WARNING] Migration needs acceleration")
        print(f"Only {metrics['compliance_rate']:.1f}% compliance achieved")
    
    # Recommendations
    print("\n[5] Recommendations:")
    if metrics['gateway_adoption'] < 50:
        print("   - Increase gateway adoption (currently {:.1f}%)".format(metrics['gateway_adoption']))
    if metrics['files_with_issues'] > 5:
        print(f"   - Fix fallback patterns in {metrics['files_with_issues']} files")
    if metrics['compliant_files'] < metrics['total_files']:
        print(f"   - Migrate {metrics['total_files'] - metrics['compliant_files']} remaining files")
    
    return 0 if metrics['compliance_rate'] >= 50 else 1

if __name__ == "__main__":
    sys.exit(main())
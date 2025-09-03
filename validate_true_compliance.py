#!/usr/bin/env python3
"""
Comprehensive LLM-First Compliance Validator
Validates TRUE 100% compliance with zero tolerance for keyword matching or fallbacks
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ComplianceValidator:
    """Comprehensive validator for LLM-first architecture compliance"""
    
    def __init__(self):
        self.violations = []
        self.files_checked = 0
        self.compliant_files = 0
        
        # Comprehensive patterns for keyword matching violations
        self.keyword_patterns = [
            # Direct keyword matching
            (r"if\s+['\"].*['\"].*in\s+.*text", "Direct keyword in text matching"),
            (r"if\s+.*keyword.*in", "Keyword variable matching"),
            (r"if\s+.*ideological.*in", "Domain keyword: ideological"),
            (r"if\s+.*temporal.*in", "Domain keyword: temporal"),
            (r"if\s+.*causal.*in.*text", "Domain keyword: causal"),
            (r"if\s+.*revolutionary.*in", "Dataset-specific: revolutionary"),
            (r"if\s+.*resistance.*in", "Dataset-specific: resistance"),
            (r"if\s+'early'\s+in", "Temporal keyword: early"),
            (r"if\s+'late'\s+in", "Temporal keyword: late"),
            (r"if\s+'before'\s+in", "Temporal keyword: before"),
            (r"if\s+'after'\s+in", "Temporal keyword: after"),
            
            # Hardcoded values
            (r"probative_value\s*=\s*0\.\d+", "Hardcoded probative value"),
            (r"confidence\s*=\s*0\.\d+", "Hardcoded confidence"),
            (r"return\s+0\.\d+\s*#.*default", "Hardcoded default return"),
            (r"threshold\s*=\s*0\.\d+", "Hardcoded threshold"),
            
            # String contains checks
            (r"\.lower\(\).*in\s+.*\.lower\(\)", "Case-insensitive keyword matching"),
            (r"any\(\[.*in.*for.*in.*\]\)", "List comprehension keyword matching"),
            
            # Dataset-specific logic
            (r"if.*american.*revolution", "Dataset-specific: American Revolution"),
            (r"if.*colonial.*period", "Dataset-specific: Colonial period"),
        ]
        
        # Patterns for fail-fast violations
        self.fail_fast_patterns = [
            (r"return\s+None\s*$", "Returns None instead of raising error"),
            (r"return\s+None\s*#", "Returns None with comment"),
            (r"except.*:\s*return\s+None", "Exception returns None"),
            (r"except.*:\s*pass", "Silent exception swallowing"),
        ]
        
        # Files that should use LLM
        self.semantic_files = [
            "enhance_evidence.py",
            "enhance_hypotheses.py", 
            "enhance_mechanisms.py",
            "semantic_analysis_service.py",
            "confidence_calculator.py",
            "analyze.py",
            "diagnostic_rebalancer.py",
            "alternative_hypothesis_generator.py",
            "evidence_connector_enhancer.py",
            "content_based_diagnostic_classifier.py",
            "research_question_generator.py",
            "primary_hypothesis_identifier.py",
            "bayesian_van_evera_engine.py",
            "advanced_van_evera_prediction_engine.py",
            "temporal_extraction.py",
            "legacy_compatibility_manager.py"
        ]
        
    def check_file_compliance(self, filepath: Path) -> Tuple[bool, List[str]]:
        """Check single file for compliance"""
        if not filepath.exists():
            return False, ["File not found"]
        
        try:
            content = filepath.read_text()
        except Exception as e:
            return False, [f"Could not read file: {e}"]
        
        violations = []
        filename = filepath.name
        
        # Check for keyword matching violations
        for pattern, description in self.keyword_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append(f"Line {line_num}: {description}")
                
        # Check for fail-fast violations in semantic files
        if any(sf in str(filepath) for sf in self.semantic_files):
            # Check if file has LLM-related code
            if any(term in content for term in ['llm', 'LLM', 'semantic', 'query_llm']):
                for pattern, description in self.fail_fast_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        # Check context to see if it's in LLM-related function
                        start = max(0, match.start() - 500)
                        context = content[start:match.end()]
                        if any(term in context for term in ['llm', 'LLM', 'semantic', 'query_llm']):
                            line_num = content[:match.start()].count('\n') + 1
                            violations.append(f"Line {line_num}: FAIL-FAST VIOLATION - {description}")
        
        # Check for required LLM usage in semantic files
        if filename in self.semantic_files:
            uses_llm = any([
                "VanEveraLLMInterface" in content,
                "semantic_analysis_service" in content,
                "semantic_service" in content,
                "LLMRequiredError" in content,
                "require_llm" in content,
                "query_llm" in content,
                "refine_evidence_assessment_with_llm" in content
            ])
            
            # Special exemptions
            if filename in ["__init__.py", "structured_models.py", "llm_required.py"]:
                uses_llm = True
                
            if not uses_llm:
                violations.append("NO LLM USAGE - File should use LLM for semantic analysis")
        
        return len(violations) == 0, violations
    
    def validate_directory(self, directory: str = "core") -> Dict[str, any]:
        """Validate all Python files in directory"""
        results = {
            'total_files': 0,
            'compliant_files': 0,
            'non_compliant_files': [],
            'compliance_rate': 0.0
        }
        
        # Find all Python files
        path = Path(directory)
        python_files = list(path.rglob("*.py"))
        
        # Filter out test files and __pycache__
        python_files = [
            f for f in python_files 
            if "__pycache__" not in str(f) and "test" not in str(f.parent)
        ]
        
        for filepath in python_files:
            self.files_checked += 1
            results['total_files'] += 1
            
            is_compliant, violations = self.check_file_compliance(filepath)
            
            if is_compliant:
                self.compliant_files += 1
                results['compliant_files'] += 1
            else:
                results['non_compliant_files'].append({
                    'file': str(filepath),
                    'violations': violations
                })
                
        if results['total_files'] > 0:
            results['compliance_rate'] = (results['compliant_files'] / results['total_files']) * 100
            
        return results

def main():
    """Main validation function"""
    print("=" * 80)
    print("TRUE LLM-First Compliance Validator")
    print("=" * 80)
    print()
    
    validator = ComplianceValidator()
    
    # Validate core directory
    print("Validating core/ directory...")
    core_results = validator.validate_directory("core")
    
    # Print results
    print(f"\nTotal files checked: {core_results['total_files']}")
    print(f"Compliant files: {core_results['compliant_files']}")
    print(f"Non-compliant files: {len(core_results['non_compliant_files'])}")
    print(f"Compliance rate: {core_results['compliance_rate']:.1f}%")
    
    # Show non-compliant files
    if core_results['non_compliant_files']:
        print("\n" + "=" * 80)
        print("NON-COMPLIANT FILES:")
        print("=" * 80)
        
        for file_info in core_results['non_compliant_files']:
            print(f"\n{file_info['file']}:")
            for violation in file_info['violations'][:5]:  # Show first 5 violations
                print(f"  - {violation}")
            if len(file_info['violations']) > 5:
                print(f"  ... and {len(file_info['violations']) - 5} more violations")
    
    # Determine success
    if core_results['compliance_rate'] == 100:
        print("\n✅ SUCCESS: TRUE 100% LLM-first compliance achieved!")
        return 0
    else:
        print(f"\n❌ INCOMPLETE: {len(core_results['non_compliant_files'])} files need migration")
        print("\nNext steps:")
        print("1. Fix fail-fast violations (raise LLMRequiredError instead of return None)")
        print("2. Remove all keyword matching patterns")
        print("3. Ensure semantic files use LLM for analysis")
        return 1

if __name__ == "__main__":
    sys.exit(main())
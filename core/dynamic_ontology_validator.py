"""
Dynamic Ontology Validator
==========================

Replaces hardcoded validation with functional requirement validation.
Validates what the system needs functionally, not specific edge type names.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class ValidationResult:
    """
    Structured validation outcome with detailed diagnostics
    """
    success: bool
    mode: str
    critical_passed: bool
    optional_passed: bool
    missing_critical: List[str]
    missing_optional: List[str]
    error_messages: List[str]
    recommendations: List[str]
    
    def summary(self) -> str:
        """Generate human-readable validation summary"""
        if self.success:
            return f"✅ Ontology validation passed ({self.mode} mode)"
        else:
            summary = f"❌ Ontology validation failed ({self.mode} mode)\n"
            for error in self.error_messages:
                summary += f"  • {error}\n"
            if self.recommendations:
                summary += "Recommendations:\n"
                for rec in self.recommendations:
                    summary += f"  • {rec}\n"
            return summary.strip()
    
    def is_usable(self) -> bool:
        """Whether system can operate (critical requirements met)"""
        return self.critical_passed


class DynamicOntologyValidator:
    """
    Dynamic ontology validator that checks functional capabilities,
    not hardcoded edge type names.
    """
    
    def __init__(self, ontology_manager):
        self.ontology_manager = ontology_manager
    
    def validate(self, mode: str = 'strict') -> ValidationResult:
        """
        Validate ontology against functional requirements
        
        Args:
            mode: 'strict' (all requirements), 'minimal' (critical only), 'schema-only' (JSON only)
        
        Returns:
            ValidationResult with success/failure and detailed findings
        """
        missing_critical = []
        missing_optional = []
        error_messages = []
        recommendations = []
        
        # Always validate schema
        schema_valid = self._validate_schema()
        if not schema_valid:
            missing_critical.append("Valid JSON schema structure")
            error_messages.append("Ontology JSON schema is invalid or corrupted")
            recommendations.append("Check ontology_config.json for syntax errors")
        
        # Skip functional validation in schema-only mode
        if mode == 'schema-only':
            critical_passed = schema_valid
            optional_passed = True  # Not checked in schema-only mode
        else:
            # Validate critical requirements
            critical_issues = self._validate_critical_requirements()
            missing_critical.extend(critical_issues)
            critical_passed = len(critical_issues) == 0 and schema_valid
            
            # Add error messages for critical failures
            if critical_issues:
                error_messages.extend([
                    f"Missing critical capability: {issue}" for issue in critical_issues
                ])
                recommendations.extend(self._get_critical_recommendations(critical_issues))
            
            # Validate optional requirements (unless minimal mode)
            if mode == 'minimal':
                optional_passed = True  # Not checked in minimal mode
            else:
                optional_issues = self._validate_optional_requirements()
                missing_optional.extend(optional_issues)
                optional_passed = len(optional_issues) == 0
                
                # Add warnings for optional failures
                if optional_issues:
                    error_messages.extend([
                        f"Missing optional capability: {issue}" for issue in optional_issues
                    ])
                    recommendations.extend(self._get_optional_recommendations(optional_issues))
        
        # Determine overall success
        if mode == 'strict':
            success = critical_passed and optional_passed
        elif mode == 'minimal':
            success = critical_passed
        else:  # schema-only
            success = schema_valid
        
        return ValidationResult(
            success=success,
            mode=mode,
            critical_passed=critical_passed,
            optional_passed=optional_passed,
            missing_critical=missing_critical,
            missing_optional=missing_optional,
            error_messages=error_messages,
            recommendations=recommendations
        )
    
    def _validate_schema(self) -> bool:
        """Check JSON structure integrity"""
        try:
            # Test that basic ontology manager operations work
            edge_types = self.ontology_manager.get_all_edge_types()
            node_types = self.ontology_manager.get_all_node_types()
            return len(edge_types) > 0 and len(node_types) > 0
        except Exception:
            return False
    
    def _validate_critical_requirements(self) -> List[str]:
        """Validate critical functional requirements"""
        issues = []
        
        # CRITICAL: At least one Evidence->Hypothesis connection capability
        evidence_hypothesis_edges = self.ontology_manager.get_evidence_hypothesis_edges()
        if not evidence_hypothesis_edges:
            issues.append("Evidence to Hypothesis connectivity")
        
        return issues
    
    def _validate_optional_requirements(self) -> List[str]:
        """Validate optional functional requirements"""
        issues = []
        
        # OPTIONAL: Van Evera diagnostic test capabilities
        van_evera_edges = self.ontology_manager.get_van_evera_edges()
        if not van_evera_edges:
            issues.append("Van Evera diagnostic test capabilities")
        
        # OPTIONAL: Probative value properties (check if edges have probative_value property)
        try:
            # Sample a few edge types to see if probative_value is supported
            sample_edges = self.ontology_manager.get_all_edge_types()[:3]
            has_probative_values = False
            for edge_type in sample_edges:
                try:
                    properties = self.ontology_manager.get_edge_properties(edge_type)
                    if 'probative_value' in properties:
                        has_probative_values = True
                        break
                except:
                    continue
            
            if not has_probative_values:
                issues.append("Probative value property support")
        except Exception:
            issues.append("Probative value property support")
        
        return issues
    
    def _get_critical_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations for critical issues"""
        recommendations = []
        
        for issue in issues:
            if "Evidence to Hypothesis connectivity" in issue:
                recommendations.append(
                    "Add at least one edge type that connects Evidence nodes to Hypothesis nodes "
                    "(e.g., 'supports', 'tests_hypothesis', 'provides_evidence_for')"
                )
        
        return recommendations
    
    def _get_optional_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations for optional issues"""
        recommendations = []
        
        for issue in issues:
            if "Van Evera diagnostic test capabilities" in issue:
                recommendations.append(
                    "Add Van Evera diagnostic edge types for enhanced analysis "
                    "(e.g., 'supports', 'refutes', 'tests_hypothesis')"
                )
            elif "Probative value property support" in issue:
                recommendations.append(
                    "Add 'probative_value' property to edge types for quantitative analysis"
                )
        
        return recommendations
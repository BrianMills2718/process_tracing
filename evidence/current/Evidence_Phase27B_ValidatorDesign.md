=== DYNAMIC VALIDATOR ARCHITECTURE ===
=== REQUIREMENT CATEGORIES ===
CRITICAL REQUIREMENTS (system cannot function without):
- At least one edge type connecting Evidence to Hypothesis
- Valid JSON schema structure

OPTIONAL REQUIREMENTS (enhanced functionality):
- Van Evera diagnostic test capabilities
- Probative value properties
=== VALIDATION MODES ===
STRICT MODE: Requires all critical + optional requirements
MINIMAL MODE: Requires only critical requirements
SCHEMA_ONLY MODE: Validates JSON structure only


=== DYNAMIC VALIDATOR CLASS DESIGN ===

class DynamicOntologyValidator:
    '''
    Replaces hardcoded validation with functional requirement validation.
    Adapts to any ontology that meets system's functional needs.
    '''
    
    def __init__(self, ontology_manager):
        self.ontology_manager = ontology_manager
    
    def validate(self, mode='strict') -> ValidationResult:
        '''
        Validate ontology against functional requirements
        
        Args:
            mode: 'strict' (all requirements), 'minimal' (critical only), 'schema-only' (JSON only)
        
        Returns:
            ValidationResult with success/failure and detailed findings
        '''
    
    VALIDATION METHODS:
    - _validate_schema(): Check JSON structure integrity
    - _validate_critical_requirements(): Evidence->Hypothesis connectivity
    - _validate_optional_requirements(): Van Evera capabilities, probative values
    - _generate_clear_error_messages(): User-friendly requirement explanations

=== VALIDATION RESULT DESIGN ===

class ValidationResult:
    '''
    Structured validation outcome with detailed diagnostics
    '''
    
    PROPERTIES:
    - success: bool
    - mode: str  
    - critical_passed: bool
    - optional_passed: bool
    - missing_critical: List[str]
    - missing_optional: List[str]
    - error_messages: List[str]
    - recommendations: List[str]
    
    METHODS:
    - summary(): Human-readable validation summary
    - is_usable(): Whether system can operate (critical requirements met)

=== INTEGRATION DESIGN ===

REPLACE: analyze_direct.py hardcoded validation
WITH: DynamicOntologyValidator.validate(mode)

COMMAND LINE OPTIONS:
--validation-mode [strict|minimal|schema-only]  (replaces --evolution-mode)

BACKWARD COMPATIBILITY:
Default mode 'strict' maintains current behavior for existing users

SUCCESS CRITERIA MET:
✅ Designed functional requirement categories (critical vs optional)
✅ Defined validation modes for different use cases  
✅ Separated schema validation from functional validation
✅ Created architecture that adapts to ontology changes


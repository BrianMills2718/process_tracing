# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: CODE QUALITY IMPROVEMENTS (Updated 2025-01-08)

**System Status**: **Production-Ready** - Van Evera pipeline operational, ready for code quality improvements  
**Current Priority**: **Implement Planned Code Quality Enhancements** - Remove debug code, add strategic testing, enhance error logging  
**Academic Quality**: **Van Evera compliant** - Complete diagnostic test implementation with structured LLM integration

**NEXT IMPROVEMENT TASKS (2025-01-08)**:
- 🎯 **TASK 1**: Remove debug print statements from production code (54 statements in core/analyze.py)
- 🎯 **TASK 2**: Add unit tests for 5-6 most critical plugins (graph_validation, van_evera_testing, content_based_diagnostic_classifier, alternative_hypothesis_generator, evidence_connector_enhancer, primary_hypothesis_identifier)
- 🎯 **TASK 3**: Enhanced error logging with structured context for improved debugging and monitoring

**VALIDATED FOUNDATION (2025-01-08)**:
- ✅ **Core Van Evera Pipeline**: Functional (8-step academic workflow executes successfully)
- ✅ **Plugin Architecture**: 16 plugins operational and properly registered
- ✅ **LLM Integration**: Gemini 2.5 Flash with structured Pydantic output
- ✅ **Type Safety**: mypy clean - no type errors in 26 core source files
- ✅ **Security**: Proper API key handling via environment variables

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **VERIFY BEFORE FIXING**: Every problem must be reproduced through direct testing before attempting solutions
- **REAL PROBLEMS ONLY**: Address only technical issues confirmed through evidence-based validation
- **MINIMAL INTERVENTION**: Use surgical fixes for confirmed bugs, avoid system overhauls
- **EVIDENCE-BASED VALIDATION**: All claims backed by measurable, reproducible testing
- **PROCESS COMPLIANCE**: Follow CLAUDE.md requirements exactly (no unnecessary files)

### Quality Standards
- **Academic Functionality**: Maintain ≥60% Van Evera compliance (currently 67.9-71.5%)
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results

## Project Overview

**LLM-Enhanced Process Tracing Toolkit** - Production-ready system implementing Van Evera academic methodology for qualitative analysis using process tracing with comprehensive diagnostic tests and structured LLM integration.

### Architecture
- **Plugin System**: 16 registered plugins with proper abstractions
- **Van Evera Workflow**: 8-step academic analysis pipeline
- **LLM Integration**: Gemini 2.5 Flash with structured Pydantic output
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management

## 🚨 IMPLEMENTATION TASKS: CODE QUALITY IMPROVEMENTS

### **TASK 1: Remove Debug Print Statements (CRITICAL)**

**Problem**: 54 debug print statements in `core/analyze.py` reduce code quality and production readiness
**Evidence**: Debug statements identified via grep pattern `DEBUG_CHAINS|DEBUG_CM_EVAL|safe_print`

**Systematic Removal Plan**:

1. **Identify Debug Categories**:
```bash
# Verify current debug statements
grep -n "DEBUG_CHAINS\|DEBUG_CM_EVAL\|DEBUG_EVIDENCE_ANALYSIS\|DEBUG_CONDITIONS\|DEBUG_ACTORS\|DEBUG_THEORY_INSIGHTS" core/analyze.py
```

2. **Removal Strategy by Category**:

**REMOVE COMPLETELY** (18+ statements):
- All `DEBUG_CHAINS` statements - Pure development debugging
- All `DEBUG_CM_EVAL` statements - Academic debugging only  
- All `DEBUG_EVIDENCE_ANALYSIS`, `DEBUG_CONDITIONS`, `DEBUG_ACTORS`, `DEBUG_THEORY_INSIGHTS`

**CONVERT TO PROPER LOGGING** (26+ statements):
- Convert `safe_print(f"[ERROR] ...")` to `logger.error(..., exc_info=True)`
- Convert `safe_print(f"[WARN] ...")` to `logger.warning(...)`
- Convert `safe_print(f"[INFO] ...")` to `logger.info(...)`
- Convert `safe_print(f"[SUCCESS] ...")` to `logger.info(...)`

3. **Implementation Pattern**:
```python
# REMOVE THIS (example):
safe_print(f"DEBUG_CHAINS: Timeout reached ({timeout_seconds}s), returning {paths_found} paths")

# CONVERT THIS (example):
safe_print(f"[ERROR] Failed to generate network data: {e}")
# TO THIS:
logger.error("Failed to generate network data", exc_info=True, extra={'operation': 'network_generation'})
```

4. **Validation Requirements**:
```bash
# After each batch of removals, validate system still works:
python -c "
from core.plugins.van_evera_workflow import VanEveraWorkflow
from core.plugins.base import PluginContext
print('SUCCESS: Core imports functional after debug removal')
"

# Verify no debug statements remain:
grep -n "DEBUG_CHAINS\|DEBUG_CM_EVAL" core/analyze.py && echo "FAILED: Debug statements remain" || echo "SUCCESS: Debug statements removed"
```

**Success Criteria**:
- Zero debug statements in production code
- All error messages converted to proper logging
- Van Evera pipeline functionality preserved
- No regression in HTML output generation

---

### **TASK 2: Add Unit Tests for Critical Plugins (HIGH PRIORITY)**

**Problem**: 16 plugins lack unit test coverage for individual functionality
**Evidence**: Plugin architecture well-designed but testing focuses on end-to-end integration

**Critical Plugins for Testing** (Priority Order):

1. **`graph_validation`** - Input validation prevents corruption
2. **`van_evera_testing`** - Core academic methodology engine
3. **`content_based_diagnostic_classifier`** - Evidence classification engine
4. **`alternative_hypothesis_generator`** - Academic rigor requirement
5. **`evidence_connector_enhancer`** - Data quality foundation  
6. **`primary_hypothesis_identifier`** - Final analysis conclusion

**Test Implementation Pattern**:

1. **Create Test Structure**:
```bash
# Create plugin test directory if not exists
mkdir -p tests/plugins/

# Create test files for each critical plugin
touch tests/plugins/test_graph_validation.py
touch tests/plugins/test_van_evera_testing.py
touch tests/plugins/test_content_based_diagnostic_classifier.py
touch tests/plugins/test_alternative_hypothesis_generator.py
touch tests/plugins/test_evidence_connector_enhancer.py
touch tests/plugins/test_primary_hypothesis_identifier.py
```

2. **Test Template Pattern**:
```python
# tests/plugins/test_[plugin_name].py
"""Unit tests for [PluginName] - focused on plugin logic, not integration"""

import pytest
import networkx as nx
from unittest.mock import Mock, patch
from core.plugins.[plugin_name] import [PluginName]
from core.plugins.base import PluginContext, PluginValidationError

class Test[PluginName]:
    """Unit tests for [PluginName]"""
    
    @pytest.fixture
    def plugin_context(self):
        """Mock plugin context with minimal dependencies"""
        context = Mock(spec=PluginContext)
        context.config = {}
        context.data_bus = {}
        return context
        
    @pytest.fixture  
    def plugin(self, plugin_context):
        """Plugin instance with mocked context"""
        return [PluginName]("[plugin_id]", plugin_context)
        
    @pytest.fixture
    def minimal_graph(self):
        """Minimal test graph with known characteristics"""
        G = nx.DiGraph()
        G.add_node("H1", type="Hypothesis", description="Test hypothesis")
        G.add_node("E1", type="Evidence", description="Test evidence")
        G.add_edge("E1", "H1", type="supports")
        return G
        
    class TestInputValidation:
        """Test validate_input() method thoroughly"""
        
        def test_valid_input_accepted(self, plugin, minimal_graph):
            """Plugin accepts valid graph input"""
            # Should not raise exception
            plugin.validate_input(minimal_graph)
            
        def test_invalid_input_rejected(self, plugin):
            """Plugin rejects invalid input with clear error"""
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input("invalid_input")
            assert "must be" in str(exc_info.value).lower()
            
        def test_missing_attributes_detected(self, plugin):
            """Plugin detects missing node attributes"""
            G = nx.DiGraph() 
            G.add_node("H1")  # Missing 'type' attribute
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(G)
            assert "missing" in str(exc_info.value).lower()
                
    class TestExecutionLogic:
        """Test execute() method with various scenarios"""
        
        def test_execute_with_valid_input(self, plugin, minimal_graph):
            """Plugin executes successfully with valid input"""
            result = plugin.execute(minimal_graph)
            assert result is not None
            # Add specific assertions based on expected plugin behavior
            
        def test_execute_idempotent(self, plugin, minimal_graph):
            """Plugin execution is idempotent (same input = same output)"""
            result1 = plugin.execute(minimal_graph.copy())
            result2 = plugin.execute(minimal_graph.copy())
            # Compare relevant aspects of results
            assert type(result1) == type(result2)
            
    class TestErrorHandling:
        """Test error conditions and edge cases"""
        
        def test_empty_graph_handling(self, plugin):
            """Plugin handles empty graph appropriately"""
            empty_graph = nx.DiGraph()
            # Should either work or fail gracefully with clear error
            try:
                result = plugin.execute(empty_graph)
                assert result is not None  # If it succeeds, result should be valid
            except PluginValidationError as e:
                assert len(str(e)) > 0  # Error message should be informative
```

3. **Plugin-Specific Test Focus**:

**`graph_validation`**:
- Test graph copying (immutability)
- Test node attribute validation
- Test edge reference validation

**`van_evera_testing`**:
- Test diagnostic test classification logic
- Test hypothesis scoring algorithms
- Test Van Evera test type assignment

**`content_based_diagnostic_classifier`**:
- Test evidence type classification
- Test classification confidence scoring
- Test edge case handling (ambiguous evidence)

4. **Validation Commands**:
```bash
# Run plugin tests
python -m pytest tests/plugins/ -v

# Check test coverage for plugins
python -m pytest tests/plugins/ --cov=core.plugins --cov-report=term-missing

# Validate plugin functionality unchanged
python -c "
from core.plugins.register_plugins import register_all_plugins
register_all_plugins()
print('SUCCESS: All 16 plugins still register after testing')
"
```

**Success Criteria**:
- 5-6 plugins with >80% test coverage of public methods
- All plugin tests passing
- Core plugin functionality validated through unit tests
- No regression in plugin integration

---

### **TASK 3: Enhanced Error Logging with Structured Context (IMPORTANT)**

**Problem**: Current logging lacks structured context for effective debugging and monitoring
**Evidence**: Basic exception logging without operational context makes debugging difficult

**Enhanced Logging Implementation**:

1. **Define Structured Context Categories**:
```python
# Add to core/plugins/base.py or create core/logging_utils.py
ERROR_CATEGORIES = {
    "validation_failure": "Input data validation failed",
    "llm_timeout": "LLM API call exceeded timeout", 
    "graph_corruption": "Graph structure became invalid",
    "plugin_execution": "Plugin processing logic failed",
    "academic_validation": "Van Evera methodology validation failed",
    "io_operation": "File system or network operation failed"
}

def log_structured_error(logger, message, error_category, operation_context=None, exc_info=True, **extra_context):
    """Log error with structured context for improved debugging"""
    extra = {
        "error_category": error_category,
        "operation": operation_context,
        **extra_context
    }
    logger.error(message, exc_info=exc_info, extra=extra)
```

2. **Plugin Error Enhancement Pattern**:
```python
# Current plugin error logging:
except Exception as e:
    raise PluginExecutionError(self.id, f"Failed to process: {e}")

# Enhanced plugin error logging:
except Exception as e:
    self.logger.error(
        "Plugin execution failed",
        exc_info=True,
        extra={
            "plugin_id": self.id,
            "operation": "graph_validation", 
            "error_category": "validation_failure",
            "graph_nodes": len(graph.nodes()) if hasattr(graph, 'nodes') else 'unknown',
            "graph_edges": len(graph.edges()) if hasattr(graph, 'edges') else 'unknown'
        }
    )
    raise PluginExecutionError(self.id, f"Graph validation failed: {e}")
```

3. **Core Analysis Error Enhancement**:
```python
# Enhance error logging in core/analyze.py functions
# Current:
logger.error(f"Failed to process graph: {e}")

# Enhanced:
logger.error(
    "Graph processing failed", 
    exc_info=True,
    extra={
        "operation": "van_evera_analysis",
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(), 
        "error_category": "graph_corruption",
        "analysis_stage": "causal_chain_identification"
    }
)
```

4. **LLM Integration Error Enhancement**:
```python
# Enhance LLM error logging in core/plugins/van_evera_llm_interface.py
# Current:
logger.error(f"LLM call failed: {e}")

# Enhanced:
logger.error(
    "LLM API call failed",
    exc_info=True,
    extra={
        "operation": "van_evera_prediction_evaluation",
        "model_type": self.model_type,
        "error_category": "llm_timeout" if "timeout" in str(e).lower() else "llm_error",
        "prediction_description": prediction_description[:100] if prediction_description else None,
        "diagnostic_type": diagnostic_type
    }
)
```

5. **Validation Requirements**:
```bash
# Test structured logging works
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s - %(error_category)s')
logger = logging.getLogger('test')
logger.error('Test error', extra={'error_category': 'test', 'operation': 'validation'})
print('SUCCESS: Structured logging functional')
"

# Validate plugin logging enhanced
python -c "
from core.plugins.graph_validation import GraphValidationPlugin
from core.plugins.base import PluginContext
import networkx as nx

# Test plugin error logging includes structured context
context = type('MockContext', (), {'config': {}, 'data_bus': {}})()
plugin = GraphValidationPlugin('test', context)
try:
    plugin.validate_input('invalid')
except Exception:
    print('SUCCESS: Plugin validation errors include context')
"
```

**Success Criteria**:
- All plugin errors include structured context
- Core analysis errors include operational context
- LLM integration errors include model and operation context
- Improved debugging efficiency during development
- No regression in functionality

---

## 🔧 IMPLEMENTATION GUIDANCE

### **Evidence-First Methodology**
1. **Validate Before Implementation**: Run baseline functionality tests before starting each task
2. **Implement Incrementally**: Complete one category at a time, validate after each change
3. **Measure Success**: Use specific validation commands provided for each task
4. **Document Evidence**: Record test results and validation outcomes

### **Quality Gates**
- **Task 1**: Zero debug statements, no functionality regression
- **Task 2**: Plugin tests passing with >80% coverage, no integration regression  
- **Task 3**: All errors include structured context, enhanced debugging capability
- **All Tasks**: Van Evera pipeline continues operating correctly after each change

**Process Compliance**: Follow minimal intervention principle - make targeted improvements without system overhauls.
- CLAUDE.md Update Command - Evidence-Based Development Workflow

## Overview
commit then date CLAUDE.md to clear out resolved tasks and populate it with instructions for resolving the next tasks using evidence-based development practices. The instructions should be detailed enough for a new LLM to implement with no context beyond CLAUDE.md and referenced files.

## Core CLAUDE.md Requirements

### 1. Coding Philosophy Section (Mandatory)
Every CLAUDE.md must include:
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### 2. Codebase Structure Section (Mandatory)  
Concisely document:
- All relevant planning and documentation files
- Key entry points and main orchestration files
- Module organization and responsibilities
- Important integration points (ResourceOrchestrator, healing_integration.py, etc.)

### 3. Evidence Structure Requirements (Updated)
**CURRENT PRACTICE**: Use structured evidence organization instead of single Evidence.md:

```
evidence/
├── current/
│   └── Evidence_[PHASE]_[TASK].md     # Current development phase only
├── completed/  
│   └── Evidence_[PHASE]_[TASK].md     # Completed phases (archived)
```

**CRITICAL**: 
- Evidence files must contain ONLY current phase work (no historical contradictions)
- Raw execution logs required for all claims
- No success declarations without demonstrable proof
- Archive completed phases to avoid chronological confusion

## Updated Workflow Process

### Phase 1: Task Implementation
1. **Implement tasks** following CLAUDE.md instructions
2. **Document evidence** in `evidence/current/Evidence_[PHASE]_[TASK].md`
3. **Include raw logs** for all validation steps
4. **Test thoroughly** - assume nothing works until proven
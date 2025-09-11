# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 PERMANENT INFORMATION -- DO NOT CHANGE ON UPDATES

### LLM-First Architecture Policy (MANDATORY)

**CORE PRINCIPLE**: This system is **LLM-FIRST** with **ZERO TOLERANCE** for rule-based or keyword-based implementations.

**PROHIBITED IMPLEMENTATIONS**:
- ❌ Keyword matching for evidence classification (`if 'ideological' in text`)
- ❌ Hardcoded probative value assignments (`probative_value = 0.7`)
- ❌ Rule-based contradiction detection (`if 'before' in hypothesis and 'after' in evidence`)
- ❌ Domain classification using keyword lists
- ❌ Confidence thresholds based on hardcoded ranges
- ❌ Any `if/elif` chains for semantic understanding
- ❌ Dataset-specific logic (American Revolution hardcoded rules)
- ❌ Historical period-specific keyword matching
- ❌ Returning None/0/[] on LLM failure (must raise LLMRequiredError)
- ❌ Mixed LLM configurations (some calls to Gemini, others to different models)
- ❌ Direct API calls bypassing LiteLLM infrastructure

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation
- ✅ Generalist process tracing without dataset-specific hardcoding
- ✅ Raise LLMRequiredError on any LLM failure (fail-fast)
- ✅ Consistent LiteLLM routing for ALL LLM operations
- ✅ Single model configuration across entire system

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: Phase 25A Complete - Architectural Refactoring Successfully Implemented (Updated 2025-01-11)

**System Status**: **✅ REFACTORING COMPLETE - System Operational with Dynamic Ontology Lookups**  
**Latest Achievement**: **Phase 25A Complete - OntologyManager abstraction layer implemented, all critical modules migrated**  
**Next Priority**: **Ready for ontology consolidation or other improvements as needed**

**PHASE 24A INVESTIGATION RESULTS**:
- ✅ **Architecture Assessment**: Hybrid system with poor abstraction - 23 files have hardcoded edge type dependencies
- ✅ **Redundancy Analysis**: 5 redundant Evidence→Hypothesis edge types violating Van Evera academic standards
- ✅ **Impact Analysis**: High-risk system-wide changes affecting LLM extraction, graph loading, HTML generation
- ✅ **Academic Gaps**: Missing temporal modeling, alternative hypothesis framework, unified evidence assessment

**EXECUTIVE DECISIONS** (User-approved):
- **NO backwards compatibility required** - clean break approach
- **Downtime acceptable** - aggressive refactoring permitted
- **Migration approach**: Create migration tools for existing data files
- **Testing strategy**: Comprehensive test coverage before deployment

## ✅ PHASE 25A COMPLETE: Aggressive Architectural Refactoring

### ACHIEVEMENTS: Successfully implemented architectural refactoring

**COMPLETED DELIVERABLES**:
- ✅ **OntologyManager abstraction layer** - Centralized ontology query system (core/ontology_manager.py)
- ✅ **Module migration** - 8 critical modules migrated to dynamic lookups
- ✅ **Data migration tool** - Comprehensive migration utility (tools/migrate_ontology.py)
- ✅ **Test coverage** - 22 unit tests with 100% coverage for OntologyManager
- ✅ **System validation** - End-to-end pipeline tested and operational

## 🔧 IMPLEMENTATION TASKS

### TASK 1: Create OntologyManager Abstraction Layer (Priority 1)

**OBJECTIVE**: Build centralized ontology query and validation system

**IMPLEMENTATION STEPS**:

1. **Create `core/ontology_manager.py`**:
```python
# core/ontology_manager.py
from typing import List, Dict, Set, Optional, Tuple
from core.ontology import NODE_TYPES, EDGE_TYPES

class OntologyManager:
    """Centralized ontology query and validation system."""
    
    def __init__(self):
        self.node_types = NODE_TYPES
        self.edge_types = EDGE_TYPES
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build efficient lookup tables for ontology queries."""
        # Build domain/range lookup tables
        self.edge_by_domain = {}
        self.edge_by_range = {}
        self.edge_by_pair = {}
        
        for edge_type, config in self.edge_types.items():
            domains = config.get('domain', [])
            ranges = config.get('range', [])
            
            for domain in domains:
                if domain not in self.edge_by_domain:
                    self.edge_by_domain[domain] = set()
                self.edge_by_domain[domain].add(edge_type)
                
                for range_type in ranges:
                    pair = (domain, range_type)
                    if pair not in self.edge_by_pair:
                        self.edge_by_pair[pair] = set()
                    self.edge_by_pair[pair].add(edge_type)
    
    def get_edge_types_for_relationship(self, source_type: str, target_type: str) -> List[str]:
        """Returns valid edge types for given node type pair."""
        return list(self.edge_by_pair.get((source_type, target_type), []))
    
    def get_evidence_hypothesis_edges(self) -> List[str]:
        """Returns all edge types that connect Evidence to Hypothesis."""
        return self.get_edge_types_for_relationship('Evidence', 'Hypothesis')
    
    def get_van_evera_edges(self) -> List[str]:
        """Returns edge types relevant to Van Evera diagnostic tests."""
        # Currently returns Evidence→Hypothesis edges with diagnostic properties
        van_evera_edges = []
        for edge_type in self.get_evidence_hypothesis_edges():
            properties = self.edge_types[edge_type].get('properties', {})
            if 'diagnostic_type' in properties or 'probative_value' in properties:
                van_evera_edges.append(edge_type)
        return van_evera_edges
    
    def validate_edge(self, edge: dict) -> Tuple[bool, Optional[str]]:
        """Validates edge against ontology constraints.
        Returns (is_valid, error_message)."""
        edge_type = edge.get('type')
        if edge_type not in self.edge_types:
            return False, f"Unknown edge type: {edge_type}"
        
        # Further validation logic here
        return True, None
    
    def get_edge_properties(self, edge_type: str) -> dict:
        """Returns required/optional properties for edge type."""
        if edge_type not in self.edge_types:
            return {}
        return self.edge_types[edge_type].get('properties', {})

# Global singleton instance
ontology_manager = OntologyManager()
```

2. **Create comprehensive tests**:
```python
# tests/test_ontology_manager.py
import pytest
from core.ontology_manager import ontology_manager

def test_get_evidence_hypothesis_edges():
    edges = ontology_manager.get_evidence_hypothesis_edges()
    assert 'tests_hypothesis' in edges
    assert 'provides_evidence_for' in edges
    
def test_backwards_compatibility():
    # Ensure old hardcoded lists match new dynamic queries
    old_list = ['supports', 'provides_evidence_for']  # From codebase
    new_edges = ontology_manager.get_evidence_hypothesis_edges()
    for edge in old_list:
        assert edge in new_edges, f"Lost edge type: {edge}"
```

**EVIDENCE DOCUMENTATION**:
- **CREATE**: `evidence/current/Evidence_Phase25A_Refactoring.md`
- **DOCUMENT**: Implementation progress with test results
- **TRACK**: Each module migration with before/after comparisons

### TASK 2: Migrate Low-Risk Modules First (Priority 2)

**OBJECTIVE**: Replace hardcoded edge type lists in non-critical modules

**MIGRATION ORDER** (Low → High Risk):
1. Test files in `tests/` directory
2. Plugin modules in `core/plugins/`
3. Utility modules (`core/streaming_html.py`)
4. Analysis modules (`core/van_evera_testing_engine.py`)

**MIGRATION PATTERN**:
```python
# BEFORE (hardcoded)
if edge_type in ['supports', 'provides_evidence_for']:
    process_edge()

# AFTER (dynamic)
from core.ontology_manager import ontology_manager

if edge_type in ontology_manager.get_evidence_hypothesis_edges():
    process_edge()
```

**VALIDATION STEPS**:
1. Run existing tests after each migration
2. Compare outputs before/after migration
3. Document any behavioral changes

### TASK 3: Migrate Critical Path Modules (Priority 3)

**OBJECTIVE**: Replace hardcoded dependencies in critical system paths

**CRITICAL MODULES**:
1. `core/html_generator.py` - Visualization logic
2. `core/disconnection_repair.py` - Graph repair system
3. `core/structured_extractor.py` - LLM extraction pipeline
4. `core/extract.py` - Core extraction logic
5. `core/analyze.py` - Analysis pipeline

**SPECIAL CONSIDERATIONS**:
- Create parallel execution paths initially
- Add extensive logging for debugging
- Prepare rollback scripts if needed

### TASK 4: Create Data Migration Tools (Priority 4)

**OBJECTIVE**: Build tools to migrate existing JSON files if ontology changes

**IMPLEMENTATION**:
```python
# tools/migrate_ontology.py
import json
import os
from typing import Dict, Any

def migrate_edge_types(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate edge types in graph data to new ontology."""
    # Map old edge types to new consolidated types
    edge_mapping = {
        'provides_evidence_for': 'tests_hypothesis',
        'updates_probability': 'tests_hypothesis',
        'weighs_evidence': 'tests_hypothesis',
        'supports': 'tests_hypothesis',
        # Keep tests_hypothesis as is
    }
    
    for edge in graph_data.get('edges', []):
        old_type = edge.get('type')
        if old_type in edge_mapping:
            edge['type'] = edge_mapping[old_type]
            edge['_original_type'] = old_type  # Preserve for reference
    
    return graph_data

def migrate_directory(input_dir: str, output_dir: str):
    """Migrate all JSON files in directory."""
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            # Process each file
            pass
```

## 📊 SUCCESS CRITERIA

### **Implementation Success Metrics:**
1. **100% Test Coverage**: All new OntologyManager methods tested
2. **Zero Hardcoded References**: All 23 files migrated to dynamic lookups
3. **Performance Maintained**: No degradation >10% in processing speed
4. **Migration Complete**: All existing JSON files can be migrated

### **Validation Requirements:**
1. **Regression Tests**: All existing tests must pass
2. **Output Comparison**: Outputs match pre-refactoring baselines
3. **Performance Tests**: Benchmark comparisons documented
4. **Integration Tests**: End-to-end pipeline validation

---

## 🏗️ Codebase Structure

### Key Entry Points  
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline with basic HTML fallback
- **`core/structured_extractor.py`**: LLM extraction (Phase 23A: enhanced with raw response capture)
- **`core/analyze.py`**: Contains `load_graph()` (Phase 23A: fixed MultiDiGraph) + hanging `generate_html_report()`

### Critical Files for Phase 25A Refactoring
- **`core/ontology_manager.py`**: NEW - Centralized ontology abstraction layer (CREATE THIS)
- **`core/ontology.py`**: Existing ontology loader (will be wrapped by manager)
- **23 files with hardcoded dependencies**: All require migration to dynamic lookups
- **`config/ontology_config.json`**: Authoritative ontology definition (DO NOT MODIFY YET)
- **`tools/migrate_ontology.py`**: NEW - Data migration tools (CREATE THIS)

### Working Components (Phase 24A Investigation Complete)
- **Rich HTML Generation**: `core/html_generator.py` with interactive vis.js network visualizations
- **Van Evera Analytics**: Evidence-hypothesis analysis revealing ontology redundancies
- **Complete Pipeline**: TEXT → JSON → Rich HTML working end-to-end
- **Cross-Input Validation**: Multiple datasets tested with consistent results

### Runtime Environment
- **Virtual Environment**: `test_env/` activated with `source test_env/bin/activate`
- **Test Inputs**: Multiple validated inputs in `input_text/` (French Revolution, American Revolution, Westminster Debate)
- **Output Structure**: `output_data/direct_extraction/` contains rich HTML reports with network visualizations

---

## 📋 Coding Philosophy

### NO LAZY IMPLEMENTATIONS
- Implement complete, working code - no stubs or placeholders
- Test every component before declaring completion
- Raw execution logs required for all claims

### FAIL-FAST PRINCIPLES  
- Surface data integrity issues immediately
- No silent data loss tolerance
- Clear error reporting with actionable information

### EVIDENCE-BASED DEVELOPMENT
- All implementation progress must be documented in `evidence/current/Evidence_Phase25A_Refactoring.md`
- Include test results and performance metrics
- Document each module migration with before/after analysis

### SYSTEMATIC VALIDATION
- Run regression tests after each module migration
- Compare outputs before/after refactoring
- Benchmark performance to ensure no degradation

---

## 📁 Evidence Structure

⚠️ **REFACTORING DOCUMENTATION REQUIREMENTS**

Evidence for Phase 25A must be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase25A_Refactoring.md           # CREATE THIS FILE
├── completed/
│   ├── Evidence_Phase24A_OntologyInvestigation.md # Archived
│   └── Evidence_Phase23B_HTMLGeneration.md        # Archived
```

**EVIDENCE FILE INSTRUCTIONS**:
- **CREATE**: `evidence/current/Evidence_Phase25A_Refactoring.md`
- **FORMAT**: Structured markdown with test results and metrics
- **CONTENT**: Implementation progress, test outputs, performance benchmarks
- **PURPOSE**: Track refactoring progress and validate no regressions

**REQUIRED EVIDENCE FOR PHASE 25A**:
- OntologyManager implementation with test results
- Module migration tracking (23 files total)
- Before/after output comparisons for validation
- Performance benchmarks showing no degradation
- Regression test results after each migration
- Data migration tool testing results

**CRITICAL**: Every module migration must be validated with tests before proceeding.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
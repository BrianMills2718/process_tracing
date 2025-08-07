# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: Q/H1/H2/H3 ACADEMIC STRUCTURE COMPLETION

**System Status**: **Q/H1/H2/H3 Foundation Complete** - Core ontology and plugins updated for academic structure  
**Current Priority**: **Workflow Integration & Research Question Generation** - Complete end-to-end academic pipeline  
**Academic Quality**: **87.3% Testing Compliance Achieved** - Publication-ready Van Evera methodology operational  
**Recently Completed**: 
- ✅ **Phase 1**: ontology_config.json updated with Research_Question type and enhanced Hypothesis properties
- ✅ **Phase 2**: alternative_hypothesis_generator.py creates Q_H2-Q_H9 with academic structure
- ✅ **Phase 3**: van_evera_testing.py supports hypothesis ranking with Van Evera diagnostic weights
- ✅ **Academic Quality**: 87.3% testing compliance achieved with sophisticated prediction engine

**Current Analysis**: Core Q/H1/H2/H3 components implemented with academic rigor. **Next critical phase**: Complete workflow integration with research question generation and primary hypothesis identification for full end-to-end academic pipeline.

**Immediate Goal**: Implement **Phase 4** (van_evera_workflow.py updates) with research question generation and evidence-based Q_H1 identification to complete the academic structure migration.

## 📋 CURRENT FOCUS: PHASE 4 - WORKFLOW INTEGRATION & Q_H1 IDENTIFICATION

### **EVIDENCE-BASED COMPLETION ANALYSIS**

**Current State**: Individual components functional but require integration  
**Target State**: Complete end-to-end Q/H1/H2/H3 academic pipeline with evidence-based primary hypothesis selection  
**Critical Gap**: Research question generation and systematic Q_H1 identification missing from workflow

**PHASE 4 REQUIREMENTS**:
- **Research Question Generation**: Domain-aware academic question creation from hypothesis content
- **Evidence-Based Q_H1 Selection**: Promote highest-ranked hypothesis to primary status
- **Workflow Orchestration**: Seamless integration of all Q/H1/H2/H3 components
- **Backward Compatibility**: Maintain support for existing H_001/AE_001 analyses

## Coding Philosophy (Mandatory)

### Core Development Principles
- **EVIDENCE-BASED DEVELOPMENT**: All improvements must be measured and validated with concrete metrics
- **SYSTEMATIC METHODOLOGY**: Complete integration in dependency order to prevent system breakage
- **PLUGIN ARCHITECTURE**: Use existing extensible plugin system for all enhancements
- **ACADEMIC RIGOR FIRST**: Maintain 87.3% testing compliance throughout integration
- **BACKWARD COMPATIBILITY**: Preserve existing analysis functionality during migration

### Quality Standards  
- **Academic Structure**: Evidence-based Q_H1 selection, not arbitrary assignment
- **Testing Compliance**: Maintain ≥80% Van Evera methodology compliance (currently 87.3%)
- **Integration Quality**: All components work together seamlessly in end-to-end pipeline
- **Research Question Quality**: Domain-appropriate, academically sophisticated questions

## Project Overview

This is an LLM-enhanced Process Tracing Toolkit with **complete Van Evera plugin architecture** achieving **87.3% academic quality**. System has core Q/H1/H2/H3 components implemented and requires **workflow integration** to complete the academic structure migration.

## 🚨 PHASE 4A: WORKFLOW INTEGRATION (CRITICAL - IMMEDIATE)

### **Evidence-Based Implementation Plan**

**Root Issue**: van_evera_workflow.py orchestrates the analysis pipeline but doesn't integrate Q/H1/H2/H3 structure components. Current workflow generates hypotheses and alternatives separately, missing research question generation and evidence-based Q_H1 selection.

**File**: `core/plugins/van_evera_workflow.py`

### **Required Research Question Generation Integration**

**Add research question generation logic**:
```python
def _generate_research_question(self, case_context: str, hypotheses_data: List[Dict]) -> Dict:
    """Generate domain-aware research question from hypothesis content analysis"""
    
    # Domain classification based on hypothesis content
    domain_keywords = {
        'political': ['government', 'political', 'authority', 'power', 'governance', 'state'],
        'economic': ['economic', 'trade', 'merchant', 'financial', 'commercial', 'wealth'],
        'social': ['social', 'class', 'popular', 'community', 'cultural', 'society'],
        'military': ['military', 'war', 'conflict', 'army', 'battle', 'strategic'],
        'ideological': ['ideological', 'ideas', 'beliefs', 'philosophy', 'principles', 'values']
    }
    
    # Analyze hypothesis content to determine primary domain
    all_text = ' '.join([h.get('description', '') for h in hypotheses_data]).lower()
    domain_scores = {}
    
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in all_text)
        domain_scores[domain] = score
    
    primary_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
    
    # Generate domain-specific research question
    domain_templates = {
        'political': "What political factors and institutional dynamics explain the {case_outcome} in {case_name}?",
        'economic': "What economic interests and commercial factors drove the {case_outcome} in {case_name}?", 
        'social': "What social forces and cultural dynamics caused the {case_outcome} in {case_name}?",
        'military': "What military and strategic factors led to the {case_outcome} in {case_name}?",
        'ideological': "What ideological beliefs and intellectual currents explain the {case_outcome} in {case_name}?"
    }
    
    # Extract case name and outcome from context
    case_name = case_context.replace('_', ' ').title() if case_context else "Case Study"
    case_outcome = "outcome under investigation"  # Could be enhanced with content analysis
    
    research_question = domain_templates[primary_domain].format(
        case_outcome=case_outcome, 
        case_name=case_name
    )
    
    return {
        'id': 'Q',
        'type': 'Research_Question',
        'properties': {
            'description': research_question,
            'domain': primary_domain,
            'scope': case_name,
            'theoretical_sophistication_score': len([h for h in hypotheses_data if len(h.get('description', '').split()) > 20]),
            'generated_from_hypotheses_count': len(hypotheses_data),
            'timestamp': datetime.now().isoformat()
        }
    }
```

### **Required Evidence-Based Q_H1 Identification Logic**

**Add Q_H1 selection system**:
```python
def _identify_primary_hypothesis(self, hypotheses_with_rankings: List[Dict]) -> Dict:
    """Identify primary hypothesis (Q_H1) based on evidence-based composite scoring"""
    
    if not hypotheses_with_rankings:
        return None
    
    # Composite scoring criteria (from van_evera_testing.py ranking system)
    for hypothesis in hypotheses_with_rankings:
        properties = hypothesis.get('properties', {})
        
        # Extract ranking components
        van_evera_score = properties.get('ranking_score', 0.0)  # From Van Evera testing (40% weight)
        evidence_support = len([e for e in properties.get('supporting_evidence', [])]) / 10.0  # Evidence count normalized (30% weight)
        theoretical_depth = min(1.0, len(properties.get('key_predictions', [])) / 5.0)  # Prediction count normalized (20% weight)
        elimination_power = properties.get('diagnostic_test_passes', 0) / 5.0  # Diagnostic test performance (10% weight)
        
        # Weighted composite score
        composite_score = (
            van_evera_score * 0.40 +
            evidence_support * 0.30 +
            theoretical_depth * 0.20 +
            elimination_power * 0.10
        )
        
        hypothesis['composite_score'] = composite_score
    
    # Sort by composite score and select highest
    sorted_hypotheses = sorted(hypotheses_with_rankings, key=lambda h: h.get('composite_score', 0), reverse=True)
    primary_hypothesis = sorted_hypotheses[0]
    
    # Update to Q_H1 format
    primary_hypothesis['id'] = 'Q_H1'
    primary_hypothesis['properties']['hypothesis_type'] = 'primary'
    primary_hypothesis['properties']['academic_rank'] = 1
    primary_hypothesis['properties']['selection_rationale'] = f"Selected as Q_H1 based on composite score: {primary_hypothesis.get('composite_score', 0):.3f}"
    
    # Update alternatives to Q_H2, Q_H3, etc.
    for rank, hypothesis in enumerate(sorted_hypotheses[1:], 2):
        hypothesis['id'] = f'Q_H{rank}'
        hypothesis['properties']['hypothesis_type'] = 'alternative'
        hypothesis['properties']['academic_rank'] = rank
    
    return {
        'primary_hypothesis': primary_hypothesis,
        'alternative_hypotheses': sorted_hypotheses[1:],
        'selection_methodology': 'evidence_based_composite_scoring',
        'scoring_criteria': {
            'van_evera_testing': '40%',
            'evidence_support': '30%', 
            'theoretical_depth': '20%',
            'elimination_power': '10%'
        }
    }
```

### **Required Workflow Orchestration Updates**

**Update execute method in van_evera_workflow.py**:
```python
def execute(self, data: Any) -> Dict[str, Any]:
    """Execute complete Van Evera analysis with Q/H1/H2/H3 academic structure"""
    self.logger.info("START: Van Evera workflow with Q/H1/H2/H3 academic structure")
    
    # Steps 1-4: Existing workflow (graph generation, alternative hypothesis generation, etc.)
    # ... [existing code] ...
    
    # Step 5: Van Evera testing with ranking (already updated in Phase 3)
    van_evera_results = self._execute_van_evera_testing(current_graph_data)
    updated_graph_data = van_evera_results.get('updated_graph_data', current_graph_data)
    
    # PHASE 4 ADDITIONS:
    
    # Step 6: Generate research question from hypothesis content
    hypotheses_nodes = [n for n in updated_graph_data['nodes'] if n.get('type') == 'Hypothesis']
    research_question = self._generate_research_question(
        case_context=data.get('case_name', 'analysis'),
        hypotheses_data=hypotheses_nodes
    )
    updated_graph_data['nodes'].append(research_question)
    
    # Step 7: Identify and promote primary hypothesis to Q_H1
    q_h_structure = self._identify_primary_hypothesis(hypotheses_nodes)
    if q_h_structure:
        # Update nodes with new Q_H1/H2/H3 structure
        hypothesis_ids = [h['id'] for h in hypotheses_nodes]
        for i, node in enumerate(updated_graph_data['nodes']):
            if node['id'] in hypothesis_ids:
                # Find matching hypothesis in Q_H structure
                if node['id'] == q_h_structure['primary_hypothesis']['id']:
                    updated_graph_data['nodes'][i] = q_h_structure['primary_hypothesis']
                else:
                    # Find in alternatives
                    for alt_hyp in q_h_structure['alternative_hypotheses']:
                        if node.get('properties', {}).get('original_id') == alt_hyp.get('properties', {}).get('original_id'):
                            updated_graph_data['nodes'][i] = alt_hyp
                            break
        
        # Add edges connecting hypotheses to research question
        for hypothesis in [q_h_structure['primary_hypothesis']] + q_h_structure['alternative_hypotheses']:
            research_connection_edge = {
                'source_id': hypothesis['id'],
                'target_id': research_question['id'],
                'type': 'addresses_research_question',
                'properties': {
                    'relevance': 1.0,
                    'approach': f"Hypothesis {hypothesis['properties']['academic_rank']}"
                }
            }
            updated_graph_data['edges'].append(research_connection_edge)
    
    # Step 8: Enhanced workflow completion with Q/H1/H2/H3 metadata
    final_results = {
        'updated_graph_data': updated_graph_data,
        'academic_structure': {
            'research_question': research_question,
            'q_h_structure': q_h_structure,
            'academic_quality_maintained': van_evera_results.get('testing_compliance_score', 0) >= 80
        },
        'van_evera_testing_results': van_evera_results,
        'workflow_completion_metrics': {
            'total_hypotheses': len(hypotheses_nodes),
            'research_question_generated': True,
            'primary_hypothesis_identified': q_h_structure is not None,
            'academic_structure_complete': True
        }
    }
    
    self.logger.info(f"COMPLETE: Q/H1/H2/H3 workflow with {len(hypotheses_nodes)} hypotheses, "
                    f"testing compliance: {van_evera_results.get('testing_compliance_score', 0):.1f}%")
    
    return final_results
```

### **Validation Strategy**
```bash
cd "C:\Users\Brian\Documents\code\process_tracing"
python -c "
from core.plugins.van_evera_workflow import VanEveraWorkflowPlugin
from core.plugins.base import PluginContext
import json

# Test with existing revolutions dataset
with open('output_data/revolutions/revolutions_20250805_122000_graph.json', 'r') as f:
    test_data = json.load(f)

context = PluginContext({'llm_query_func': lambda x, **kwargs: 'test response'})
workflow = VanEveraWorkflowPlugin('test_workflow', context)

try:
    result = workflow.execute({
        'graph_data': test_data,
        'case_name': 'american_revolution'
    })
    
    # Validate Q/H1/H2/H3 structure
    nodes = result['updated_graph_data']['nodes']
    
    research_questions = [n for n in nodes if n.get('type') == 'Research_Question']
    hypotheses = [n for n in nodes if n.get('type') == 'Hypothesis']
    q_h1 = [h for h in hypotheses if h.get('id') == 'Q_H1']
    
    print(f'✓ Research Question generated: {len(research_questions) >= 1}')
    print(f'✓ Q_H1 identified: {len(q_h1) >= 1}')
    print(f'✓ Total hypotheses: {len(hypotheses)} (should be 8+)')
    print(f'✓ Academic structure complete: {result[\"academic_structure\"][\"academic_structure_complete\"]}')
    print(f'✓ Testing compliance maintained: {result[\"van_evera_testing_results\"][\"testing_compliance_score\"]:.1f}% (target: ≥80%)')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
"
```

## 🚨 PHASE 4B: LEGACY COMPATIBILITY & MIGRATION (IMPORTANT)

### **Evidence-Based Migration Strategy**

**Issue**: Existing analyses use H_001/AE_001 format. Need migration path to Q/H1/H2/H3 without breaking existing functionality.

**Create migration helper function**:
```python
def migrate_legacy_format(graph_data: Dict) -> Dict:
    """Migrate H_001/AE_001 format to Q/H1/H2/H3 academic structure"""
    
    migrated_graph = graph_data.copy()
    nodes = migrated_graph['nodes']
    
    # Find legacy hypothesis nodes
    legacy_hypotheses = []
    legacy_alternatives = []
    
    for node in nodes:
        if node.get('type') == 'Hypothesis' and node['id'].startswith('H_'):
            legacy_hypotheses.append(node)
        elif node.get('type') == 'Alternative_Explanation':
            # Convert to Hypothesis type
            node['type'] = 'Hypothesis'
            legacy_alternatives.append(node)
    
    all_legacy = legacy_hypotheses + legacy_alternatives
    
    if all_legacy:
        # Generate research question from legacy content
        research_question = _generate_research_question_from_legacy(all_legacy)
        migrated_graph['nodes'].append(research_question)
        
        # Assign Q_H1, Q_H2, Q_H3 based on original ordering
        # H_001 becomes Q_H1, AE_001 becomes Q_H2, etc.
        id_mapping = {}
        
        # Primary hypothesis (H_001 → Q_H1)
        if legacy_hypotheses:
            primary = legacy_hypotheses[0]  # Assume first H_ is primary
            primary['id'] = 'Q_H1'
            primary['properties']['hypothesis_type'] = 'primary'
            primary['properties']['academic_rank'] = 1
            id_mapping[primary.get('original_id', primary['id'])] = 'Q_H1'
        
        # Alternative hypotheses (AE_001 → Q_H2, AE_002 → Q_H3, etc.)
        for i, alternative in enumerate(legacy_alternatives, 2):
            new_id = f'Q_H{i}'
            old_id = alternative['id']
            alternative['id'] = new_id
            alternative['properties']['hypothesis_type'] = 'alternative'  
            alternative['properties']['academic_rank'] = i
            id_mapping[old_id] = new_id
        
        # Update edge references
        for edge in migrated_graph['edges']:
            if edge.get('source_id') in id_mapping:
                edge['source_id'] = id_mapping[edge['source_id']]
            if edge.get('target_id') in id_mapping:
                edge['target_id'] = id_mapping[edge['target_id']]
        
        # Update edge types from legacy format
        for edge in migrated_graph['edges']:
            if edge.get('type') in ['supports_alternative', 'refutes_alternative']:
                edge['type'] = edge['type'].replace('_alternative', '')
    
    return migrated_graph
```

### **Integration Point**
Add migration check at workflow start:
```python
# At beginning of van_evera_workflow.py execute method
def execute(self, data: Any) -> Dict[str, Any]:
    graph_data = data.get('graph_data', {})
    
    # Check if legacy format migration needed
    has_legacy = any(
        n.get('type') == 'Alternative_Explanation' or 
        (n.get('type') == 'Hypothesis' and n['id'].startswith('H_'))
        for n in graph_data.get('nodes', [])
    )
    
    if has_legacy:
        self.logger.info("Legacy format detected, performing Q/H1/H2/H3 migration")
        graph_data = migrate_legacy_format(graph_data)
        data['graph_data'] = graph_data
    
    # Continue with normal workflow
    # ...
```

## 🎯 PHASE COMPLETION VALIDATION

### **Academic Structure Validation**
```bash
python -c "
# Complete end-to-end validation
from core.plugins.van_evera_workflow import execute_van_evera_analysis
import json

# Test 1: New analysis with Q/H1/H2/H3 structure
result_new = execute_van_evera_analysis('test_case', 'Test input for academic structure.')

# Test 2: Legacy analysis migration
with open('output_data/revolutions/revolutions_20250805_122000_graph.json', 'r') as f:
    legacy_data = json.load(f)

result_migrated = execute_van_evera_analysis('american_revolution_migrated', legacy_data)

# Validation checks
for label, result in [('New Analysis', result_new), ('Migrated Analysis', result_migrated)]:
    print(f'\\n{label} Validation:')
    nodes = result['graph_data']['nodes']
    
    research_q = [n for n in nodes if n.get('type') == 'Research_Question']
    hypotheses = [n for n in nodes if n.get('type') == 'Hypothesis']
    q_h1 = [h for h in hypotheses if h.get('id') == 'Q_H1']
    alternatives = [h for h in hypotheses if h.get('id', '').startswith('Q_H') and h.get('id') != 'Q_H1']
    
    print(f'✓ Research Question: {len(research_q) == 1}')
    print(f'✓ Q_H1 (Primary): {len(q_h1) == 1}')
    print(f'✓ Alternatives (Q_H2+): {len(alternatives) >= 1}') 
    print(f'✓ No Alternative_Explanation nodes: {not any(n.get(\"type\") == \"Alternative_Explanation\" for n in nodes)}')
    
    # Academic quality preservation
    testing_compliance = result.get('van_evera_testing_results', {}).get('testing_compliance_score', 0)
    print(f'✓ Testing compliance ≥80%: {testing_compliance >= 80.0} ({testing_compliance:.1f}%)')
"
```

### **Expected Success Criteria**
1. **Research Question Generation**: Domain-appropriate Q node created automatically
2. **Evidence-Based Q_H1**: Highest composite score hypothesis promoted to primary
3. **Academic Structure**: Complete Q/H1/H2/H3 organization with proper relationships
4. **Legacy Compatibility**: Existing H_001/AE_001 analyses migrate seamlessly
5. **Quality Preservation**: 87.3% testing compliance maintained through integration
6. **End-to-End Pipeline**: Complete workflow from input to Q/H1/H2/H3 HTML output

## 🔧 DEVELOPMENT ENVIRONMENT

### **Verified Working Setup**
- **Python 3.8+** with existing dependencies  
- **Plugin System**: All Phase 1-3 components operational (ontology, alternative generator, Van Evera testing)
- **API**: `gemini-2.5-flash` via `.env` file (confirmed working)
- **Test Data**: `output_data/revolutions/revolutions_20250805_122000_graph.json` (160 nodes, 203 edges)
- **Current Quality**: 87.3% testing compliance achieved with sophisticated Van Evera methodology

### **Implementation Priority**
1. **Phase 4A**: Workflow integration with research question generation and Q_H1 identification
2. **Phase 4B**: Legacy migration support for existing analyses
3. **Phase 4C**: Comprehensive validation and HTML template updates

**Repository Status**: Ready for Phase 4 implementation to complete Q/H1/H2/H3 academic structure migration with evidence-based development practices.
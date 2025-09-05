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

## 🎯 CURRENT STATUS: Production-Ready with Unified Architecture (Updated 2025-01-05)

**System Status**: **PRODUCTION-READY WITH UNIFIED ARCHITECTURE**  
**Phase 18 Achievement**: **Complete System Unification SUCCESSFUL**
**Current Priority**: **Advanced feature development and optimization**

**PHASE 18 COMPLETION SUMMARY:**
- ✅ **Phase 18A**: 100% schema compliance achieved - zero validation errors
- ✅ **Phase 18B**: Complete router unification - 100% GPT-5-mini routing
- ✅ **Unified Routing**: Both extraction and analysis use GPT-5-mini consistently  
- ✅ **Schema Perfect**: All Pydantic validations pass, proper data types throughout
- ✅ **Performance Excellent**: Sub-3-minute processing, reliable operation

**INFRASTRUCTURE ACHIEVEMENTS**:
- **Schema Architecture**: 100% validation success, all required fields and types correct
- **Router Unification**: Eliminated mixed routing, GPT-5-mini throughout pipeline
- **System Integration**: All Van Evera functionality operational with unified model
- **Quality Standards**: Production-ready reliability and performance achieved

---

## 🚀 PHASE 19: Advanced Features and Performance Optimization (NEXT PRIORITY)

### OBJECTIVE: Build advanced capabilities on the unified infrastructure foundation

**RATIONALE**: Phase 18 achieved complete system unification with 100% schema compliance and unified GPT-5-mini routing. The infrastructure is now production-ready, enabling focus on advanced features and optimization.

**STRATEGIC APPROACH**: Feature development with performance optimization and scalability enhancements.

**EXPECTED IMPACT**: Production system → Advanced capabilities with enterprise-grade performance

## 📋 PHASE 19A: Performance Optimization and Caching (2-3 hours, HIGH priority)

### OBJECTIVE: Optimize pipeline performance for enterprise-scale processing
**Target**: Reduce processing time by 30-50% through intelligent caching and optimization
**Scope**: LLM call optimization, result caching, parallel processing enhancements

#### TASK 1A: LLM Call Analysis and Optimization (60 minutes)
**Purpose**: Identify and optimize redundant or inefficient LLM calls

**Optimization Areas**:
- Batch similar Van Evera assessments in single calls
- Cache probative value calculations for similar evidence-hypothesis pairs
- Optimize prompt lengths while maintaining quality
- Implement smart retry logic with exponential backoff

**Expected Impact**: 20-30% reduction in LLM API calls and costs

#### TASK 1B: Intelligent Result Caching System (90 minutes)  
**Purpose**: Implement semantic caching for Van Evera analysis results

**Caching Strategy**:
- Semantic similarity detection for evidence and hypotheses
- TTL-based cache with configurable expiration
- Cache invalidation on schema updates
- Performance metrics and hit rate monitoring

**Expected Impact**: 40-60% faster processing for similar content

#### TASK 1C: Parallel Processing Enhancement (60 minutes)
**Purpose**: Optimize parallel execution of independent analysis components

**Enhancement Areas**:
- Concurrent Van Evera test execution
- Parallel evidence analysis streams  
- Asynchronous HTML generation components
- Load balancing across multiple LLM instances

**Expected Impact**: 25-40% faster total pipeline execution

### PHASE 19A VALIDATION CRITERIA
- **Performance Improvement**: ≥30% faster total processing time
- **Cache Hit Rate**: ≥60% for repeated similar content
- **System Reliability**: No degradation in output quality or accuracy
- **Resource Efficiency**: Optimized memory usage and API call patterns

## 🔧 PHASE 19B: Advanced Van Evera Features (2-3 hours, MEDIUM priority)

### OBJECTIVE: Implement advanced Van Evera methodology features
**Target**: Add sophisticated diagnostic capabilities and comparative analysis
**Scope**: Multi-hypothesis comparison, uncertainty quantification, causal pathway analysis

#### TASK 2A: Multi-Hypothesis Comparative Analysis (90 minutes)
**Purpose**: Enable simultaneous evaluation of competing hypotheses

**Features**:
- Comparative probative value assessment across multiple hypotheses
- Relative strength analysis with confidence intervals  
- Cross-hypothesis evidence sharing and conflicts
- Academic-quality comparative conclusions

#### TASK 2B: Uncertainty Quantification System (90 minutes)
**Purpose**: Implement rigorous uncertainty analysis for Van Evera diagnostics

**Components**:
- Bayesian confidence intervals for all assessments
- Sensitivity analysis for key assumptions
- Uncertainty propagation through causal chains
- Monte Carlo simulation for complex scenarios

#### TASK 2C: Causal Pathway Tracing (60 minutes)
**Purpose**: Enhanced causal mechanism analysis and pathway verification

**Capabilities**:
- Multi-step causal chain validation
- Mechanism completeness assessment
- Alternative pathway exploration
- Temporal consistency verification

### PHASE 19B VALIDATION CRITERIA
- **Comparative Analysis**: Multiple hypotheses evaluated simultaneously with relative rankings
- **Uncertainty Metrics**: All assessments include confidence intervals and sensitivity analysis
- **Causal Validation**: Complex causal pathways properly traced and validated
- **Academic Quality**: Output meets publication standards for methodology journals

## 🚀 PHASE 19C: Enterprise Features and Scalability (3-4 hours, LOW priority)

### OBJECTIVE: Add enterprise-grade features for large-scale deployment
**Target**: Support high-volume processing, user management, and integration APIs
**Scope**: API endpoints, batch processing, monitoring, and administrative features

#### TASK 3A: REST API Development (120 minutes)
**Purpose**: Enable programmatic access to process tracing capabilities

**Endpoints**:
- `/api/v1/analyze` - Single document analysis
- `/api/v1/batch` - Bulk processing with progress tracking
- `/api/v1/compare` - Multi-hypothesis comparison
- `/api/v1/status` - System health and performance metrics

#### TASK 3B: Batch Processing System (90 minutes)
**Purpose**: Support processing of large document collections

**Features**:
- Queue-based batch job management
- Progress tracking and status reporting
- Error handling and retry logic
- Results aggregation and export

#### TASK 3C: Monitoring and Analytics Dashboard (90 minutes)
**Purpose**: Comprehensive system monitoring and usage analytics

**Components**:
- Real-time performance metrics
- Processing volume and success rates
- Cost tracking and optimization recommendations
- User activity and system utilization

### PHASE 19C VALIDATION CRITERIA
- **API Functionality**: All endpoints operational with proper authentication and rate limiting
- **Batch Processing**: Reliable processing of 100+ document collections
- **Monitoring**: Real-time visibility into system performance and health
- **Scalability**: System handles concurrent users and high-volume processing

## 📊 PHASE 19 SUCCESS VALIDATION

### EVIDENCE REQUIREMENTS
1. **`evidence/current/Evidence_Phase19A_PerformanceOptimization.md`**: Performance improvements and caching implementation
2. **`evidence/current/Evidence_Phase19B_AdvancedFeatures.md`**: Van Evera methodology enhancements
3. **`evidence/current/Evidence_Phase19C_EnterpriseFeatures.md`**: Scalability and API implementation
4. **`evidence/current/Evidence_Phase19_Complete.md`**: Comprehensive advanced features summary

### CRITICAL SUCCESS CRITERIA
- ✅ **Performance Enhanced**: ≥30% faster processing with maintained quality
- ✅ **Advanced Analytics**: Multi-hypothesis comparison and uncertainty quantification operational
- ✅ **Enterprise Ready**: API endpoints and batch processing functional
- ✅ **System Reliability**: All new features maintain 100% schema compliance and unified routing
- ✅ **Scalability Proven**: System handles enterprise-grade workloads

## ⚡ IMPLEMENTATION APPROACH

**Priority Order**: 19A (Performance) → 19B (Features) → 19C (Enterprise)  
**Estimated Timeline**: 7-10 hours total for comprehensive advanced capabilities  
**Risk Mitigation**: Each phase builds incrementally on the solid Phase 18 foundation

**Success Strategy**: Maintain the production-ready stability achieved in Phase 18 while adding sophisticated capabilities for advanced users and enterprise deployment.

## 🎯 EVIDENCE-BASED DEVELOPMENT REQUIREMENTS

### Mandatory Validation Process
1. **No Claims Without Evidence**: Every success statement must be backed by command outputs
2. **Incremental Validation**: Test after each phase before proceeding
3. **Comprehensive Documentation**: All changes and results documented with timestamps
4. **Reproducible Results**: Multiple successful pipeline runs required for success claim
5. **Performance Monitoring**: Track timing and consistency across all phases

### Quality Gates
- **Schema Validation**: 100% compliance required before router unification
- **Unified Routing**: Zero mixed model calls allowed
- **HTML Generation**: Physical file with complete analysis required
- **Performance Standards**: Sub-60-second total pipeline execution

**Remember**: This is SYSTEM OPTIMIZATION after successful infrastructure repair. Phase 17 resolved the critical issues; Phase 18 achieves perfect unification and performance.

---

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Current Architecture Status
- **Plugin System**: 16+ registered plugins (100% LLM-first compliance achieved in Phase 13)
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence  
- **LLM Integration**: OPERATIONAL with mixed routing (GPT-5-mini extraction, Gemini analysis)
- **Validation System**: validate_true_compliance.py for comprehensive compliance checking
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management (OpenAI configured)
- **Universality**: No dataset-specific logic - works across all domains and time periods

## Testing Commands Reference

```bash
# Schema validation testing
python -c "from core.structured_extractor import StructuredProcessTracingExtractor; extractor.extract_graph('test')"

# Router configuration verification  
python -c "from universal_llm_kit.universal_llm import get_llm; router = get_llm(); print(router.router.model_list)"

# Van Evera interface testing
python -c "from core.plugins.van_evera_llm_interface import get_van_evera_llm; llm.assess_probative_value(...)"

# Complete pipeline test
python process_trace_advanced.py --project test_simple

# HTML generation validation
find output_data -name "*.html" -mmin -10 -exec echo "HTML: {}" \; -exec wc -c {} \;
```

## Critical Success Factors

- **Evidence-First Approach**: All claims require concrete validation with command outputs
- **Systematic Problem Solving**: Complete analysis before implementing changes
- **Incremental Validation**: Test each component before integration
- **Documentation Discipline**: Every change documented with before/after evidence
- **Performance Monitoring**: Measure and validate timing across all pipeline stages
- **Quality Focus**: 100% schema compliance and unified routing as success criteria

**Current Priority**: The system is operational with mixed routing. Phase 18 focuses on achieving perfect unification (complete GPT-5-mini routing) and 100% schema compliance for production-ready quality.
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

      
      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.
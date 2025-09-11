# CLAUDE.md Update Command - Evidence-Based Development Workflow

## Overview
commit so that we can rollback to this point if we need to  then update CLAUDE.md to clear out resolved tasks and populate it with instructions for resolving the next tasks using evidence-based development practices. The instructions should be detailed enough for a new LLM to implement with no context beyond CLAUDE.md and referenced files.

## Core CLAUDE.md Requirements

### 1. PERMANENT INFORMATION Section (MANDATORY - NEVER MODIFY)
Every CLAUDE.md must include at the top after the header:
```markdown
## 🚨 PERMANENT INFORMATION -- DO NOT CHANGE ON UPDATES

### LLM-First Architecture Policy (MANDATORY)

**CORE PRINCIPLE**: This system is **LLM-FIRST** with **ZERO TOLERANCE** for rule-based or keyword-based implementations.

**PROHIBITED IMPLEMENTATIONS**:
- ❌ Keyword matching for evidence classification
- ❌ Hardcoded probative value assignments  
- ❌ Rule-based contradiction detection
- ❌ Domain classification using keyword lists
- ❌ Any if/elif chains for semantic understanding

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ Structured Pydantic outputs for ALL semantic decisions

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification.
```

### 2. Coding Philosophy Section (Mandatory)
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


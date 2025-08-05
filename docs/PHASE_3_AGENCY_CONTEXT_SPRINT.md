# Phase 3: Agency & Context Sprint

## 🎯 Sprint Objective
**Goal**: Complete academic process tracing methodology with actor intentions, strategic interactions, and scope conditions

**Current Problem**: Missing human agency and contextual factors that drive causal processes. Process tracing often requires understanding WHY actors made specific decisions and UNDER WHAT CONDITIONS mechanisms operate.

**Success Criteria**:
- Extract Actor nodes with intentions, beliefs, and constraints
- Extract Condition nodes for scope conditions and enabling factors
- Link actors to events via initiates edges
- Link conditions to mechanisms via enables/constrains edges

## 📊 Impact Assessment
**Methodology Coverage**: 95% → 100% (+5%)
**Key Academic Methods Enabled**:
- ✅ Strategic interaction analysis
- ✅ Scope condition specification  
- ✅ Counterfactual reasoning with actor alternatives
- ✅ Complete academic process tracing methodology suite

## 🏗️ Implementation Tasks

### 1. New Node Type: Actor
**Properties**:
- name: Actor identification
- role: Position or function in events
- intentions: Goals and motivations
- beliefs: What they believe to be true
- constraints: Limitations on their actions
- capabilities: What they can actually accomplish

**Purpose**: Model human agency and decision-making that drives causal processes

### 2. New Node Type: Condition  
**Properties**:
- description: Condition description
- type: background/enabling/constraining/scope
- necessity: How necessary for outcomes (0.0-1.0)
- temporal_scope: When condition applies
- spatial_scope: Where condition applies

**Purpose**: Model background factors and scope conditions that enable or constrain causal mechanisms

### 3. Agency & Context Edge Types

**initiates**:
- Domain: [Actor] → Range: [Event]
- Properties: intentionality, capability_assessment, constraint_factors
- Purpose: Actor-driven event causation with strategic reasoning

**enables**:
- Domain: [Condition] → Range: [Event, Causal_Mechanism]
- Properties: necessity, temporal_scope, enabling_type
- Purpose: Conditions that make events/mechanisms possible

**constrains**:
- Domain: [Condition] → Range: [Event, Causal_Mechanism]  
- Properties: constraint_strength, temporal_scope, constraint_type
- Purpose: Conditions that limit or prevent events/mechanisms

## 🧪 Test Cases

### Test Case 1: American Revolution Actor Modeling
**Expected Actors**:
```json
{
  "id": "actor_george_washington",
  "type": "Actor",
  "properties": {
    "name": "George Washington",
    "role": "Continental Army Commander",
    "intentions": "Achieve colonial independence while maintaining legitimacy",
    "beliefs": "Military victory requires avoiding major battles until strong enough",
    "constraints": "Limited resources, untrained militia, British naval superiority",
    "capabilities": "Military leadership, strategic planning, political legitimacy"
  }
}
```

**Expected Actor-Event Connections**:
```json
{
  "type": "initiates",
  "source": "actor_george_washington",
  "target": "event_strategic_retreat_1776",
  "properties": {
    "intentionality": "strategic",
    "capability_assessment": 0.8,
    "constraint_factors": "Limited army size, British pursuit"
  }
}
```

### Test Case 2: Scope Conditions
**Expected Conditions**:
```json
{
  "id": "condition_british_naval_dominance", 
  "type": "Condition",
  "properties": {
    "description": "British naval superiority limits colonial military options",
    "type": "constraining",
    "necessity": 0.7,
    "temporal_scope": "1775-1778"
  }
}
```

**Expected Condition-Mechanism Connections**:
```json
{
  "type": "constrains",
  "source": "condition_british_naval_dominance",
  "target": "mechanism_guerrilla_warfare_strategy",
  "properties": {
    "constraint_strength": 0.8,
    "constraint_type": "strategic_limitation"
  }
}
```

### Test Case 3: Strategic Interaction
**Expected Multi-Actor Analysis**:
- British Government: intentions to maintain control, beliefs about colonial submission
- Colonial Leaders: intentions for independence, beliefs about popular support
- French Government: intentions to weaken Britain, beliefs about colonial viability

## 🔍 Quality Criteria

### Actor Modeling Quality
- **Psychological Realism**: Intentions and beliefs reflect historical evidence
- **Constraint Recognition**: Limitations on actor capabilities properly modeled
- **Strategic Reasoning**: Actor decisions explained by intentions + constraints
- **Historical Accuracy**: Actor characterizations supported by primary sources

### Condition Modeling Quality  
- **Scope Clarity**: Clear temporal and spatial boundaries
- **Causal Relevance**: Conditions actually affect mechanism operation
- **Necessity Assessment**: Realistic evaluation of how necessary conditions are
- **Type Distinction**: Clear difference between enabling vs constraining conditions

### Strategic Interaction Analysis
- **Multi-Actor Dynamics**: Interaction between actor intentions
- **Belief Updating**: How actor beliefs change based on events
- **Strategic Responses**: How actors respond to other actor actions
- **Unintended Consequences**: Gap between intentions and outcomes

## 📋 Definition of Done

### Technical Completion
- [ ] Actor nodes extracted with intentions, beliefs, constraints
- [ ] Condition nodes extracted with proper type classification
- [ ] initiates edges connecting actors to events
- [ ] enables/constrains edges connecting conditions to mechanisms
- [ ] All agency and context properties properly populated

### Quality Validation  
- [ ] American Revolution case generates 4-6 key actor nodes
- [ ] Actors have realistic intentions, beliefs, and constraints
- [ ] Scope conditions properly identified and typed
- [ ] Strategic interaction patterns visible in actor-event connections
- [ ] No trivial or anachronistic actor characterizations

### Academic Methodology
- [ ] Strategic interaction analysis functional
- [ ] Scope condition specification operational
- [ ] Counterfactual reasoning with actor alternatives supported
- [ ] Complete process tracing methodology suite integrated

## 🧪 Extended Test Cases

### Multi-Actor Strategic Analysis
**British Parliamentary Faction Analysis**:
- Hardliners: Intentions to punish rebellion, beliefs about deterrence
- Moderates: Intentions to reconcile, beliefs about colonial grievances  
- King George III: Intentions to maintain authority, beliefs about divine right

**Colonial Leadership Spectrum**:
- Radicals (Adams): Intentions for full independence, beliefs about popular support
- Moderates (Dickinson): Intentions for reconciliation, beliefs about compromise
- Conservatives (Loyalists): Intentions to maintain connection, beliefs about benefits

### Scope Condition Analysis
**Economic Conditions**:
- Colonial Economic Growth: Enables resistance funding
- British Debt Crisis: Constrains military spending
- Atlantic Trade Networks: Enable/constrain both sides

**Political Conditions**:
- Enlightenment Ideas: Enable legitimation of resistance
- Traditional Monarchy: Constrains British flexibility
- Colonial Self-Governance Experience: Enables organizational capacity

## 🚨 Risk Mitigation

### Risk 1: Over-Psychologizing Historical Actors
**Mitigation**: Focus on intentions/beliefs explicitly stated in historical sources

### Risk 2: Anachronistic Actor Characterizations  
**Mitigation**: Ground all actor properties in contemporary evidence and values

### Risk 3: Condition Over-Specification
**Mitigation**: Focus on conditions that actually affected mechanism operation

### Risk 4: Strategic Interaction Complexity
**Mitigation**: Start with bilateral interactions, build toward multi-actor dynamics

## 📈 Success Metrics

### Quantitative Targets
- **Actor Nodes**: 4-6 per case (major decision-makers)
- **Condition Nodes**: 3-5 per case (key scope conditions)
- **initiates Edges**: 6-10 (actor-driven events)
- **enables/constrains Edges**: 4-8 (condition-mechanism connections)

### Qualitative Assessment
- Actors have psychologically realistic intentions and beliefs
- Conditions actually affect mechanism operation  
- Strategic interactions explain event patterns
- Scope conditions properly delimit mechanism operation

## 🔄 Implementation Sequence

### Sprint 3A: Basic Actor Modeling
- Add Actor nodes to extraction prompt
- Test intention and belief extraction
- Validate historical accuracy and psychological realism

### Sprint 3B: Actor-Event Integration
- Implement initiates edges
- Test strategic reasoning in actor-event connections
- Validate capability assessment and constraint factors

### Sprint 3C: Scope Condition Modeling  
- Add Condition nodes to extraction
- Test condition type classification
- Validate temporal and spatial scope specification

### Sprint 3D: Condition-Mechanism Integration
- Implement enables/constrains edges
- Test condition effects on mechanism operation
- Final integration testing and complete methodology validation

## 🎓 Academic Impact

### Complete Methodology Suite
With Phase 3 completion, the system will support:
- **Van Evera Diagnostic Tests**: All four types with mechanism application
- **Beach & Pedersen Theory-Testing**: Complete mechanism analysis
- **George & Bennett Congruence**: Systematic confirming/disconfirming evidence
- **Explaining-Outcome Process Tracing**: Alternative elimination with actor analysis
- **Strategic Interaction Analysis**: Multi-actor decision-making patterns
- **Scope Condition Specification**: Boundary conditions for causal claims

### Research Applications
- Historical case analysis with complete causal modeling
- Comparative case studies with standardized analytical framework
- Policy analysis with actor intention and constraint modeling
- Counterfactual analysis with alternative actor decision scenarios
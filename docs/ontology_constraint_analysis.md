# Ontology Constraint Analysis: Edge Type Usage and Directed Property Graph Implications

## Critical Issues with Current Edge Type Usage

Based on analysis of the ontology configuration and current extraction results, there are several significant constraint violations and design issues affecting the methodological validity of the missing edge types.

## 1. `confirms_occurrence` - **CORRECTLY DESIGNED** ✅

**Current Ontology Definition**:
```json
"confirms_occurrence": {
  "domain": ["Evidence"],
  "range": ["Event"],
  "properties": {
    "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": false},
    "source_text_quote": {"type": "string", "required": false},
    "diagnostic_type": {"type": "string", "allowed_values": ["hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "general"], "required": false}
  }
}
```

**Methodological Assessment**: ✅ **CORRECTLY CONSTRAINED**
- Domain restriction to Evidence is methodologically sound
- Range restriction to Event aligns with Van Evera framework
- Diagnostic type properties support proper Van Evera categorization
- Directed property graph direction (Evidence → Event) matches causal logic

---

## 2. `infers` - **MAJOR CONSTRAINT PROBLEM** ❌

**Current Ontology Definition**:
```json
"infers": {
  "domain": ["Inference_Rule"],
  "range": ["Hypothesis", "Causal_Mechanism"],
  "properties": {
    "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": false},
    "logic_type": {"type": "string", "required": false}
  }
}
```

**CRITICAL ISSUE**: Domain restriction to `Inference_Rule` only is **methodologically invalid**

**Problems**:
1. **Missing Evidence Domain**: Most process tracing inference comes from Evidence → Hypothesis
2. **Missing Event Domain**: Events often allow inference to mechanisms/hypotheses  
3. **Inference_Rule Overconstraint**: This forces artificial creation of Inference_Rule nodes

**Methodologically Correct Domain Should Be**: `["Evidence", "Event", "Inference_Rule"]`

**Example of Current Constraint Breaking Process Tracing**:
```
❌ Current: Must create artificial Inference_Rule → infers → Hypothesis
✅ Should be: Evidence → infers → Hypothesis (direct inferential reasoning)
```

---

## 3. `refutes` - **CORRECTLY DESIGNED** ✅

**Current Ontology Definition**:
```json
"refutes": {
  "domain": ["Evidence", "Event"],
  "range": ["Hypothesis", "Event", "Causal_Mechanism"],
  "properties": {
    "diagnostic_type": {"type": "string", "allowed_values": ["hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "general"], "required": false}
  }
}
```

**Methodological Assessment**: ✅ **CORRECTLY CONSTRAINED**
- Domain includes both Evidence and Event (methodologically sound)
- Range covers all primary targets of refutation
- Diagnostic type properties support Van Evera framework
- Directed graph allows proper falsification chains

---

## 4. `tests_alternative` - **CORRECTLY DESIGNED** ✅

**Current Ontology Definition**:
```json
"tests_alternative": {
  "domain": ["Evidence", "Event"],
  "range": ["Alternative_Explanation"],
  "properties": {
    "diagnostic_type": {"type": "string", "allowed_values": ["hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "general"], "required": false},
    "test_result": {"type": "string", "allowed_values": ["supports", "refutes", "inconclusive"], "required": false}
  }
}
```

**Methodological Assessment**: ✅ **CORRECTLY CONSTRAINED**
- Domain includes Evidence and Event (George & Bennett methodology)
- Range restricted to Alternative_Explanation (specific competitive testing)
- Test result property captures outcome (supports/refutes/inconclusive)
- Directed graph enables systematic alternative evaluation

---

## 5. `weighs_evidence` - **DESIGN LIMITATION** ⚠️

**Current Ontology Definition**:
```json
"weighs_evidence": {
  "domain": ["Evidence"],
  "range": ["Evidence"],
  "properties": {
    "comparison_strength": {"type": "float", "min": 0.0, "max": 1.0, "required": false},
    "comparison_type": {"type": "string", "allowed_values": ["stronger_than", "weaker_than", "equivalent_to", "complements", "contradicts"], "required": false}
  }
}
```

**Methodological Assessment**: ⚠️ **POTENTIALLY PROBLEMATIC**

**Issue**: Evidence → Evidence constraint may be too restrictive for full Beach & Pedersen methodology

**Problems**:
1. **Missing Evidence → Hypothesis weighing**: Evidence quality affects hypothesis credibility
2. **Meta-analytical limitation**: Cannot capture evidence quality impact on theoretical claims

**Possible Enhancement**: Consider expanding range to `["Evidence", "Hypothesis", "Causal_Mechanism"]`

---

## Directed Property Graph Implications

### **Positive Aspects**:
1. **Causal Direction**: Directed edges properly capture causal/evidential flow
2. **Asymmetric Relations**: Van Evera diagnostic tests are inherently directional
3. **Inference Chains**: Support complex reasoning chains (Evidence → Hypothesis → Mechanism)

### **Constraint Benefits**:
1. **Domain/Range Validation**: Prevents methodologically invalid connections
2. **Property Typing**: Ensures consistent diagnostic categorization
3. **Structured Reasoning**: Forces explicit evidential relationships

### **Potential Issues**:
1. **Over-Constraint**: `infers` domain restriction breaks natural reasoning patterns
2. **Bidirectional Relations**: Some process tracing relationships might need bidirectionality
3. **Meta-Relations**: `weighs_evidence` constraint may limit meta-analytical capabilities

## Recommendations

### **Immediate Fix Required**: 
**`infers` Domain Expansion**
```json
"infers": {
  "domain": ["Evidence", "Event", "Inference_Rule"],  // ADD Evidence and Event
  "range": ["Hypothesis", "Causal_Mechanism"]
}
```

### **Consider for Enhancement**:
**`weighs_evidence` Range Expansion**
```json
"weighs_evidence": {
  "domain": ["Evidence"],
  "range": ["Evidence", "Hypothesis", "Causal_Mechanism"]  // ADD theoretical targets
}
```

### **Current Status Summary**:
- ✅ 3/5 edge types correctly constrained (`confirms_occurrence`, `refutes`, `tests_alternative`)
- ❌ 1/5 edge type has major constraint problem (`infers`)  
- ⚠️ 1/5 edge type has potential limitation (`weighs_evidence`)

**The `infers` constraint violation likely explains why this edge type is not appearing in extractions - the domain restriction makes it nearly impossible to use in natural process tracing analysis.**
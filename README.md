# Process Tracing Toolkit (Advanced LLM-Enhanced)

## Overview

**Production-Ready System** for advanced qualitative analysis using process tracing methodology with comprehensive Van Evera diagnostic tests, LLM-enhanced evidence assessment, and optional Bayesian uncertainty analysis.

## ✅ Current Status: Production Ready

- **Core Functionality**: Fully operational Van Evera process tracing pipeline
- **Plugin Architecture**: 16 registered plugins with proper abstractions
- **Van Evera Integration**: Complete diagnostic test methodology implementation
- **LLM Integration**: Structured output with Gemini 2.5 Flash and Pydantic validation
- **Interactive Reports**: Rich HTML dashboards with network visualizations

## 🚀 Key Features

### **LLM-Powered Graph Extraction**
- Extracts detailed causal process tracing graphs from text using Gemini API
- Comprehensive ontology with 10 node types and 16 relationship types
- Structured output with full validation and error handling

### **Van Evera Diagnostic Tests**
- **Hoop Tests**: Necessary conditions for hypothesis validity
- **Smoking Gun Tests**: Sufficient evidence for hypothesis confirmation
- **Doubly Decisive Tests**: Both necessary and sufficient evidence
- **Straw-in-the-Wind Tests**: Weak but cumulative evidence
- **Bayesian Tests**: Probabilistic evidence assessment

### **LLM-Enhanced Evidence Analysis**
- Evidence type refinement and classification
- Probative value assessment with justification
- Bayesian likelihood calculations (P(E|H) and P(E|¬H))
- Textual reasoning and source quotation extraction

### **Advanced Mechanism Analysis**
- Causal mechanism completeness assessment
- Missing micro-step identification
- Coherence evaluation and narrative generation
- Van Evera mechanism validation

### **Bayesian Uncertainty Analysis** (Optional)
- Monte Carlo simulation for confidence intervals
- Prior probability assignment and updating
- Posterior probability calculation
- Comprehensive uncertainty quantification

### **Interactive Visualizations**
- Network graph visualizations with vis.js
- Evidence strength charts and node type distributions
- Causal pathway analysis and centrality metrics
- Comparative analysis dashboards

## 📁 Directory Structure

```
├── CLAUDE.md                    # Development guidance
├── README.md                    # This file
├── process_trace_advanced.py    # 🎯 Main entry point
├── process_trace_bayesian.py    # Bayesian analysis entry
├── process_trace_comparative.py # Comparative analysis
├── study_config.json           # Configuration
│
├── core/                       # Core analysis modules
│   ├── analyze.py              # Main analysis engine
│   ├── extract.py              # Text-to-graph extraction
│   ├── enhance_evidence.py     # Evidence enhancement
│   ├── enhance_mechanisms.py   # Mechanism analysis
│   ├── bayesian_*.py          # Bayesian components
│   └── plugins/               # Plugin architecture
│
├── config/                     # Configuration files
├── input_text/                 # Input documents
│   └── revolutions/           # American Revolution texts
├── output_data/               # Analysis results
├── docs/                      # Documentation
│   ├── phases/               # Development phases
│   ├── guides/               # User guides
│   ├── testing/              # Test scripts
│   └── validation/           # Validation tools
│
└── temp_results/              # Temporary outputs
```

## 🎯 Quick Start

### **1. Setup**
```bash
# Install dependencies
pip install google-genai networkx matplotlib python-dotenv pydantic scipy numpy

# Set up API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### **2. Run American Revolution Analysis**
```bash
# Basic analysis
python process_trace_advanced.py --project revolutions

# Enhanced Bayesian analysis
python process_trace_advanced.py --project revolutions --bayesian

# High-quality analysis with uncertainty quantification
python process_trace_advanced.py --project revolutions --bayesian --simulations 2000 --confidence-level 0.99
```

### **3. Custom Analysis**
```bash
# Analyze your own document
# 1. Place text file in input_text/your_project/
# 2. Run analysis
python process_trace_advanced.py --project your_project --bayesian
```

## 📊 Output

The system generates:
- **Interactive HTML Report**: Comprehensive analysis dashboard
- **Network Visualization**: Interactive causal graph
- **JSON Summary**: Structured analysis results
- **Evidence Tables**: Van Evera diagnostic assessments
- **Mechanism Analysis**: Causal pathway evaluation

Results are saved in `output_data/[project_name]/`

## 🧪 Advanced Features

### **Bayesian Configuration Options**
```bash
# Skip uncertainty analysis (faster)
--no-uncertainty

# Custom simulation count
--simulations 5000

# Different confidence levels
--confidence-level 0.99

# Disable visualizations
--no-visualizations
```

### **Analysis Modes**
```bash
# Extract graph only
--extract-only

# Analyze existing graph
--analyze-only --graph-file path/to/graph.json

# Comparative analysis
--comparative --case-files case1.json case2.json
```

## 🔬 System Validation

Core functionality verified:
- ✅ Ontology loads from configuration
- ✅ Evidence balance calculations correct
- ✅ Graph processing preserves data integrity
- ✅ Path finding optimized for performance
- ✅ Enhancement processing runs once per analysis

## 🛠️ Development

### **Test Suite**
```bash
# Run core verification
python docs/testing/test_all_critical_fixes.py

# Individual test scripts in docs/testing/
```

### **Documentation**
- `docs/guides/` - User guides and references
- `docs/phases/` - Development phase documentation  
- `docs/development/` - Technical development notes
- `CLAUDE.md` - Comprehensive development guidance

## 📋 Requirements

- **Python 3.8+**
- **Dependencies**: google-genai, networkx, matplotlib, python-dotenv, pydantic, scipy, numpy
- **API Key**: Google Gemini API key (required)
- **Memory**: 4GB+ recommended for large documents
- **Disk**: 1GB+ for output storage

## 🎓 Methodology

This toolkit implements:
- **Process Tracing**: Detailed causal pathway analysis
- **Van Evera Tests**: Four diagnostic test types for evidence evaluation
- **Bayesian Inference**: Probabilistic reasoning and uncertainty quantification
- **Structured Analysis**: Systematic hypothesis testing and mechanism validation

## 📚 Academic Applications

Ideal for:
- **Political Science**: Policy analysis, institutional change, conflict studies
- **History**: Causal analysis of historical processes and events
- **Sociology**: Social mechanism analysis and theory testing
- **Economics**: Policy impact assessment and causal inference
- **Public Policy**: Program evaluation and outcome analysis

## 🤝 Support

- **Issues**: Check `docs/debug/` for troubleshooting
- **Validation**: Use `docs/validation/` tools for system verification
- **Testing**: Run `docs/testing/` scripts to verify functionality

---

**Ready for production use** with comprehensive process tracing analysis, Van Evera methodology, and optional Bayesian enhancements.
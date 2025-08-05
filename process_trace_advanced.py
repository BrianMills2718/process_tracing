# This script loads the Gemini API key from a .env file if not set in the environment.
# Create a .env file with: GOOGLE_API_KEY=your_key_here

import os
import sys
import json
import datetime
from pathlib import Path
import argparse
import glob
import subprocess
import time

# --- Load .env if present ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, but recommended

# --- Environment and Dependency Check ---
try:
    import google.generativeai as genai
except ImportError:
    print("[ERROR] google-generativeai is not installed. Please activate the correct environment and install it.")
    sys.exit(1)

from core.ontology import get_gemini_graph_json_schema

# Bayesian Integration (optional enhancement)
try:
    from core.bayesian_integration import integrate_bayesian_analysis
    from core.bayesian_reporting import BayesianReportConfig
    from core.enhance_evidence import refine_evidence_assessment_with_llm
    BAYESIAN_AVAILABLE = True
except ImportError as e:
    BAYESIAN_AVAILABLE = False
    print(f"[INFO] Bayesian components not available due to import error: {e}")
    print("Use --bayesian flag to see setup instructions.")

# Add a safe print function to handle UnicodeEncodeError
def safe_print(*args, **kwargs):
    """Windows-compatible print function that handles Unicode encoding issues"""
    import re
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace Unicode characters that cause issues on Windows
        cleaned_args = []
        for arg in args:
            text = str(arg)
            # Replace common Unicode characters with ASCII equivalents
            text = re.sub(r'[→⇒←⇄]', '->', text)  # Arrows
            text = re.sub(r'[✅]', '[OK]', text)   # Checkmarks
            text = re.sub(r'[❌]', '[ERROR]', text) # X marks
            text = re.sub(r'[⚠️]', '[WARN]', text)  # Warning
            text = re.sub(r'[🔄]', '[PROCESSING]', text) # Processing
            text = re.sub(r'[📊]', '[DATA]', text)  # Data
            text = re.sub(r'[🎯]', '[TARGET]', text) # Target
            # Remove any remaining non-ASCII characters
            text = text.encode('ascii', errors='replace').decode('ascii')
            cleaned_args.append(text)
        print(*cleaned_args, **kwargs)

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def list_projects(input_text_dir):
    return [d for d in os.listdir(input_text_dir)
            if os.path.isdir(os.path.join(input_text_dir, d)) and not d.startswith('.')]

def prompt_for_project(projects):
    print("Available projects:")
    for i, proj in enumerate(projects, 1):
        print(f"  {i}. {proj}")
    while True:
        choice = input(f"Select a project [1-{len(projects)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(projects):
            return projects[int(choice)-1]
        print("Invalid selection. Please try again.")

def find_first_txt(project_dir):
    txts = glob.glob(str(Path(project_dir) / '*.txt'))
    if not txts:
        print(f"ERROR: No .txt files found in {project_dir}")
        sys.exit(1)
    return txts[0]

def load_prompt(prompt_path="prompts/extraction_prompt.txt"):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Export/Import Graph Utilities ---
def export_graph_for_editing(graph_json_path, output_dir):
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    with open(os.path.join(output_dir, 'editable_nodes.json'), 'w', encoding='utf-8') as f:
        json.dump(nodes, f, indent=2)
    with open(os.path.join(output_dir, 'editable_edges.json'), 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2)

def import_edited_graph(edited_nodes_path, edited_edges_path, output_path):
    with open(edited_nodes_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    with open(edited_edges_path, 'r', encoding='utf-8') as f:
        edges = json.load(f)
    graph = {'nodes': nodes, 'edges': edges}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2)

# --- Read Input Text ---
def read_input_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Build Gemini Schema ---
def get_schema():
    return get_gemini_graph_json_schema()

# --- Focused Extraction Prompt (TDD Phase 1 Implementation) ---
FOCUSED_EXTRACTION_PROMPT = """
Extract causal events and hypotheses from the historical text with detailed descriptions.

CRITICAL REQUIREMENTS:
1. Each Event description must be extracted from the source text (minimum 20 words)
2. Each Hypothesis must be a testable causal claim (minimum 30 words)
3. Each Evidence must include the exact source text quote that supports it
4. Descriptions cannot be "N/A", "Description_Not_Found", or generic placeholders

EXAMPLE (GOOD):
Event: "French and Indian War ends with defeat of France, leaving Britain with massive war debt that requires new colonial taxation policies"
Hypothesis: "British taxation policies caused colonial rebellion because they violated colonial understanding of their rights as Englishmen"
Evidence: "taxation without representation violated their rights as Englishmen"

EXAMPLE (BAD):
Event: "N/A"
Hypothesis: "Colonial discontent"
Evidence: "Colonists were unhappy"

STEP-BY-STEP EXTRACTION:

1. **Extract Events with Rich Descriptions:**
   - Find 3-8 major historical events
   - Each Event needs detailed description from source text
   - Classify as: triggering (starts process), intermediate (middle), outcome (end result)
   - Connect with "causes" edges to show sequence

2. **Extract Hypotheses as Testable Claims:**
   - Find causal claims like "X caused Y because Z"
   - Minimum 30 words explaining the causal mechanism
   - Must be testable with evidence from the text

3. **Extract Evidence with Source Quotes:**
   - For each Hypothesis, find supporting/refuting evidence with DIVERSE diagnostic strength
   - Include exact quote from source text
   - CRITICAL: Set "type" field to Van Evera diagnostic type - aim for variety across all 4 types:
     * "hoop": necessary but not sufficient (hypothesis fails if this evidence is absent) - look for foundational requirements
     * "smoking_gun": sufficient but not necessary (hypothesis confirmed if this evidence is present) - look for definitive proof
     * "straw_in_the_wind": neither necessary nor sufficient (weakly suggestive) - look for circumstantial indicators  
     * "doubly_decisive": both necessary and sufficient (confirms one hypothesis, eliminates others) - look for critical turning points
   - Prioritize finding strong evidence types (smoking_gun, doubly_decisive) when the text supports them
   - Estimate probative_value (0.3-1.0 for meaningful evidence)

4. **Connect Everything with Flexible Academic Process Tracing:**
   - Events → Events (causes) - for causal sequences
   - Evidence → Events (confirms_occurrence/disproves_occurrence) - confirming events happened
   - Evidence → Hypotheses (supports/refutes) - testing causal claims
   - Events → Hypotheses (provides_evidence_for) - events as evidence for broader patterns
   - Evidence → Mechanisms (tests_mechanism) - testing how causation works
   - Use diagnostic_type property on all evidence connections when possible
   - Ensure complete chain from triggering to outcome events

{global_hypothesis_section}

Focus on quality over quantity - better to have 5 well-described entities than 20 with "N/A" descriptions.

**Output Format:**
Return JSON with 'nodes' and 'edges' lists. Use flexible connection types for rigorous process tracing:

**Available Edge Types:**
- "causes" (Event → Event): Causal sequences
- "supports" (Evidence/Event → Hypothesis/Event/Mechanism): Supporting evidence 
- "refutes" (Evidence/Event → Hypothesis/Event/Mechanism): Refuting evidence
- "confirms_occurrence" (Evidence → Event): Evidence confirms event happened
- "provides_evidence_for" (Event → Hypothesis/Mechanism): Event serves as evidence
- "tests_hypothesis" (Evidence/Event → Hypothesis): Formal hypothesis testing

**Edge Properties to Include:**
- diagnostic_type: "hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive"
- probative_value: 0.0-1.0 strength assessment
- source_text_quote: Exact quote supporting the connection
- target_type: "event_occurrence", "causal_relationship", "mechanism_operation"

Example connections:
{{
  "source": "evidence1", "target": "event1", "type": "confirms_occurrence",
  "properties": {{
    "diagnostic_type": "smoking_gun",
    "source_text_quote": "Exact quote proving event occurred"
  }}
}},
{{
  "source": "event2", "target": "hypothesis1", "type": "provides_evidence_for", 
  "properties": {{
    "diagnostic_type": "hoop",
    "probative_value": 0.7,
    "reasoning": "This event is necessary for the hypothesis to be true"
  }}
}}
"""

# --- Refactored Single-Case Processing Function ---
def execute_single_case_processing(case_file_path_str, output_dir_for_case_str, project_name_str, global_hypothesis_text=None, global_hypothesis_id=None, bayesian_config=None):
    import subprocess
    from datetime import datetime
    case_file_path = Path(case_file_path_str)
    output_dir_for_case = Path(output_dir_for_case_str)
    output_dir_for_case.mkdir(parents=True, exist_ok=True)
    # Read input text
    with open(case_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Issue #85 Fix: Hard fail if text exceeds 1M tokens (NO chunking)
    # Rough approximation: 1 token ≈ 4 characters for English text
    estimated_tokens = len(text) // 4
    MAX_TOKENS = 1_000_000
    
    if estimated_tokens >= MAX_TOKENS:
        print(f"[ERROR] Input text too large: ~{estimated_tokens:,} tokens (limit: {MAX_TOKENS:,})")
        print(f"[ERROR] File: {case_file_path}")
        print(f"[ERROR] Text length: {len(text):,} characters")
        print("[ERROR] Please reduce input size. Chunking is not supported for quality reasons.")
        sys.exit(1)
    
    print(f"[INFO] Input size: ~{estimated_tokens:,} tokens ({len(text):,} characters)")
    # Build schema
    schema = get_schema()
    # Format prompt
    active_prompt_template = FOCUSED_EXTRACTION_PROMPT
    # Default values for prompt placeholders
    gh_text_for_prompt = "Not Applicable (no global hypothesis specified for this run)"
    gh_id_for_prompt = "N/A"
    global_hypothesis_section_text = (
        "**Global Study Hypothesis to Consider for this Case:**\n"
        "No specific global hypothesis is pre-defined for this particular case analysis. "
        "Your task is to identify ALL hypotheses that emerge naturally from the provided text based on their causal claims or relation to causal mechanisms, "
        "and then diligently seek all relevant evidence for each of these emergent hypotheses.\n---"
    )
    if global_hypothesis_text and global_hypothesis_id:
        gh_text_for_prompt = global_hypothesis_text
        gh_id_for_prompt = global_hypothesis_id
        global_hypothesis_section_text = (
            f"**Global Study Hypothesis to Consider for this Case (CRITICAL FOCUS):**\n"
            f"- ID: {global_hypothesis_id}\n"
            f"- Text: {global_hypothesis_text}\n"
            f"You MUST create a 'Hypothesis' node with this exact ID and description. Then, you MUST diligently search the entire text for all `Evidence` that supports or refutes this specific global hypothesis and link it accordingly. This is a primary objective for this analysis.\n---"
        )
    final_system_prompt = active_prompt_template.format(
        global_hypothesis_section=global_hypothesis_section_text,
        global_hypothesis_text_for_prompt=gh_text_for_prompt,
        global_hypothesis_id_for_prompt=gh_id_for_prompt
    )
    # Two-pass extraction with connectivity repair
    from core.extract import parse_json, analyze_graph_connectivity, create_connectivity_repair_prompt, extract_connectivity_relationships
    
    # Save raw output path for compatibility
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_json_path = output_dir_for_case / f"{project_name_str}_{now_str}_graph.json"
    
    try:
        print("\n" + "="*60)
        print("PERFORMING TWO-PASS GRAPH EXTRACTION")
        print("="*60)
        
        # Pass 1: Standard extraction
        print("[INFO] Pass 1: Standard graph extraction...")
        raw_json = query_llm(text, schema, final_system_prompt)
        graph_data = parse_json(raw_json)
        
        # Analyze connectivity
        print("[INFO] Analyzing graph connectivity...")
        connectivity = analyze_graph_connectivity(graph_data)
        
        if not connectivity['needs_repair']:
            print(f"[INFO] Graph connectivity acceptable: {connectivity['disconnection_rate']:.1%} disconnection rate")
        else:
            print(f"[INFO] Graph needs connectivity repair: {connectivity['disconnection_rate']:.1%} disconnection rate")
            print(f"[INFO] Found {len(connectivity['isolated_nodes'])} isolated nodes and {len(connectivity['small_components'])} small components")
            
            # Pass 2: Connectivity repair
            if connectivity['disconnected_entity_details']:
                print("[INFO] Pass 2: Connectivity repair...")
                print(f"[INFO] Attempting to connect {len(connectivity['disconnected_entity_details'])} disconnected entities")
                main_graph_summary = f"Main graph has {connectivity['giant_component_size']} connected nodes including Events, Hypotheses, Evidence, and Mechanisms"
                
                repair_prompt = create_connectivity_repair_prompt(
                    text, 
                    connectivity['disconnected_entity_details'],
                    main_graph_summary
                )
                
                additional_edges = extract_connectivity_relationships(repair_prompt)
                
                if additional_edges:
                    print(f"[INFO] Found {len(additional_edges)} additional relationships")
                    # Merge edges into original graph
                    graph_data['edges'].extend(additional_edges)
                    
                    # Re-analyze connectivity
                    final_connectivity = analyze_graph_connectivity(graph_data)
                    print(f"[INFO] Final disconnection rate: {final_connectivity['disconnection_rate']:.1%}")
                    if final_connectivity['total_components'] == 1:
                        print("[SUCCESS] Graph is now fully connected!")
                    else:
                        print(f"[INFO] Reduced to {final_connectivity['total_components']} components")
                else:
                    print("[WARNING] No additional relationships found in connectivity repair")
        
        # Save the enhanced graph data
        with open(graph_json_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
            
        print(f"\n[SAVE] Enhanced graph data saved to {graph_json_path}")
        
    except Exception as e:
        print(f"[ERROR] Two-pass extraction failed: {e}")
        print(f"[INFO] Falling back to single-pass extraction...")
        
        # Fallback to single-pass extraction
        try:
            raw_json = query_llm(text, schema, final_system_prompt)
            graph_data = parse_json(raw_json)
            with open(graph_json_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2)
        except Exception as fallback_error:
            print(f"[ERROR] Fallback extraction also failed: {fallback_error}")
            return None
    
    # Validate extraction connectivity
    try:
        from core.extraction_validator import validate_causal_connectivity, print_validation_report
        print("\n" + "="*60)
        print("VALIDATING EXTRACTED CAUSAL GRAPH")
        print("="*60)
        
        validation_results = validate_causal_connectivity(graph_data)
        print_validation_report(validation_results)
        
        if not validation_results['is_valid']:
            print("\n[WARNING] Extracted graph has connectivity issues that may prevent analysis.")
            print("Consider re-running extraction with enhanced connectivity requirements.")
            
    except ImportError:
        print("[INFO] Extraction validator not available, skipping validation.")
    except Exception as e:
        print(f"[WARNING] Validation failed: {e}")
            
    # Visualize as standalone HTML (keeping this for backward compatibility)
    html_path = output_dir_for_case / f"{project_name_str}_{now_str}_graph.html"
    visualize_graph(graph_data, html_path, project_name_str)
    
    # Extract node and edge data for integrate visualization
    nodes_js = []
    edges_js = []
    
    for node in graph_data.get("nodes", []):
        nodes_js.append({
            "id": node.get("id"),
            "label": node.get("type"),
            "properties": node.get("properties", {}),
            "title": json.dumps(node.get("properties", {}), indent=2)
        })
    
    for edge in graph_data.get("edges", []):
        edges_js.append({
            "from": edge.get("source"),
            "to": edge.get("target"),
            "label": edge.get("type"),
            "properties": edge.get("properties", {}),
            "title": json.dumps(edge.get("properties", {}), indent=2)
        })
    
    # Serialize the network data for passing to analyze.py
    network_data_json = json.dumps({
        "nodes": nodes_js,
        "edges": edges_js,
        "project_name": project_name_str
    })

    # Write network data to a temporary file to avoid command line length issues
    network_data_file = output_dir_for_case / f"{project_name_str}_network_data.json"
    with open(network_data_file, "w", encoding="utf-8") as f:
        f.write(network_data_json)

    # Pass the path to the network data file as an argument
    result = subprocess.run([
        sys.executable, '-m', 'core.analyze', 
        str(graph_json_path), 
        '--html',
        '--network-data', str(network_data_file)
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    # Find the analysis summary JSON
    # The project_name_str already contains the stem of the graph file, 
    # e.g., "project_case_timestamp" if the graph was "project_case_timestamp_graph.json"
    # core/analyze.py names its summary: {graph_stem_without_graph_suffix}_analysis_summary_{timestamp}.json
    # So, project_name_str might be something like "revolutions_american_revolution_20230101_120000"
    # and the summary file will be "revolutions_american_revolution_20230101_120000_analysis_summary_sometimestamp.json"
    glob_pattern = f"{project_name_str.replace('_graph', '')}*_analysis_summary_*.json"
    safe_print(f"[DEBUG] Globbing for summary JSON in {output_dir_for_case} with pattern: {glob_pattern}")
    summary_json_files = sorted(output_dir_for_case.glob(glob_pattern), reverse=True)
    
    if summary_json_files:
        safe_print(f"[DEBUG] Found summary JSON files: {summary_json_files}")
        summary_path = str(summary_json_files[0])
        
        # Bayesian Enhancement (if enabled)
        if bayesian_config and BAYESIAN_AVAILABLE:
            summary_path = enhance_with_bayesian_analysis(
                summary_path, case_file_path_str, output_dir_for_case, project_name_str, bayesian_config
            )
        elif bayesian_config and not BAYESIAN_AVAILABLE:
            print("[WARNING] Bayesian analysis requested but components not available.")
            print("Install required packages: pip install scipy numpy")
        
        return summary_path
    else:
        print(f"[ERROR] Could not find analysis summary JSON in {output_dir_for_case} using pattern {glob_pattern}")
        return None

def create_bayesian_config_from_args(args):
    """Create BayesianReportConfig from command line arguments."""
    if not args.bayesian:
        return None
    
    if not BAYESIAN_AVAILABLE:
        print("[ERROR] Bayesian analysis requested but components not available.")
        print("Please ensure all required packages are installed:")
        print("  pip install scipy numpy")
        print("And that the Bayesian components are properly integrated.")
        return None
    
    config = BayesianReportConfig(
        include_uncertainty_analysis=not args.no_uncertainty,
        include_sensitivity_analysis=not args.no_uncertainty,
        include_visualizations=not args.no_visualizations,
        uncertainty_simulations=args.simulations,
        confidence_level=args.confidence_level
    )
    
    print(f"[INFO] Bayesian analysis enabled:")
    print(f"  Simulations: {args.simulations}")
    print(f"  Confidence level: {args.confidence_level:.1%}")
    print(f"  Uncertainty analysis: {'enabled' if not args.no_uncertainty else 'disabled'}")
    print(f"  Visualizations: {'enabled' if not args.no_visualizations else 'disabled'}")
    
    return config

def enhance_with_bayesian_analysis(summary_path, case_file_path, output_dir, project_name, bayesian_config):
    """Enhance traditional analysis with Bayesian confidence assessment."""
    try:
        print(f"[INFO] Starting Bayesian enhancement of analysis...")
        
        # Load traditional analysis results
        import json
        with open(summary_path, 'r', encoding='utf-8') as f:
            analysis_results = json.load(f)
        
        # Load original text for evidence assessment
        with open(case_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Extract evidence for Van Evera assessment
        print("[INFO] Performing Van Evera evidence assessment...")
        evidence_assessments = []
        
        # Try to extract evidence from analysis results or create mock evidence for basic integration
        evidence_nodes = [node for node in analysis_results.get('nodes', []) 
                         if node.get('type', '').lower() == 'evidence']
        
        if evidence_nodes:
            for i, evidence_node in enumerate(evidence_nodes[:3]):  # Limit to 3 for performance
                try:
                    assessment = refine_evidence_assessment_with_llm(
                        evidence_node.get('description', 'Evidence description'),
                        text_content,
                        context_info=f"Analysis context: {analysis_results.get('narrative_summary', '')}"
                    )
                    evidence_assessments.append(assessment)
                except Exception as e:
                    print(f"[WARNING] Failed to assess evidence {i+1}: {e}")
                    continue
        
        if not evidence_assessments:
            print("[INFO] No evidence assessments available - using structure-only Bayesian analysis")
        
        # Integrate Bayesian analysis
        print("[INFO] Integrating Bayesian confidence assessment...")
        enhanced_analysis = integrate_bayesian_analysis(
            analysis_results,
            evidence_assessments,
            output_dir=output_dir,
            config=bayesian_config
        )
        
        # Save enhanced analysis
        enhanced_filename = f"{project_name}_bayesian_enhanced_analysis_{timestamp()}.json"
        enhanced_path = Path(output_dir) / enhanced_filename
        
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"[SUCCESS] Bayesian enhancement completed: {enhanced_path.name}")
        
        # Extract and display key metrics
        bayesian_section = enhanced_analysis.get('bayesian_analysis', {})
        if bayesian_section:
            confidence_assessments = bayesian_section.get('confidence_assessments', {})
            print("[INFO] Bayesian Analysis Summary:")
            for hyp_id, assessment in confidence_assessments.items():
                confidence = assessment.get('overall_confidence', 0)
                level = assessment.get('confidence_level', 'unknown')
                print(f"  {hyp_id.replace('_', ' ').title()}: {confidence:.1%} confidence ({level})")
        
        return str(enhanced_path)
        
    except Exception as e:
        print(f"[ERROR] Bayesian enhancement failed: {e}")
        print("[INFO] Continuing with traditional analysis results")
        return summary_path

# --- Universal LLM Interface (replaces 108-line query_gemini) ---
from universal_llm_kit.universal_llm import structured, chat

def query_llm(text_content, schema=None, system_instruction_text="", use_structured_output=True):
    """Fixed structured output parsing with proper Pydantic model creation"""
    import google.generativeai as genai
    import os
    import json
    from core.extract import parse_json
    
    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"{system_instruction_text}\n\n{text_content}" if system_instruction_text else text_content
    
    if use_structured_output and schema:
        if hasattr(schema, 'model_json_schema'):
            prompt += f"\n\nRespond with valid JSON matching this schema: {schema.model_json_schema()}"
        else:
            prompt += f"\n\nRespond with valid JSON matching this schema: {json.dumps(schema, indent=2)}"
    
    response = model.generate_content(prompt)
    raw_result = response.text
    print(f"[DEBUG] Gemini response length: {len(raw_result)} chars")
    print(f"[DEBUG] Gemini response start: {raw_result[:200]}...")
    
    # If structured output requested, parse and create Pydantic model
    if use_structured_output and schema and hasattr(schema, 'model_validate'):
        try:
            # Use existing JSON cleaning from core.extract
            cleaned_json = parse_json(raw_result)
            print(f"[DEBUG] Parsed JSON keys: {list(cleaned_json.keys()) if isinstance(cleaned_json, dict) else 'Not a dict'}")
            
            # Create Pydantic model instance
            structured_response = schema.model_validate(cleaned_json)
            print(f"[DEBUG] Created structured response: {type(structured_response).__name__}")
            return structured_response
            
        except Exception as e:
            print(f"[WARNING] Failed to create structured response: {e}")
            print(f"[WARNING] Falling back to raw response")
            # Fall back to raw response for backward compatibility
            return raw_result
    
    return raw_result

# --- Save Output ---
def save_output(data, out_dir, project, suffix):
    out_path = out_dir / f"{project}_{timestamp()}_{suffix}"
    with open(out_path, "w", encoding="utf-8") as f:
        if isinstance(data, (dict, list)):
            json.dump(data, f, indent=2)
        else:
            f.write(data)
    return out_path

# --- Visualization (vis.js HTML) ---
def visualize_graph(graph_data, html_path, project):
    # Minimal vis.js HTML generator
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    node_js = ",\n        ".join([
        f"{{id: '{n.get('id')}', label: '{n.get('type')}', title: `{json.dumps(n.get('properties', {}), indent=2)}`}}" for n in nodes
    ])
    edge_js = ",\n        ".join([
        f"{{from: '{e.get('source')}', to: '{e.get('target')}', label: '{e.get('type')}', title: `{json.dumps(e.get('properties', {}), indent=2)}`}}" for e in edges
    ])
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Process Tracing Network</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style> #mynetwork {{ width: 100vw; height: 90vh; border: 1px solid lightgray; }} </style>
</head>
<body>
<h2>Process Tracing Network ({project})</h2>
<div id="mynetwork"></div>
<script type="text/javascript">
  var nodes = new vis.DataSet([
    {node_js}
  ]);
  var edges = new vis.DataSet([
    {edge_js}
  ]);
  var container = document.getElementById('mynetwork');
  var data = {{ nodes: nodes, edges: edges }};
  var options = {{
    nodes: {{ shape: 'dot', size: 20, font: {{ size: 16 }} }},
    edges: {{ arrows: 'to', font: {{ align: 'middle' }} }},
    physics: {{ stabilization: true }}
  }};
  var network = new vis.Network(container, data, options);
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

# --- Main Pipeline ---
def main():
    parser = argparse.ArgumentParser(
        description="Advanced Process Tracing Pipeline (Gemini JSON Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Traditional analysis
  python process_trace_advanced.py --project myproject
  
  # Enhanced analysis with Bayesian confidence assessment
  python process_trace_advanced.py --project myproject --bayesian
  
  # High-quality Bayesian analysis with custom settings
  python process_trace_advanced.py --project myproject --bayesian --simulations 2000 --confidence-level 0.99
  
  # Fast Bayesian analysis (skip uncertainty analysis)
  python process_trace_advanced.py --project myproject --bayesian --no-uncertainty --simulations 100
        """
    )
    parser.add_argument("-p", "--project", type=str, help="Project name (subdirectory of input_text/)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Directory for outputs (default: output_data/<project>)")
    parser.add_argument("--extract-only", action="store_true", help="Run extraction and save initial graph + editable files, then exit.")
    parser.add_argument("--export-editable", action="store_true", help="Export editable_nodes.json and editable_edges.json from a graph JSON file.")
    parser.add_argument("--import-edited", action="store_true", help="Import edited node/edge files and create a corrected graph JSON.")
    parser.add_argument("--input-nodes", type=str, help="Path to editable_nodes.json for import.")
    parser.add_argument("--input-edges", type=str, help="Path to editable_edges.json for import.")
    parser.add_argument("--graph-file", type=str, help="Path to a graph JSON file to analyze or export.")
    parser.add_argument("--analyze-only", action="store_true", help="Run analysis only on a specified graph file.")
    parser.add_argument("--comparative", action="store_true", help="Run comparative analysis on multiple cases.")
    parser.add_argument("--case-directory", type=str, help="Directory containing case JSON files for comparative analysis.")
    parser.add_argument("--case-files", nargs='+', help="Specific case files for comparative analysis.")
    parser.add_argument("--comparison-types", nargs='+', choices=['mss', 'mds', 'diverse'], 
                       default=['mss', 'mds'], help="Types of comparisons to perform.")
    
    # Bayesian Enhancement Options
    parser.add_argument("--bayesian", action="store_true", 
                       help="Enable Bayesian confidence assessment and uncertainty analysis")
    parser.add_argument("--simulations", type=int, default=1000,
                       help="Number of Monte Carlo simulations for uncertainty analysis (default: 1000)")
    parser.add_argument("--confidence-level", type=float, default=0.95,
                       help="Confidence level for uncertainty intervals (default: 0.95)")
    parser.add_argument("--no-uncertainty", action="store_true",
                       help="Skip uncertainty analysis (faster execution)")
    parser.add_argument("--no-visualizations", action="store_true", 
                       help="Disable visualization generation in Bayesian reports")
    
    args = parser.parse_args()

    # --- Export editable nodes/edges from a graph JSON ---
    if args.export_editable:
        if not args.graph_file or not args.output:
            print("[ERROR] --export-editable requires --graph-file and --output.")
            sys.exit(1)
        export_graph_for_editing(args.graph_file, args.output)
        print(f"[SUCCESS] Exported editable_nodes.json and editable_edges.json to {args.output}")
        sys.exit(0)

    # --- Import edited nodes/edges to create a corrected graph JSON ---
    if args.import_edited:
        if not args.input_nodes or not args.input_edges or not args.output:
            print("[ERROR] --import-edited requires --input-nodes, --input-edges, and --output.")
            sys.exit(1)
        import_edited_graph(args.input_nodes, args.input_edges, args.output)
        print(f"[SUCCESS] Imported and created corrected graph JSON at {args.output}")
        sys.exit(0)

    # --- Analyze only mode ---
    if args.analyze_only:
        if not args.graph_file:
            print("[ERROR] --analyze-only requires --graph-file.")
            sys.exit(1)
        print(f"[INFO] Running analysis on {args.graph_file} ...")
        result = subprocess.run([
            sys.executable, '-m', 'core.analyze', args.graph_file, '--html'
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print("ERROR: Analysis failed.")
            sys.exit(1)
        sys.exit(0)

    # --- Comparative analysis mode ---
    if args.comparative:
        from process_trace_comparative import ComparativeProcessTracer
        from core.comparative_models import ComparisonType
        
        # Map comparison type strings to enums
        comparison_type_map = {
            'mss': ComparisonType.MOST_SIMILAR_SYSTEMS,
            'mds': ComparisonType.MOST_DIFFERENT_SYSTEMS,
            'diverse': ComparisonType.DIVERSE_CASE
        }
        comparison_types = [comparison_type_map[ct] for ct in args.comparison_types]
        
        # Set up output directory
        out_dir = Path(args.output) if args.output else Path('comparative_output')
        
        print(f"[INFO] Starting comparative analysis...")
        
        try:
            # Initialize comparative tracer
            tracer = ComparativeProcessTracer(
                case_directory=args.case_directory,
                output_directory=str(out_dir)
            )
            
            # Load cases
            case_ids = tracer.load_cases(case_files=args.case_files)
            print(f"[INFO] Loaded {len(case_ids)} cases: {case_ids}")
            
            if len(case_ids) < 2:
                print("[ERROR] Need at least 2 cases for comparative analysis.")
                sys.exit(1)
            
            # Conduct analysis
            results = tracer.conduct_comparative_analysis(comparison_types=comparison_types)
            print(f"[INFO] Analysis completed. Detected {len(results.mechanism_patterns)} mechanisms.")
            
            # Generate reports
            html_report = tracer.generate_comparative_report(output_format='html')
            json_report = tracer.generate_comparative_report(output_format='json')
            md_report = tracer.generate_comparative_report(output_format='md')
            
            print(f"[SUCCESS] Comparative analysis completed!")
            print(f"  HTML Report: {html_report}")
            print(f"  JSON Report: {json_report}")
            print(f"  Markdown Report: {md_report}")
            
        except Exception as e:
            print(f"[ERROR] Comparative analysis failed: {e}")
            sys.exit(1)
        
        sys.exit(0)

    # --- Extraction only mode (save editable files and exit) ---
    if args.extract_only:
        input_text_dir = Path('input_text')
        projects = list_projects(input_text_dir)
        if not projects:
            print("ERROR: No projects found in input_text/.")
            sys.exit(1)
        project = args.project
        if not project or project not in projects:
            if len(projects) == 1:
                project = projects[0]
                print(f"[INFO] Only one project found: {project}")
            else:
                project = prompt_for_project(projects)
        project_dir = input_text_dir / project
        input_path = find_first_txt(project_dir)
        print(f"\nUsing input: {input_path}\n")
        out_dir = Path(args.output) if args.output else Path('output_data') / project
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Reading input text from {input_path} ...")
        # Create Bayesian configuration if requested
        bayesian_config = create_bayesian_config_from_args(args)
        # Use the new function for single-case processing
        execute_single_case_processing(str(input_path), str(out_dir), project, bayesian_config=bayesian_config)
        sys.exit(0)

    # --- Default: Full pipeline (extract, visualize, analyze) ---
    input_text_dir = Path('input_text')
    projects = list_projects(input_text_dir)
    if not projects:
        print("ERROR: No projects found in input_text/.")
        sys.exit(1)

    project = args.project
    if not project or not project in projects:
        if len(projects) == 1:
            project = projects[0]
            print(f"[INFO] Only one project found: {project}")
        else:
            project = prompt_for_project(projects)

    project_dir = input_text_dir / project
    input_path = find_first_txt(project_dir)
    safe_print(f"\nFile: Using input: {input_path}\n")
    out_dir = Path(args.output) if args.output else Path('output_data') / project
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading input text from {input_path} ...")
    # Create Bayesian configuration if requested
    bayesian_config = create_bayesian_config_from_args(args)
    # Use the new function for single-case processing
    execute_single_case_processing(str(input_path), str(out_dir), project, bayesian_config=bayesian_config)

if __name__ == "__main__":
    main() 
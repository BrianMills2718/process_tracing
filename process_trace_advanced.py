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

# Add a safe print function to handle UnicodeEncodeError
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Try printing with errors replaced
        try:
            print(*(str(a).encode('utf-8', errors='replace').decode('utf-8') for a in args), **kwargs)
        except Exception:
            pass  # Silently ignore if still fails

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

# --- LLM Prompt Template ---
COMPREHENSIVE_LLM_PROMPT_TEMPLATE = """
You are an expert qualitative researcher and analyst specializing in process tracing methodology.
Your task is to meticulously analyze the provided text and extract entities (nodes) and relationships (edges) to construct a DETAILED, DEEPLY INTERCONNECTED, and CAUSALLY RICH network graph.
This graph MUST strictly adhere to the provided JSON schema, which is derived from the Process Tracing Ontology. Your primary goal is to create a graph that explicitly details causal processes, enabling robust analysis. Failure to connect nodes or provide detailed properties where the text allows will significantly diminish the utility of the output.

{global_hypothesis_section}

**Overall Goal (Revised and Emphasized):**
Your primary objective is to deconstruct the provided text into a granular causal narrative. This means you MUST:
1.  Identify and extract all relevant `Event`, `Actor`, `Causal_Mechanism`, `Hypothesis`, `Evidence`, `Condition`, and `Alternative_Explanation` nodes.
2.  **Establish clear CAUSAL CHAINS:** Extract `Event` nodes and meticulously link them sequentially using `causes` (or `triggers`, `leads_to`) edges. Clearly identify `triggering` and `outcome` event subtypes.
3.  **Deconstruct CAUSAL MECHANISMS:** For each `Causal_Mechanism`, extract its constituent `Event` nodes representing its internal steps/stages, linking them with `part_of_mechanism` edges. Mechanisms like 'colonial resistance' or 'political mobilization' MUST be broken down into multiple specific action/event steps.
4.  **Rigorously Test HYPOTHESES:** For EVERY `Hypothesis` node (especially any globally provided one), actively search the entire text for ALL relevant `Evidence`.
5.  **Classify EVIDENCE Diagnostically:** For each `Evidence` node, assign a `type` from the Van Evera categories (`hoop`, `smoking_gun`, `straw_in_the_wind`, `doubly_decisive`) in its properties. Use `general` only as a last resort.
6.  **Uncover ALTERNATIVE EXPLANATIONS:** Identify and extract any alternative explanations for key outcomes.
7.  **Detail CONDITIONS:** Extract `Condition` nodes and explicitly link them to the `Event` or `Causal_Mechanism` nodes they enable or constrain.

---
**CRITICAL INSTRUCTIONS FOR GRAPH CONSTRUCTION (NEW SECTION):**

* **Connectivity is Paramount:** Do NOT leave nodes isolated if the text implies any relationship. Maximize relevant connections according to the ontology.
* **Sequential Flow for Events:** Pay extremely close attention to temporal and causal sequences. If event A leads to event B, ensure a `causes` (or similar) edge exists.
* **Detail over Brevity:** Provide detailed descriptions in node properties. For `source_text_quote` in edge properties, use concise but directly supportive quotes.
* **Ontology Adherence:** All node and edge `type` fields, and all property names, MUST match the definitions in the provided JSON schema derived from `ontology.py`.
---

**Instructions:**

1. **Node Extraction:**
   - Extract all relevant entities as nodes, using the following types: Event, Actor, Causal_Mechanism, Hypothesis, Evidence, Condition, Alternative_Explanation, Data_Source.

2. **Event Nodes:**
   - For each Event, include a properties dictionary with at least:
     - description (string)
     - type (string: triggering, intermediate, outcome, unspecified)
     - date (string, optional)
     - location (string, optional)
   - Ensure `type` properties are assigned thoughtfully: `triggering` for initial events in key sequences, `outcome` for significant results, and `intermediate` for events within a process. An Event without a `causes` edge leading from it or to it (unless it's a clear start/end) might indicate a missed connection.

3. **Causal_Mechanism (CM) Nodes:**
   - For each CM, include a properties dictionary with:
     - description (string)
     - confidence (float, 0.0-1.0)
     - level_of_detail (string: low, medium, high)
   - **Internal Steps (CRITICAL FOR MECHANISM COMPLETENESS):** For each CM, you MUST identify multiple distinct Event nodes from the text that represent its internal, sequential steps, stages, or constituent parts. Link each such Event to the CM using a `part_of_mechanism` edge (i.e., `(Event_Step) -[part_of_mechanism]-> (CM)`). Do not simply list factors; extract them as Event nodes and link them. The more distinct, linked Event parts you extract for a CM, the higher its 'completeness' will be evaluated. Conceptual mechanisms (e.g., 'escalation', 'mobilization', 'resistance') described in the text MUST be broken down this way.
   - Also identify the main Event(s) or Condition(s) that initiate or trigger the CM, linking them to the CM with a `causes` edge.
   - Specify `confidence` and `level_of_detail` for the CM based on the textual description.

4. **Hypothesis (HY) Nodes:**
   - A Hypothesis node MUST represent a specific, testable statement about a causal relationship or the operation, components, or significance of a Causal_Mechanism. Avoid creating Hypothesis nodes for mere assertions or beliefs unless they are framed as part of a causal claim.
   - You MUST actively seek to link Evidence (see next section) to EVERY Hypothesis node you create, especially the global study hypothesis if provided.
   - If a global study hypothesis (ID: '{global_hypothesis_id_for_prompt}', Text: '{global_hypothesis_text_for_prompt}') is provided, you MUST create a Hypothesis node with this exact ID and description. Then, diligently search the entire text for all Evidence that supports or refutes it.
   - If a Hypothesis makes a claim about a specific Causal_Mechanism you've extracted, you MUST link it using an `explains_mechanism` edge: `(Hypothesis) -[explains_mechanism]-> (CM)`. Specify `type_of_claim` in edge properties.

5. **Evidence (EV) Nodes:**
   - For each Evidence node, include a properties dictionary with:
     - description (string)
     - evidence_type (string: 'hoop', 'smoking_gun', 'straw_in_the_wind', 'doubly_decisive', or 'general')
     - certainty (string, optional)
   - For each Evidence node, its properties dictionary MUST include an evidence_type field. The value for evidence_type MUST be one of the allowed diagnostic values from the ontology: hoop, smoking_gun, straw_in_the_wind, doubly_decisive. Do NOT default to general unless no other classification is remotely appropriate. Think critically:
     - Is this Evidence NECESSARY for the Hypothesis to be viable (potential hoop test)? If absent, would the Hypothesis be disconfirmed?
     - Is this Evidence SUFFICIENT to confirm the Hypothesis (potential smoking_gun)? If present, does it strongly point to the Hypothesis being true, being very unlikely if the Hypothesis were false?
     - Assign the most diagnostic type possible based on your interpretation of the text's claim about the evidence. If the text explicitly discusses the diagnostic power of a piece of evidence, reflect that in your choice.
   - Edges linking Evidence to Hypothesis nodes (using supports or refutes edge types) MUST include probative_value (float, your best estimate of its inferential strength, e.g., 0.1-1.0 where 1.0 is very strong) and a concise source_text_quote (string) in their properties dictionary.

6. **Condition Nodes (CRITICAL FOR CONTEXT AND SCOPE):**
   - Extract all relevant `Condition` nodes that describe background factors, context, or scope conditions. [Refer to Condition node properties in ontology.py]
   - For each `Condition`, its `properties` MUST include `type` (e.g., `background`, `enabling`, `facilitating`, `constraining`, `unspecified`).
   - **CRITICAL LINKAGE:** You MUST link `Condition` nodes to the `Event` or `Causal_Mechanism` nodes they affect using `enables` or `constrains` edges. A `Condition` node should not be left isolated if its influence on a process is described.

7. **Alternative_Explanation Nodes (CRITICAL FOR RIGOR):**
   - Actively search the text for any alternative explanations proposed for the main outcomes or causal processes.
   - If found, extract these as `Alternative_Explanation` nodes. [Refer to Alternative_Explanation node properties in ontology.py]
   - Link any `Evidence` found in the text that supports or refutes these alternatives using `supports_alternative` or `refutes_alternative` edges, respectively. Include `probative_value` in the edge properties. [Refer to these edge types in ontology.py]

8. **General Edge Creation:**
   - It is critical that you create as many relevant edges as the text supports, following the specified types and property structures. An interconnected graph is vital. Do not leave nodes isolated if the text implies a relationship.
   - Beliefs, arguments, or normative claims found in the text should be represented as properties of Actor nodes (e.g., beliefs), Condition nodes (e.g., prevailing ideology), or Event nodes (e.g., articulation of an argument), NOT as Hypothesis nodes unless they are part of a testable causal claim as defined above.

---

**Output:**
- Return a JSON object with two lists: 'nodes' and 'edges', strictly following the provided schema.
- Each node must have a unique 'id', 'type', and a 'properties' dictionary.
- Each edge must have a unique 'id' (optional, can be auto-generated if not provided by LLM), 'type', 'source' (node_id), 'target' (node_id), and a 'properties' dictionary.
"""

# --- Refactored Single-Case Processing Function ---
def execute_single_case_processing(case_file_path_str, output_dir_for_case_str, project_name_str, global_hypothesis_text=None, global_hypothesis_id=None):
    import subprocess
    from datetime import datetime
    case_file_path = Path(case_file_path_str)
    output_dir_for_case = Path(output_dir_for_case_str)
    output_dir_for_case.mkdir(parents=True, exist_ok=True)
    # Read input text
    with open(case_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Build schema
    schema = get_schema()
    # Format prompt
    active_prompt_template = COMPREHENSIVE_LLM_PROMPT_TEMPLATE
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
    # Query Gemini
    raw_json = query_gemini(text, schema, final_system_prompt)
    # Save raw output
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_json_path = output_dir_for_case / f"{project_name_str}_{now_str}_graph.json"
    
    # Parse the JSON response
    try:
        graph_data = json.loads(raw_json)
        with open(graph_json_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        with open(graph_json_path, "w", encoding="utf-8") as f:
            f.write(raw_json)
        return None
            
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
        return str(summary_json_files[0])
    else:
        print(f"[ERROR] Could not find analysis summary JSON in {output_dir_for_case} using pattern {glob_pattern}")
        return None

# --- Update query_gemini to accept system_instruction ---
def query_gemini(text_content, schema, system_instruction_text):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY environment variable not set and not found in .env file.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    MODEL_ID = "gemini-1.5-flash"  # Use stable model version
    # Save and print all inputs for debugging
    with open("debug_input_text.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    with open("debug_prompt.txt", "w", encoding="utf-8") as f:
        f.write(system_instruction_text)
    with open("debug_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print("\n[DEBUG] Gemini Extraction Debugging:")
    print(f"[DEBUG] Input text length: {len(text_content)} characters (saved to debug_input_text.txt)")
    print(f"[DEBUG] Prompt length: {len(system_instruction_text)} characters (saved to debug_prompt.txt)")
    print(f"[DEBUG] Schema saved to debug_schema.json")
    # LLM call with corrected API
    model = genai.GenerativeModel(MODEL_ID)
    
    # Build generation config
    generation_config = {}
    if schema:
        generation_config['response_mime_type'] = 'application/json'
        generation_config['response_schema'] = schema
    
    # Create content with system instruction
    if system_instruction_text:
        prompt_content = f"System: {system_instruction_text}\n\nUser: {text_content}"
    else:
        prompt_content = text_content
    
    result = model.generate_content(
        prompt_content,
        generation_config=generation_config if generation_config else None
    )
    print(f"[DEBUG] Gemini result type: {type(result)}")
    print(f"[DEBUG] Gemini result repr: {repr(result)}")
    # Print and save the full raw output
    print(f"[DEBUG] Raw Gemini output (first 500 chars):\n{getattr(result, 'text', str(result))[:500]}\n---END GEMINI OUTPUT SAMPLE---\n")
    with open("debug_raw_gemini_output.txt", "w", encoding="utf-8") as f:
        f.write(getattr(result, 'text', str(result)))
    print("[DEBUG] Full raw Gemini output saved to debug_raw_gemini_output.txt\n")
    # Extract and parse JSON from markdown-wrapped response
    raw_text = getattr(result, 'text', str(result))
    try:
        # Handle markdown-wrapped JSON
        if '```json' in raw_text:
            json_start = raw_text.find('```json') + 7
            json_end = raw_text.find('```', json_start)
            json_text = raw_text[json_start:json_end].strip()
        else:
            json_text = raw_text.strip()
        
        parsed = json.loads(json_text)
        print(f"[SUCCESS] Parsed JSON successfully")
        return json_text
    except Exception as e:
        print(f"[ERROR] Could not parse Gemini output as JSON: {e}")
        with open("debug_gemini_json_error.txt", "w", encoding="utf-8") as f:
            f.write(f"Error: {e}\n\nRaw output:\n{raw_text}")
        return raw_text

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
    parser = argparse.ArgumentParser(description="Advanced Process Tracing Pipeline (Gemini JSON Mode)")
    parser.add_argument("-p", "--project", type=str, help="Project name (subdirectory of input_text/)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Directory for outputs (default: output_data/<project>)")
    parser.add_argument("--extract-only", action="store_true", help="Run extraction and save initial graph + editable files, then exit.")
    parser.add_argument("--export-editable", action="store_true", help="Export editable_nodes.json and editable_edges.json from a graph JSON file.")
    parser.add_argument("--import-edited", action="store_true", help="Import edited node/edge files and create a corrected graph JSON.")
    parser.add_argument("--input-nodes", type=str, help="Path to editable_nodes.json for import.")
    parser.add_argument("--input-edges", type=str, help="Path to editable_edges.json for import.")
    parser.add_argument("--graph-file", type=str, help="Path to a graph JSON file to analyze or export.")
    parser.add_argument("--analyze-only", action="store_true", help="Run analysis only on a specified graph file.")
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
        # Use the new function for single-case processing
        execute_single_case_processing(str(input_path), str(out_dir), project)
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
    print(f"\n📄 Using input: {input_path}\n")
    out_dir = Path(args.output) if args.output else Path('output_data') / project
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading input text from {input_path} ...")
    # Use the new function for single-case processing
    execute_single_case_processing(str(input_path), str(out_dir), project)

if __name__ == "__main__":
    main() 
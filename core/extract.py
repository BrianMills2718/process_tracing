"""
ADVANCED PROCESS‑TRACING GRAPH – VERSION 3.0
Robust Gemini streaming → JSON → Visualization with comprehensive ontology

Expected JSON structure for LLM output (based on advanced ontology):

{
  "nodes": [
    {"id": "evt1", "type": "Event", "properties": {"description": "Tea Act passed", "timestamp": "1773-05-10", "certainty": 0.9}},
    {"id": "hyp1", "type": "Hypothesis", "properties": {"description": "Taxation without representation causes unrest", "prior_probability": 0.6, "status": "active"}},
    {"id": "evd1", "type": "Evidence", "properties": {"description": "Public protests against Tea Act", "type": "straw_in_the_wind", "source": "doc1"}},
    {"id": "ds1", "type": "Data_Source", "properties": {"type": "document", "credibility": 0.8}}
  ],
  "edges": [
    {"source_id": "evd1", "target_id": "hyp1", "type": "tests_hypothesis", "properties": {"probative_value": 0.3, "test_result": "passed"}},
    {"source_id": "ds1", "target_id": "evd1", "type": "provides_evidence", "properties": {}}
  ]
}

"""

import os, glob, re, json, ast, sys, textwrap, time
from itertools import count
from datetime import datetime
import webbrowser
from core.ontology import NODE_TYPES, EDGE_TYPES, NODE_COLORS

# Check if Google Generative AI package is installed
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("❌ Google Generative AI package not found.")
    print("   Please install it with: pip install google-generativeai")
    print("   Or activate the environment where it's installed.")
    print("\n🔍 You can still use this script to visualize existing JSON data.")

# ────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDXaLhSWAQhGNHZqdbvY-qFB0jxyPbiiow"  # Only used if HAS_GEMINI is True
INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input_text")
# Default: use the first .txt file in a project subdirectory if not specified
INPUT_FILE_PATH = None
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith('.txt'):
            INPUT_FILE_PATH = os.path.join(root, file)
            break
    if INPUT_FILE_PATH:
        break
if not INPUT_FILE_PATH:
    raise FileNotFoundError("No .txt file found in input_text/ or its subdirectories.")

# Determine project name from input file path
project_name = os.path.relpath(os.path.dirname(INPUT_FILE_PATH), INPUT_DIR)
project_name = project_name.replace(os.sep, "_")  # flatten subdirs to single name
# Timestamp for output files
now_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# Output paths
OUTPUT_DIR = os.path.join("output_data", project_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"{project_name}_{now_str}_graph.json")
OUTPUT_HTML = os.path.join(OUTPUT_DIR, f"{project_name}_{now_str}_graph.html")
MODEL_NAME  = "gemini-2.5-flash-preview-04-17"
MAX_RETRIES = 3  # Maximum number of retries for Gemini API

# ────────────────────────────────────────────────────────────────────
# GEMINI PROMPT WITH EXPANDED ONTOLOGY
# ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are an expert in causal process tracing methodology. Extract a structured causal graph from the provided text using the following ontology subset. Output your answer as a JSON object with two arrays: 'nodes' and 'edges'.

Each node must have:
- id: unique string
- type: one of [Event, Hypothesis, Evidence]
- properties: a dictionary of the node's properties (see below)

Each edge must have:
- source_id: id of the source node
- target_id: id of the target node
- type: one of [causes, tests_hypothesis]
- properties: a dictionary of the edge's properties (see below)

Node Types (with Properties):
1. Event
   - description (string, required)
   - timestamp (string, optional)
   - certainty (float 0-1, optional)
   - type (string: triggering, intermediate, outcome, optional)
2. Hypothesis
   - description (string, required)
   - prior_probability (float 0-1, optional)
   - status (string: active, confirmed, refuted, optional)
3. Evidence
   - description (string, required)
   - type (string: hoop, smoking_gun, straw_in_the_wind, optional)
   - source (string, optional)
   - certainty (float 0-1, optional)

Edge Types (with Properties):
1. causes (Event → Event)
   - certainty (float 0-1, optional)
   - type (string: direct, indirect, optional)
2. tests_hypothesis (Evidence → Hypothesis)
   - probative_value (float 0-1, optional)
   - test_result (string: passed, failed, ambiguous, optional)

Output format example:
{{
  "nodes": [
    {{"id": "evt1", "type": "Event", "properties": {{"description": "Tea Act passed", "timestamp": "1773-05-10", "certainty": 0.9}}}},
    {{"id": "hyp1", "type": "Hypothesis", "properties": {{"description": "Taxation without representation causes unrest", "prior_probability": 0.6, "status": "active"}}}},
    {{"id": "evd1", "type": "Evidence", "properties": {{"description": "Public protests against Tea Act", "type": "straw_in_the_wind", "source": "doc1"}}}}
  ],
  "edges": [
    {{"source_id": "evd1", "target_id": "hyp1", "type": "tests_hypothesis", "properties": {{"probative_value": 0.3, "test_result": "passed"}}}},
    {{"source_id": "evt1", "target_id": "evt2", "type": "causes", "properties": {{"certainty": 0.8, "type": "direct"}}}}
  ]
}}

TEXT TO ANALYZE:
{text}
"""

# ────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────────
def read_all_text(path: str) -> str:
    """Read all text files in the given directory and concatenate them."""
    files = glob.glob(os.path.join(path, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {path}")

    big_text, total = "", 0
    print(f"[INFO] Reading {len(files)} file(s):")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            txt = f.read()
            big_text += txt + "\n\n"
            total += len(txt)
            print(f"  • {os.path.basename(fp):<40s} ({len(txt):,} chars)")
    print(f"[INFO] Total characters loaded: {total:,}\n")
    return big_text


def clean_json_block(raw: str) -> str:
    """Remove ``` fences, language tags, stray commas, BOM, etc."""
    # strip code fences
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "")
    # remove leading / trailing whitespace & BOM
    cleaned = cleaned.strip().lstrip("\ufeff")
    # remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # replace null with empty string for description fields to ensure they exist
    cleaned = re.sub(r'"description":\s*null', '"description": ""', cleaned)
    return cleaned


def parse_json(raw: str) -> dict:
    """Parse JSON from raw string, handling common issues."""
    raw_clean = clean_json_block(raw)
    try:
        return json.loads(raw_clean)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_clean)
        except Exception as e:
            snippet = textwrap.shorten(raw_clean, width=300, placeholder=" […]")
            print("\n[WARN]  JSON parsing failed. First 300 chars Gemini sent:\n"
                  "────────────────────────────────────────────────────")
            print(snippet)
            print("────────────────────────────────────────────────────")
            raise e


def validate_json_against_ontology(data: dict) -> tuple[bool, list[str]]:
    """
    Validate the JSON data against the ontology definition (new structure).
    Returns a tuple of (is_valid, list_of_errors).
    """
    errors = []
    
    # Check basic structure
    if not isinstance(data, dict):
        return False, ["Data is not a dictionary"]
    
    if "nodes" not in data:
        return False, ["Missing 'nodes' array in data"]
    
    if "edges" not in data:
        return False, ["Missing 'edges' array in data"]
    
    # Empty graph is valid
    if len(data["nodes"]) == 0 and len(data["edges"]) == 0:
        return True, []
        
    # Map of node IDs for edge validation
    node_ids = {}
    node_types = {}
    
    # Validate nodes
    for i, node in enumerate(data["nodes"]):
        if not isinstance(node, dict):
            errors.append(f"Node at index {i} is not a dictionary")
            continue
        if "id" not in node:
            errors.append(f"Node at index {i} is missing required 'id' field")
        else:
            node_id = node["id"]
            if node_id in node_ids:
                errors.append(f"Duplicate node ID: {node_id}")
            node_ids[node_id] = i
        if "type" not in node:
            errors.append(f"Node {node.get('id', f'at index {i}')} is missing required 'type' field")
        else:
            node_type = node["type"]
            node_types[node.get('id', f'at index {i}')] = node_type
            if node_type not in NODE_TYPES:
                errors.append(f"Node {node.get('id', f'at index {i}')} has invalid type: {node_type}")
            else:
                # Check required fields in properties
                props = node.get("properties", {})
                for prop, prop_def in NODE_TYPES[node_type]["properties"].items():
                    if prop_def.get("required") and prop not in props:
                        errors.append(f"Node {node.get('id', f'at index {i}')} of type {node_type} is missing required property: {prop}")
                    if prop in props:
                        val = props[prop]
                        # Type check
                        if prop_def["type"] == "float":
                            if not isinstance(val, (float, int)):
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' should be a float")
                            if "min" in prop_def and val < prop_def["min"]:
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' below min {prop_def['min']}")
                            if "max" in prop_def and val > prop_def["max"]:
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' above max {prop_def['max']}")
                        if prop_def["type"] == "string":
                            if not isinstance(val, str):
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' should be a string")
                            if "allowed_values" in prop_def and val not in prop_def["allowed_values"]:
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' has invalid value '{val}'")
    # Validate edges
    for i, edge in enumerate(data["edges"]):
        if not isinstance(edge, dict):
            errors.append(f"Edge at index {i} is not a dictionary")
            continue
        if "source_id" not in edge:
            errors.append(f"Edge at index {i} is missing required 'source_id' field")
        elif edge["source_id"] not in node_ids:
            errors.append(f"Edge at index {i} references non-existent source node: {edge['source_id']}")
        if "target_id" not in edge:
            errors.append(f"Edge at index {i} is missing required 'target_id' field")
        elif edge["target_id"] not in node_ids:
            errors.append(f"Edge at index {i} references non-existent target node: {edge['target_id']}")
        if "type" not in edge:
            errors.append(f"Edge at index {i} is missing required 'type' field")
        else:
            edge_type = edge["type"]
            if edge_type not in EDGE_TYPES:
                errors.append(f"Edge at index {i} has invalid type: {edge_type}")
            elif "source_id" in edge and "target_id" in edge and edge["source_id"] in node_ids and edge["target_id"] in node_ids:
                source_type = node_types.get(edge["source_id"])
                target_type = node_types.get(edge["target_id"])
                if source_type and target_type:
                    if source_type not in EDGE_TYPES[edge_type]["domain"]:
                        errors.append(f"Edge {edge_type} at index {i} has incompatible source node type: {source_type}")
                    if target_type not in EDGE_TYPES[edge_type]["range"]:
                        errors.append(f"Edge {edge_type} at index {i} has incompatible target node type: {target_type}")
            # Check edge properties
            props = edge.get("properties", {})
            for prop, prop_def in EDGE_TYPES[edge_type]["properties"].items():
                if prop_def.get("required") and prop not in props:
                    errors.append(f"Edge {edge_type} at index {i} is missing required property: {prop}")
                if prop in props:
                    val = props[prop]
                    if prop_def["type"] == "float":
                        if not isinstance(val, (float, int)):
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' should be a float")
                        if "min" in prop_def and val < prop_def["min"]:
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' below min {prop_def['min']}")
                        if "max" in prop_def and val > prop_def["max"]:
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' above max {prop_def['max']}")
                    if prop_def["type"] == "string":
                        if not isinstance(val, str):
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' should be a string")
                        if "allowed_values" in prop_def and val not in prop_def["allowed_values"]:
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' has invalid value '{val}'")
    return len(errors) == 0, errors


def query_gemini(big_text: str, retry_count=0) -> dict:
    """Query Gemini API with retry mechanism for validation failures."""
    print("[DEBUG] About to query Gemini...", flush=True)
    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [types.Content(role="user",
                              parts=[types.Part.from_text(
                                  text=PROMPT_TEMPLATE.format(text=big_text))])]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    print(f"[RUN] Streaming from Gemini (attempt {retry_count + 1}/{MAX_RETRIES})…")
    json_str = ""
    chunk_count = 0
    for chunk in client.models.generate_content_stream(
            model=MODEL_NAME, contents=contents, config=config):
        if hasattr(chunk, "text"):
            json_str += chunk.text
        chunk_count += 1
        print(f"  chunk {chunk_count}")

    print(f"[OK] Stream complete. Received {chunk_count} chunks. Parsing JSON…")
    
    try:
        # Parse the returned JSON
        graph_data = parse_json(json_str)
        
        # Validate against ontology
        is_valid, errors = validate_json_against_ontology(graph_data)
        
        if is_valid:
            print("[OK] Validation successful: JSON conforms to ontology!")
            return graph_data
        else:
            print("[WARN] Validation failed. Errors found:")
            for error in errors:
                print(f"  ⚠️ {error}")
                
            # Retry if we haven't exceeded the maximum attempts
            if retry_count < MAX_RETRIES - 1:
                print(f"[RETRY] Retrying query ({retry_count + 2}/{MAX_RETRIES})...")
                time.sleep(2)  # Short delay between retries
                return query_gemini(big_text, retry_count + 1)
            else:
                print("[ERROR] Maximum retry attempts reached. Using last response despite validation errors.")
                return graph_data
                
    except Exception as e:
        print(f"[ERROR] Error processing response: {str(e)}")
        
        # Retry if we haven't exceeded the maximum attempts
        if retry_count < MAX_RETRIES - 1:
            print(f"[RETRY] Retrying query ({retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(2)  # Short delay between retries
            return query_gemini(big_text, retry_count + 1)
        else:
            print("[ERROR] Maximum retry attempts reached. Raising error.")
            raise e


def build_visualization(data, html_out):
    """
    Creates a standalone HTML visualization using vis.js
    for the comprehensive process tracing ontology.
    """
    # Create formatted nodes for vis.js
    vis_nodes = []
    for node in data["nodes"]:
        props = node.get("properties", {})
        # Prepare label based on node type
        label = f"{node['type']}: {props.get('description', node['id'])}"
        if node["type"] == "Actor":
            label = f"{node['type']}: {props.get('name', props.get('description', node['id']))}"
        # Truncate long labels
        if len(label) > 100:
            label = label[:97] + "..."
        # Prepare tooltip with all properties
        tooltip = "<div style='max-width:300px;'>"
        tooltip += f"<strong>ID:</strong> {node['id']}<br>"
        tooltip += f"<strong>Type:</strong> {node['type']}<br>"
        for k, v in props.items():
            tooltip += f"<strong>{k.replace('_', ' ').title()}:</strong> {v}<br>"
        tooltip += "</div>"
        vis_nodes.append({
            "id": node["id"],
            "label": label,
            "group": node["type"],
            "color": NODE_COLORS.get(node["type"], "#dddddd"),
            "title": tooltip
        })
    # Create formatted edges for vis.js
    vis_edges = []
    node_ids = {n["id"] for n in data["nodes"]}
    for edge in data["edges"]:
        if edge["source_id"] not in node_ids:
            print(f"⚠️  Skipping edge from non-existent source node '{edge['source_id']}' to '{edge['target_id']}'.")
            continue
        if edge["target_id"] not in node_ids:
            print(f"⚠️  Skipping edge from '{edge['source_id']}' to non-existent target node '{edge['target_id']}'.")
            continue
        props = edge.get("properties", {})
        tooltip = "<div style='max-width:250px;'>"
        tooltip += f"<strong>Type:</strong> {edge['type']}<br>"
        for k, v in props.items():
            tooltip += f"<strong>{k.replace('_', ' ').title()}:</strong> {v}<br>"
        tooltip += "</div>"
        vis_edges.append({
            "from": edge["source_id"],
            "to": edge["target_id"],
            "label": edge["type"],
            "title": tooltip,
            "width": props.get("certainty", 1) if isinstance(props.get("certainty"), (int, float)) else 1
        })
    # Create groups JSON for the visualization
    groups_json = {}
    for node_type, color in NODE_COLORS.items():
        groups_json[node_type] = {"color": {"background": color, "border": "#444444"}}
    
    # Create legend items HTML
    legend_items = ""
    for node_type, color in NODE_COLORS.items():
        legend_items += f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <div>{node_type.replace('_', ' ')}</div>
        </div>"""
    
    # Generate timestamp for footer
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create HTML template parts
    html_head = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Advanced Process Tracing Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #visualization {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border: 1px solid #ccc;
        }}
        h1 {{
            margin-top: 0;
        }}
        .footer {{
            margin-top: 10px;
            font-size: 0.8em;
            color: #666;
        }}
        .controls {{
            margin-bottom: 10px;
        }}
        button {{
            margin-right: 10px;
            padding: 5px 10px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <h1>Advanced Process Tracing Visualization</h1>
    
    <div class="controls">
        <button onclick="stabilize()">Stabilize Layout</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="fitView()">Fit View</button>
    </div>
    
    <div class="legend">{legend_items}
    </div>
    
    <div id="visualization"></div>
    
    <div class="footer">
        Generated: {timestamp} • 
        Source: {os.path.basename(OUTPUT_JSON)} •
        Nodes: {len(vis_nodes)} • Edges: {len(vis_edges)}
    </div>"""
    
    html_script = f"""
    <script type="text/javascript">
        // Create a network
        var container = document.getElementById('visualization');
        
        // Provide the data in the vis format
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});
        
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        // Options for the network
        var options = {{
            nodes: {{
                shape: 'box',
                margin: 12,
                widthConstraint: {{
                    maximum: 250
                }},
                font: {{
                    size: 14
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                arrows: 'to',
                smooth: {{
                    type: 'dynamic',
                    forceDirection: 'none',
                    roundness: 0.5
                }},
                font: {{
                    size: 12,
                    align: 'middle'
                }},
                color: {{
                    inherit: false,
                    color: '#666666'
                }},
                width: 2,
                shadow: true
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -60,
                    centralGravity: 0.01,
                    springLength: 150,
                    springConstant: 0.05,
                    damping: 0.4
                }},
                stabilization: {{
                    iterations: 300
                }}
            }},
            layout: {{
                improvedLayout: true
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }},
            groups: {json.dumps(groups_json)}
        }};
        
        // Initialize the network
        var network = new vis.Network(container, data, options);
        
        // Define control functions
        function stabilize() {{
            network.stabilize(100);
        }}
        
        var physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
        
        function fitView() {{
            network.fit();
        }}
        
        // After the network is stabilized, disable physics for better user interaction
        network.on("stabilizationIterationsDone", function () {{
            setTimeout(function() {{
                network.setOptions({{ physics: false }});
                physicsEnabled = false;
            }}, 1000); // Small delay to ensure layout is complete
        }});
    </script>
</body>
</html>"""
    
    # Combine the HTML parts
    full_html = html_head + html_script
    
    # Write the HTML file
    with open(html_out, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"[SUCCESS] Graph written to {html_out}")
    
    # Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(html_out))


def read_text_file(file_path: str) -> str:
    """Read a single text file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"[INFO] Reading file: {os.path.basename(file_path)}")
    with open(file_path, encoding="utf-8") as f:
        txt = f.read()
    print(f"[INFO] Total characters loaded: {len(txt):,}\n")
    return txt


# ────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main execution function for the process tracing application."""
    try:
        # If the JSON file already exists, give the user an option to reuse it
        if os.path.exists(OUTPUT_JSON) and os.path.getsize(OUTPUT_JSON) > 0:
            print(f"[INFO] Found existing JSON data file: {OUTPUT_JSON}")
            if not HAS_GEMINI:
                print("[INFO] Will use existing JSON data since Gemini API is not available.")
                use_existing = True
            else:
                response = input("   Would you like to reuse this data instead of querying Gemini? (y/n): ").strip().lower()
                use_existing = response == 'y' or response == 'yes'
        else:
            use_existing = False

        if use_existing:
            # Load existing JSON data
            try:
                with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                print(f"[OK] Successfully loaded data from {OUTPUT_JSON}")
            except Exception as e:
                print(f"[ERROR] Error loading JSON data: {str(e)}")
                if not HAS_GEMINI:
                    print("[ERROR] Cannot proceed without valid JSON data and Gemini API.")
                    return
                use_existing = False
        
        if not use_existing:
            # Process requires Gemini API
            if not HAS_GEMINI:
                print("[ERROR] Cannot query Gemini API. Package not available.")
                return
                
            # Read the specified input file
            text = read_text_file(INPUT_FILE_PATH)
            
            # Query Gemini API with the text
            graph_data = query_gemini(text)

            # Save the graph data to a JSON file
            try:
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=4)
                print(f"[SAVE] Raw graph data saved to {OUTPUT_JSON}")
            except IOError as e:
                print(f"[WARN]  Error saving JSON data to {OUTPUT_JSON}: {e}")

        # Create visualization (works regardless of where data came from)
        build_visualization(graph_data, OUTPUT_HTML)
        
    except Exception as e:
        print(f"[ERROR] Error in main execution: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback:\n{traceback_str}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        import traceback
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")

"""
ADVANCED PROCESS‑TRACING GRAPH – VERSION 3.0
Robust Gemini streaming → JSON → Visualization with comprehensive ontology
"""

import os, glob, re, json, ast, sys, textwrap, time
from itertools import count
from datetime import datetime
import webbrowser

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
INPUT_DIR   = r"C:\Users\Brian\Downloads\process_tracing\input_text"
INPUT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_text", "american_revolution.txt") # Default, can be overridden
OUTPUT_HTML = "advanced_process_trace.html"
OUTPUT_JSON = "advanced_process_trace_data.json"
MODEL_NAME  = "gemini-2.5-flash-preview-04-17"
MAX_RETRIES = 3  # Maximum number of retries for Gemini API

# ────────────────────────────────────────────────────────────────────
# ONTOLOGY DEFINITION FOR VALIDATION
# ────────────────────────────────────────────────────────────────────

# Node types and their required/optional properties
NODE_TYPES = {
    "Event": {
        "required": ["id", "description"],
        "optional": ["timestamp", "location", "certainty", "type"]
    },
    "Causal_Mechanism": {
        "required": ["id", "description"],
        "optional": ["confidence", "status", "level_of_detail"]
    },
    "Hypothesis": {
        "required": ["id", "description"],
        "optional": ["prior_probability", "posterior_probability", "status", "scope"]
    },
    "Evidence": {
        "required": ["id", "description", "type"],
        "optional": ["probative_value", "certainty", "source", "credibility"]
    },
    "Condition": {
        "required": ["id", "description"],
        "optional": ["necessity", "certainty", "type"]
    },
    "Actor": {
        "required": ["id", "name"],
        "optional": ["role", "intentions", "beliefs", "credibility"]
    },
    "Inference_Rule": {
        "required": ["id", "description", "type"],
        "optional": []
    },
    "Inferential_Test": {
        "required": ["id", "description", "type"],
        "optional": ["conditions"]
    },
    "Alternative_Explanation": {
        "required": ["id", "description"],
        "optional": ["probability", "status"]
    },
    "Data_Source": {
        "required": ["id", "type"],
        "optional": ["credibility", "bias_risk"]
    }
}

# Edge types and their properties
EDGE_TYPES = {
    "causes": {
        "source_types": ["Event"],
        "target_types": ["Event"],
        "required": [],
        "optional": ["certainty", "mechanism_id", "type"]
    },
    "part_of_mechanism": {
        "source_types": ["Event"],
        "target_types": ["Causal_Mechanism"],
        "required": [],
        "optional": ["role"]
    },
    "tests_hypothesis": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["inferential_test_id", "probative_value", "test_result"]
    },
    "tests_mechanism": {
        "source_types": ["Evidence"],
        "target_types": ["Causal_Mechanism"],
        "required": [],
        "optional": ["inferential_test_id", "probative_value", "test_result"]
    },
    "supports_alternative": {
        "source_types": ["Evidence"],
        "target_types": ["Alternative_Explanation"],
        "required": [],
        "optional": ["probative_value", "certainty"]
    },
    "refutes_alternative": {
        "source_types": ["Evidence"],
        "target_types": ["Alternative_Explanation"],
        "required": [],
        "optional": ["probative_value", "certainty"]
    },
    "enables": {
        "source_types": ["Condition"],
        "target_types": ["Event", "Causal_Mechanism"],
        "required": [],
        "optional": ["necessity", "certainty", "type"]
    },
    "constrains": {
        "source_types": ["Condition"],
        "target_types": ["Event", "Causal_Mechanism"],
        "required": [],
        "optional": ["certainty", "type"]
    },
    "provides_evidence": {
        "source_types": ["Data_Source"],
        "target_types": ["Evidence"],
        "required": [],
        "optional": ["credibility", "bias_risk", "certainty"]
    },
    "initiates": {
        "source_types": ["Actor"],
        "target_types": ["Event"],
        "required": [],
        "optional": ["certainty", "intention", "agency"]
    },
    "infers": {
        "source_types": ["Inference_Rule"],
        "target_types": ["Hypothesis", "Causal_Mechanism"],
        "required": [],
        "optional": ["certainty", "logic_type"]
    },
    "updates_probability": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["prior_probability", "posterior_probability", "Bayes_factor"]
    },
    "contradicts": {
        "source_types": ["Evidence"],
        "target_types": ["Evidence"],
        "required": [],
        "optional": ["certainty", "reason"]
    },
    "supports": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["certainty", "strength"]
    },
    "refutes": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["certainty", "strength"]
    }
}

# Node type colors for visualization
NODE_COLORS = {
    "Event": "#66b3ff",
    "Causal_Mechanism": "#99ff99",
    "Hypothesis": "#ffcc00",
    "Evidence": "#ff6666",
    "Condition": "#ccccff",
    "Actor": "#ff99cc",
    "Inference_Rule": "#cc99ff",
    "Inferential_Test": "#ffb366",
    "Alternative_Explanation": "#ff9966",
    "Data_Source": "#c2c2f0"
}

# ────────────────────────────────────────────────────────────────────
# GEMINI PROMPT WITH EXPANDED ONTOLOGY
# ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are an expert in causal process tracing methodology. Extract a structured causal graph
from the provided text using this comprehensive ontology:

### Node Types (with Properties)
1. Event (id, description, timestamp, location, certainty, type)
   - Observable occurrences in the causal sequence
   - Type can be: triggering, intermediate, outcome

2. Causal_Mechanism (id, description, confidence, status, level_of_detail)
   - Pathways through which causal influence is transmitted
   - Status can be: verified, contested, hypothetical
   - Level_of_detail can be: abstract, concrete

3. Hypothesis (id, description, prior_probability, posterior_probability, status, scope)
   - Proposed causal explanations being evaluated
   - Status can be: active, confirmed, refuted
   - Scope can be: general, case-specific

4. Evidence (id, description, type, probative_value, certainty, source, credibility)
   - Observations used to evaluate hypotheses/mechanisms
   - Type can be: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, bayesian

5. Condition (id, description, necessity, certainty, type)
   - Factors affecting causal sequences without being events
   - Type can be: background, enabling, facilitating, constraining

6. Actor (id, name, role, intentions, beliefs, credibility)
   - Entities whose decisions/actions shape events or provide evidence
   - Role can be: causal_agent, source_of_evidence, decision_maker

7. Inference_Rule (id, description, type)
   - Formal rules guiding logical inference
   - Type can be: bayesian_updating, abductive, deductive, inductive, heuristic

8. Inferential_Test (id, description, type, conditions)
   - Evaluation frameworks for interpreting evidence
   - Type can be: hoop, smoking_gun, doubly_decisive, bayesian

9. Alternative_Explanation (id, description, probability, status)
   - Rival causal hypotheses
   - Status can be: active, refuted

10. Data_Source (id, type, credibility, bias_risk)
    - Origin of evidence
    - Type can be: interview, document, observation, artifact

### Edge Types (Connections between Nodes)
1. causes (Event → Event)
   - Direct/indirect causal relationship between events
   - Properties: certainty, mechanism_id, type

2. part_of_mechanism (Event → Causal_Mechanism)
   - Event is component of a causal mechanism
   - Properties: role (trigger, intermediate, outcome)

3. tests_hypothesis (Evidence → Hypothesis)
   - Evaluation results against hypotheses
   - Properties: inferential_test_id, probative_value, test_result

4. tests_mechanism (Evidence → Causal_Mechanism)
   - Evaluates existence/operation of mechanisms
   - Properties: inferential_test_id, probative_value, test_result

5. supports_alternative (Evidence → Alternative_Explanation)
   - Evidence supporting rival hypothesis
   - Properties: probative_value, certainty

6. refutes_alternative (Evidence → Alternative_Explanation)
   - Evidence weakening rival hypothesis
   - Properties: probative_value, certainty

7. enables (Condition → Event/Causal_Mechanism)
   - Conditions allowing events/mechanisms
   - Properties: necessity, certainty, type

8. constrains (Condition → Event/Causal_Mechanism)
   - Factors restricting events
   - Properties: certainty, type

9. provides_evidence (Data_Source → Evidence)
   - Source and quality of evidence
   - Properties: credibility, bias_risk, certainty

10. initiates (Actor → Event)
    - Agent initiates/participates in event
    - Properties: certainty, intention, agency

11. infers (Inference_Rule → Hypothesis/Causal_Mechanism)
    - Logical link from inference to outcomes
    - Properties: certainty, logic_type

12. updates_probability (Evidence → Hypothesis)
    - Bayesian updating documented via probability
    - Properties: prior_probability, posterior_probability, Bayes_factor

13. contradicts (Evidence → Evidence)
    - Conflicting evidence points
    - Properties: certainty, reason

14. supports (Evidence → Hypothesis)
    - Evidence supporting a hypothesis
    - Properties: certainty, strength

15. refutes (Evidence → Hypothesis)
    - Evidence refuting a hypothesis
    - Properties: certainty, strength

Return valid JSON conforming to this structure:
{{
  "nodes": [
    {{
      "id": "string",
      "type": "string (one of the node types)",
      "description": "string",
      // Additional properties as appropriate for the node type
    }},
    // ... more nodes
  ],
  "edges": [
    {{
      "source": "string (id of source node)",
      "target": "string (id of target node)",
      "label": "string (one of the edge types)",
      // Additional properties as appropriate for the edge type
    }},
    // ... more edges
  ]
}}

IMPORTANT:
- All node IDs should follow the pattern [Type initial][Number], e.g., E1 for Event 1, H2 for Hypothesis 2
- Every edge must reference existing node IDs in the nodes array
- If the text lacks sufficient detail for this comprehensive analysis, extract what you can but maintain the ontology structure
- Keep descriptions concise but informative (under 200 characters)
- If the text contains no relevant information, return valid JSON with empty arrays: {{"nodes": [], "edges": []}}

TEXT TO ANALYSE:
\"\"\"{text}\"\"\"
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
    print(f"📄 Reading {len(files)} file(s):")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            txt = f.read()
            big_text += txt + "\n\n"
            total += len(txt)
            print(f"  • {os.path.basename(fp):<40s} ({len(txt):,} chars)")
    print(f"📝 Total characters loaded: {total:,}\n")
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
            print("\n⚠️  JSON parsing failed. First 300 chars Gemini sent:\n"
                  "────────────────────────────────────────────────────")
            print(snippet)
            print("────────────────────────────────────────────────────")
            raise e


def validate_json_against_ontology(data: dict) -> tuple[bool, list[str]]:
    """
    Validate the JSON data against the ontology definition.
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
            
        # Check for required fields
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
            
            # Check if node type is valid
            if node_type not in NODE_TYPES:
                errors.append(f"Node {node.get('id', f'at index {i}')} has invalid type: {node_type}")
            else:
                # Check required fields for this node type
                for req_field in NODE_TYPES[node_type]["required"]:
                    if req_field not in node:
                        errors.append(f"Node {node.get('id', f'at index {i}')} of type {node_type} is missing required field: {req_field}")
    
    # Validate edges
    for i, edge in enumerate(data["edges"]):
        if not isinstance(edge, dict):
            errors.append(f"Edge at index {i} is not a dictionary")
            continue
            
        # Check for required fields
        if "source" not in edge:
            errors.append(f"Edge at index {i} is missing required 'source' field")
        elif edge["source"] not in node_ids:
            errors.append(f"Edge at index {i} references non-existent source node: {edge['source']}")
            
        if "target" not in edge:
            errors.append(f"Edge at index {i} is missing required 'target' field")
        elif edge["target"] not in node_ids:
            errors.append(f"Edge at index {i} references non-existent target node: {edge['target']}")
            
        if "label" not in edge:
            errors.append(f"Edge at index {i} is missing required 'label' field")
        else:
            edge_type = edge["label"]
            
            # Check if edge type is valid
            if edge_type not in EDGE_TYPES:
                errors.append(f"Edge at index {i} has invalid type: {edge_type}")
            elif "source" in edge and "target" in edge and edge["source"] in node_ids and edge["target"] in node_ids:
                # Check source and target node types are compatible with this edge type
                source_type = node_types.get(edge["source"])
                target_type = node_types.get(edge["target"])
                
                if source_type and target_type:
                    if source_type not in EDGE_TYPES[edge_type]["source_types"]:
                        errors.append(f"Edge {edge_type} at index {i} has incompatible source node type: {source_type}")
                    
                    if target_type not in EDGE_TYPES[edge_type]["target_types"]:
                        errors.append(f"Edge {edge_type} at index {i} has incompatible target node type: {target_type}")
    
    return len(errors) == 0, errors


def query_gemini(big_text: str, retry_count=0) -> dict:
    """Query Gemini API with retry mechanism for validation failures."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [types.Content(role="user",
                              parts=[types.Part.from_text(
                                  text=PROMPT_TEMPLATE.format(text=big_text))])]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    print(f"🚀 Streaming from Gemini (attempt {retry_count + 1}/{MAX_RETRIES})…")
    json_str, chunk_counter = "", count(1)
    for chunk in client.models.generate_content_stream(
            model=MODEL_NAME, contents=contents, config=config):
        if hasattr(chunk, "text"):
            json_str += chunk.text
        # pulse every 5 chunks
        if next(chunk_counter) % 5 == 0:
            print("•", end="", flush=True)

    print("\n✅ Stream complete. Parsing JSON…")
    
    try:
        # Parse the returned JSON
        graph_data = parse_json(json_str)
        
        # Validate against ontology
        is_valid, errors = validate_json_against_ontology(graph_data)
        
        if is_valid:
            print("✅ Validation successful: JSON conforms to ontology!")
            return graph_data
        else:
            print("❌ Validation failed. Errors found:")
            for error in errors:
                print(f"  ⚠️ {error}")
                
            # Retry if we haven't exceeded the maximum attempts
            if retry_count < MAX_RETRIES - 1:
                print(f"🔄 Retrying query ({retry_count + 2}/{MAX_RETRIES})...")
                time.sleep(2)  # Short delay between retries
                return query_gemini(big_text, retry_count + 1)
            else:
                print("❌ Maximum retry attempts reached. Using last response despite validation errors.")
                return graph_data
                
    except Exception as e:
        print(f"❌ Error processing response: {str(e)}")
        
        # Retry if we haven't exceeded the maximum attempts
        if retry_count < MAX_RETRIES - 1:
            print(f"🔄 Retrying query ({retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(2)  # Short delay between retries
            return query_gemini(big_text, retry_count + 1)
        else:
            print("❌ Maximum retry attempts reached. Raising error.")
            raise e


def build_visualization(data, html_out):
    """
    Creates a standalone HTML visualization using vis.js
    for the comprehensive process tracing ontology.
    """
    # Create formatted nodes for vis.js
    vis_nodes = []
    for node in data["nodes"]:
        # Prepare label based on node type
        if node["type"] == "Actor":
            label = f"{node['type']}: {node.get('name', node.get('description', node['id']))}"
        else:
            label = f"{node['type']}: {node.get('description', node['id'])}"
            
        # Truncate long labels
        if len(label) > 100:
            label = label[:97] + "..."
            
        # Prepare tooltip with all properties
        tooltip = "<div style='max-width:300px;'>"
        tooltip += f"<strong>ID:</strong> {node['id']}<br>"
        tooltip += f"<strong>Type:</strong> {node['type']}<br>"
        
        # Add all other properties to tooltip
        for key, value in node.items():
            if key not in ["id", "type"] and value:
                tooltip += f"<strong>{key.replace('_', ' ').title()}:</strong> {value}<br>"
                
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
        # Skip edges with non-existent source or target nodes
        if edge["source"] not in node_ids:
            print(f"⚠️  Skipping edge from non-existent source node '{edge['source']}' to '{edge['target']}'.")
            continue
        if edge["target"] not in node_ids:
            print(f"⚠️  Skipping edge from '{edge['source']}' to non-existent target node '{edge['target']}'.")
            continue
            
        # Prepare tooltip with edge properties
        tooltip = "<div style='max-width:250px;'>"
        tooltip += f"<strong>Type:</strong> {edge['label']}<br>"
        
        # Add all other properties to tooltip
        for key, value in edge.items():
            if key not in ["source", "target", "label"] and value:
                tooltip += f"<strong>{key.replace('_', ' ').title()}:</strong> {value}<br>"
                
        tooltip += "</div>"
        
        # Create edge with formatted properties
        vis_edges.append({
            "from": edge["source"],
            "to": edge["target"],
            "label": edge["label"],
            "title": tooltip,
            "width": edge.get("certainty", 1) if isinstance(edge.get("certainty"), (int, float)) else 1
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
    
    print(f"🎉 Graph written to {html_out}")
    
    # Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(html_out))


def read_text_file(file_path: str) -> str:
    """Read a single text file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"📄 Reading file: {os.path.basename(file_path)}")
    with open(file_path, encoding="utf-8") as f:
        txt = f.read()
    print(f"📝 Total characters loaded: {len(txt):,}\n")
    return txt


# ────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main execution function for the process tracing application."""
    try:
        # If the JSON file already exists, give the user an option to reuse it
        if os.path.exists(OUTPUT_JSON) and os.path.getsize(OUTPUT_JSON) > 0:
            print(f"📄 Found existing JSON data file: {OUTPUT_JSON}")
            if not HAS_GEMINI:
                print("ℹ️ Will use existing JSON data since Gemini API is not available.")
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
                print(f"✅ Successfully loaded data from {OUTPUT_JSON}")
            except Exception as e:
                print(f"❌ Error loading JSON data: {str(e)}")
                if not HAS_GEMINI:
                    print("❌ Cannot proceed without valid JSON data and Gemini API.")
                    return
                use_existing = False
        
        if not use_existing:
            # Process requires Gemini API
            if not HAS_GEMINI:
                print("❌ Cannot query Gemini API. Package not available.")
                return
                
            # Read the specified input file
            text = read_text_file(INPUT_FILE_PATH)
            
            # Query Gemini API with the text
            graph_data = query_gemini(text)

            # Save the graph data to a JSON file
            try:
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=4)
                print(f"💾 Raw graph data saved to {OUTPUT_JSON}")
            except IOError as e:
                print(f"⚠️  Error saving JSON data to {OUTPUT_JSON}: {e}")

        # Create visualization (works regardless of where data came from)
        build_visualization(graph_data, OUTPUT_HTML)
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback:\n{traceback_str}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        import traceback
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")

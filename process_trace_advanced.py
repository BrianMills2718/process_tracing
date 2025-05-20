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
    from google import genai
except ImportError:
    print("[ERROR] google-genai is not installed. Please activate the correct environment and install it.")
    sys.exit(1)

from core.ontology import get_gemini_graph_json_schema

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
        print(f"❌ No .txt files found in {project_dir}")
        sys.exit(1)
    return txts[0]

# --- Read Input Text ---
def read_input_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Build Gemini Schema ---
def get_schema():
    return get_gemini_graph_json_schema()

# --- Query Gemini API ---
def query_gemini(text, schema):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY environment variable not set and not found in .env file.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    MODEL_ID = "gemini-2.5-flash-preview-04-17"
    prompt = f"""
    Analyze the following historical text and extract a process tracing causal network using the provided ontology. Return the result as a JSON object with 'nodes' and 'edges'.\nStrictly follow the ontology for node and edge types and properties. If no relevant information is found, return empty arrays.\nText:\n{text}\n\nOntology (for reference): {schema}\n"""
    result = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': schema,
        },
    )
    return result.text

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
    args = parser.parse_args()

    input_text_dir = Path('input_text')
    projects = list_projects(input_text_dir)
    if not projects:
        print("❌ No projects found in input_text/.")
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
    print(f"\n📄 Using input: {input_path}\n")

    out_dir = Path(args.output) if args.output else Path('output_data') / project
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading input text from {input_path} ...")
    text = read_input_text(input_path)
    print("[INFO] Building Gemini schema...")
    schema = get_schema()
    print("[INFO] Querying Gemini (JSON mode)...")
    raw_json = query_gemini(text, schema)
    print("[INFO] Parsing and saving raw Gemini output...")
    try:
        graph_data = json.loads(raw_json)
    except Exception as e:
        print("[ERROR] Failed to parse Gemini output as JSON:", e)
        save_output(raw_json, out_dir, project, "raw_gemini_output.txt")
        sys.exit(1)
    json_path = save_output(graph_data, out_dir, project, "graph.json")
    print(f"[INFO] Saved graph JSON to {json_path}")
    html_path = out_dir / f"{project}_{timestamp()}_graph.html"
    print(f"[INFO] Visualizing graph to {html_path} ...")
    visualize_graph(graph_data, html_path, project)
    print("[DONE] Visualization complete.")

    # --- Automatically open the HTML visualization ---
    try:
        import webbrowser
        webbrowser.open_new_tab(str(html_path.resolve()))
        print(f"[INFO] Opened {html_path} in your default web browser.")
    except Exception as e:
        print(f"[WARNING] Could not open HTML automatically: {e}")

    # --- Run analysis and open analysis HTML ---
    print("[INFO] Running analysis on the generated graph JSON...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'core.analyze', str(json_path), '--html'
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print("❌ Analysis failed.")
            sys.exit(1)
        # Find the latest analysis HTML in the output directory
        html_files = sorted(out_dir.glob(f"*analysis.html"), reverse=True)
        if html_files:
            analysis_html = html_files[0]
            print(f"[INFO] Opening analysis HTML: {analysis_html}")
            webbrowser.open_new_tab(str(analysis_html.resolve()))
        else:
            print("[INFO] Analysis HTML file should have opened automatically if it was generated.")
    except Exception as e:
        print(f"[ERROR] Exception during analysis: {e}")

if __name__ == "__main__":
    main() 
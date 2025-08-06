#!/usr/bin/env python
"""
Process Tracing Project Orchestrator
-----------------------------------
Run extraction and analysis for a selected project in input_text/.
Automatically opens both the network graph and analysis HTML outputs.

USAGE:
  python process_trace.py --project PROJECT_NAME
  # or just
  python process_trace.py
  # and select a project interactively
"""
import os
import sys
import argparse
import subprocess
import glob
import time
from pathlib import Path
import webbrowser
from datetime import datetime

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
    txts = glob.glob(os.path.join(project_dir, '*.txt'))
    if not txts:
        print(f"❌ No .txt files found in {project_dir}")
        sys.exit(1)
    return txts[0]

def run_extract():
    # Run extraction as a module so core package is found
    result = subprocess.run([sys.executable, '-m', 'core.extract'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print("❌ Extraction failed.")
        sys.exit(1)

def find_latest_output(project_name, output_type):
    # output_type: 'graph.html', 'graph.json', 'analysis.html', etc.
    out_dir = Path('output_data') / project_name
    files = sorted(out_dir.glob(f"{project_name}_*_*.{output_type}"), reverse=True)
    return str(files[0]) if files else None

def run_analysis(json_path, project_name):
    # Run analysis and open HTML automatically
    result = subprocess.run([
        sys.executable, '-m', 'core.analyze', json_path, '--html'
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print("❌ Analysis failed.")
        sys.exit(1)
    # Find the latest analysis HTML
    html_path = find_latest_output(project_name, 'html')
    if html_path:
        webbrowser.open('file://' + str(Path(html_path).resolve()))
    else:
        print("⚠️ Could not find analysis HTML to open.")

def main():
    parser = argparse.ArgumentParser(description='Process Tracing Project Orchestrator')
    parser.add_argument('--project', type=str, help='Project name (subdirectory of input_text/)')
    args = parser.parse_args()

    input_text_dir = Path('input_text')
    projects = list_projects(input_text_dir)
    if not projects:
        print("❌ No projects found in input_text/.")
        sys.exit(1)

    project = args.project
    if not project or project not in projects:
        project = prompt_for_project(projects)

    project_dir = input_text_dir / project
    txt_file = find_first_txt(project_dir)
    print(f"\n📄 Using input: {txt_file}\n")

    # Run extraction (core.extract will pick up the first .txt in the selected project)
    run_extract()

    # Find the latest graph JSON for this project
    json_path = find_latest_output(project, 'json')
    if not json_path:
        print(f"❌ No graph JSON found for project {project}.")
        sys.exit(1)
    print(f"\n🔍 Using graph JSON: {json_path}\n")

    # Open the network graph HTML (from extraction)
    html_graph_path = find_latest_output(project, 'html')
    if html_graph_path:
        print(f"🌐 Opening network graph: {html_graph_path}")
        webbrowser.open('file://' + str(Path(html_graph_path).resolve()))
        time.sleep(1)  # Give browser a moment to open
    else:
        print("⚠️ Could not find network graph HTML to open.")

    # Run analysis and open the analysis HTML
    run_analysis(json_path, project)

if __name__ == "__main__":
    main() 
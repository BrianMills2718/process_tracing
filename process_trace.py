#!/usr/bin/env python
"""
Process Tracing Universal Text Analyzer
--------------------------------------
A utility script to analyze any text file using the core process tracing modules.
It handles environment checks, input preparation, LLM data extraction (via core.extract),
optional ontology fixing, and subsequent theoretical analysis (via core.analyze).

USAGE: (from project root)
  conda activate crest
  python process_trace.py path/to/your/textfile.txt [options]
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# Core modules are now in core/ directory
CORE_DIR = Path(__file__).parent / "core"
PROJECT_ROOT = Path(__file__).parent

def check_conda_env():
    # ... (implementation as before) ...
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env != 'crest':
        print(f"❌ Env: {current_env or 'None'}. Expected 'crest'. Activate 'crest' and retry.")
        return False
    print(f"✅ Conda env: {current_env}")
    return True

def import_google_genai():
    # ... (implementation as before) ...
    try:
        from google import genai
        print("✅ google.genai imported.")
        return True
    except ImportError:
        print("❌ Failed to import google.genai. pip install google-generativeai")
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Tracing Universal Text Analyzer')
    parser.add_argument('input_file', type=str, help='Text file to analyze (path relative to project root or absolute)')
    parser.add_argument('--output-base', '-o', type=str, help='Base name for output files (e.g., my_study). Outputs will be in output_data/.')
    parser.add_argument('--fix', '-f', action='store_true', help='Run ontology-fixing script after data extraction')
    parser.add_argument('--analyze', '-a', action='store_true', help='Run MD analysis on generated/fixed JSON')
    parser.add_argument('--analyze-html', '-ah', action='store_true', help='Generate HTML analysis with visualizations (implies --analyze and --theory)')
    return parser.parse_args()

def get_basename(filepath):
    return Path(filepath).stem

def prepare_input_text_for_pt2(input_file_rel_to_project_root: str) -> Path:
    """Ensures the input text is under input_text/ and returns its absolute path."""
    input_file_path = PROJECT_ROOT / input_file_rel_to_project_root
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    # pt2.py expects its input file to be in input_text/ within the project.
    # This function now just resolves the path for configure_pt2.
    # configure_pt2 will use this absolute path.
    # The original logic copied files; now we just ensure pt2 gets the correct absolute path.
    print(f"✅ Using input file: {input_file_path}")
    return input_file_path.resolve()

def configure_pt2_for_text(pt2_source_path: Path, temp_script_path: Path, 
                         input_text_abs_path: Path, 
                         output_json_abs_path: Path, output_html_abs_path: Path):
    """Configure a temporary copy of extract.py for a specific text file and outputs."""
    with open(pt2_source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Paths need to be escaped for string literals in the generated script
    input_file_str = str(input_text_abs_path).replace('\\', '\\\\')
    output_json_str = str(output_json_abs_path).replace('\\', '\\\\')
    output_html_str = str(output_html_abs_path).replace('\\', '\\\\')

    lines = content.split('\n')
    config_map = {
        "INPUT_FILE_PATH": f'INPUT_FILE_PATH = r"{input_file_str}"',
        "OUTPUT_JSON": f'OUTPUT_JSON = r"{output_json_str}"',
        "OUTPUT_HTML": f'OUTPUT_HTML = r"{output_html_str}"'
    }
    for i, line in enumerate(lines):
        for key, val_str in config_map.items():
            if line.strip().startswith(key):
                lines[i] = val_str
                config_map.pop(key)
                break
        if not config_map: break
    
    # Non-interactive modification for pt2.py's main() logic
    prompt_logic_found = False
    for i in range(len(lines)):
        if "if os.path.exists(OUTPUT_JSON) and os.path.getsize(OUTPUT_JSON) > 0:" in lines[i]:
            lines[i+1] = "            print(f\"📄 Found existing JSON data file: {{OUTPUT_JSON}}\")"
            lines[i+2] = "            if not HAS_GEMINI:"
            lines[i+3] = "                print(\"ℹ️ Gemini unavailable. Will use existing JSON if valid.\")"
            lines[i+4] = "                should_reuse_data = True" # Corresponds to var in pt2's main
            lines[i+5] = "            else:"
            lines[i+6] = "                print(\"ℹ️ Orchestrated call: Forcing data regeneration from Gemini.\")"
            lines[i+7] = "                should_reuse_data = False"
            prompt_logic_found = True
            break 
    if not prompt_logic_found: 
        print("⚠️ Warning: Could not auto-modify pt2.py for non-interactive mode.")

    with open(temp_script_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"✅ temp_extract.py configured for: {input_text_abs_path}")
    return True

def run_fix_script(json_to_fix_path: Path, fixed_json_output_path: Path, fixed_html_vis_output_path: Path):
    """Runs the ontology fixing script (core/fix_json_ontology.py)."""
    fix_script_module_path = CORE_DIR / "fix_json_ontology.py"
    cmd = [
        sys.executable, str(fix_script_module_path),
        str(json_to_fix_path),
        "--output-json", str(fixed_json_output_path),
        "--output-html", str(fixed_html_vis_output_path)
    ]
    print(f"🔧 Running fix script: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print("--- Fix Script STDOUT ---"); print(process.stdout)
        if process.stderr: print("--- Fix Script STDERR ---"); print(process.stderr)
        return fixed_json_output_path.exists() and fixed_json_output_path.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running fix script: {e}")
        print(f"STDOUT:\n{e.stdout}"); print(f"STDERR:\n{e.stderr}")
        return False

def main():
    args = parse_arguments()
    if not check_conda_env() or not import_google_genai(): return

    input_file_abs_path = (PROJECT_ROOT / args.input_file).resolve()
    if not input_file_abs_path.is_file():
        print(f"❌ Input file not found: {input_file_abs_path}"); return

    output_base_name = args.output_base if args.output_base else get_basename(args.input_file)
    
    # Define output directories relative to project root
    json_dir = PROJECT_ROOT / "output_data" / "json"
    reports_dir = PROJECT_ROOT / "output_data" / "reports"
    charts_dir = PROJECT_ROOT / "output_data" / "charts"
    json_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60 + f"\nUNIVERSAL TEXT ANALYSIS: {input_file_abs_path.name}\n" + "=" * 60)

    # Outputs for extract.py execution
    pt2_json_output = json_dir / f"{output_base_name}_llm_data.json"
    pt2_html_vis_output = reports_dir / f"{output_base_name}_llm_visualization.html"
    temp_pt2_script_path = PROJECT_ROOT / "temp_extract.py" # temp script at project root

    if not configure_pt2_for_text(CORE_DIR / "extract.py", temp_pt2_script_path, 
                                input_file_abs_path, 
                                pt2_json_output, pt2_html_vis_output):
        print("❌ Configuration of extract.py failed. Halting."); return

    print(f"\n🚀 Running core.extract (via {temp_pt2_script_path.name}) for data extraction...")
    data_extraction_successful = False
    try:
        process = subprocess.run([sys.executable, str(temp_pt2_script_path)], check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print("--- core.extract STDOUT ---"); print(process.stdout)
        if process.stderr: print("--- core.extract STDERR ---"); print(process.stderr)
        if pt2_json_output.exists() and pt2_json_output.stat().st_size > 0:
            print(f"\n✅ core.extract finished. Data saved to {pt2_json_output}")
            data_extraction_successful = True
        else:
            print(f"\n❌ core.extract ran but {pt2_json_output} was not created/empty.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {temp_pt2_script_path.name}: {e}")
        print(f"STDOUT:\n{e.stdout}"); print(f"STDERR:\n{e.stderr}")
    finally:
        if temp_pt2_script_path.exists(): temp_pt2_script_path.unlink()

    if not data_extraction_successful:
        print("\n🛑 Data extraction failed. Halting."); return

    json_to_analyze = pt2_json_output
    analysis_file_base = f"{output_base_name}_llm_data"
    vis_file_to_analyze = pt2_html_vis_output # Visualization from extract.py

    if args.fix:
        print(f"\n🔧 Attempting to fix ontology for {pt2_json_output}...")
        fixed_json_output = json_dir / f"{output_base_name}_fixed_data.json"
        fixed_html_vis_output = reports_dir / f"{output_base_name}_fixed_visualization.html"
        
        if run_fix_script(pt2_json_output, fixed_json_output, fixed_html_vis_output):
            print(f"✅ Fixing script successful. Using {fixed_json_output} for analysis.")
            json_to_analyze = fixed_json_output
            analysis_file_base = f"{output_base_name}_fixed_data"
            vis_file_to_analyze = fixed_html_vis_output
        else:
            print(f"❌ Fixing script failed or produced no output. Using original data: {pt2_json_output}")
    
    if args.analyze or args.analyze_html:
        if not json_to_analyze.exists():
            print(f"\n❌ Cannot analyze: {json_to_analyze} not found."); return
            
        print(f"\n📊 Generating analysis for {json_to_analyze.name}...")
        analyze_script_path = CORE_DIR / "analyze.py"
        report_extension = "html" if args.analyze_html else "md"
        report_output_path = reports_dir / f"{analysis_file_base}_analysis.{report_extension}"
        
        analyze_cmd = [sys.executable, str(analyze_script_path), str(json_to_analyze), "--theory"]
        if args.analyze_html: analyze_cmd.append("--html")
        analyze_cmd.extend(["--output", str(report_output_path)])
        analyze_cmd.extend(["--charts-dir", str(charts_dir)]) # Pass charts_dir
        
        try:
            process = subprocess.run(analyze_cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
            print("--- Analysis Script STDOUT ---"); print(process.stdout)
            if process.stderr: print("--- Analysis Script STDERR ---"); print(process.stderr)
            print(f"\n✅ Analysis saved to {report_output_path}")
            if args.analyze_html:
                try: 
                    import webbrowser
                    webbrowser.open(report_output_path.resolve().as_uri())
                except Exception: pass
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error running analysis: {e}"); print(f"STDOUT:\n{e.stdout}"); print(f"STDERR:\n{e.stderr}")
        except FileNotFoundError:
            print(f"\n❌ Error: {analyze_script_path} not found.")

    print(f"\n🎉 {Path(__file__).name} finished.")

if __name__ == "__main__":
    main() 
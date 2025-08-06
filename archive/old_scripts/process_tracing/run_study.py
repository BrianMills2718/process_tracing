import json
import os
import argparse
from pathlib import Path
import datetime
from process_trace_advanced import execute_single_case_processing
from core.cross_case_synthesis import perform_cross_case_synthesis_and_report

def parse_study_args():
    parser = argparse.ArgumentParser(description="Process Tracing Multi-Case Study Runner")
    parser.add_argument("study_config_file", help="Path to the JSON study configuration file")
    return parser.parse_args()

def run_study():
    args = parse_study_args()

    try:
        with open(args.study_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Study configuration file not found: {args.study_config_file}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in study configuration file: {args.study_config_file}")
        return

    study_name = config.get("study_name", "UnnamedStudy")
    global_hypothesis_text = config.get("global_hypothesis_text")
    global_hypothesis_id = config.get("global_hypothesis_id")

    if not global_hypothesis_text:
        global_hypothesis_text = None
    if not global_hypothesis_id:
        global_hypothesis_id = None

    case_files_config = config.get("case_files", [])
    output_dir_base_str = config.get("output_directory_base", "output_data/studies")

    if not case_files_config:
        print("[ERROR] 'case_files' list is empty or not defined in the study config.")
        return

    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_output_dir = Path(output_dir_base_str) / f"{study_name}_{current_timestamp}"
    study_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Study output directory: {study_output_dir}")

    all_case_analysis_summary_paths = []

    for case_file_rel_path_str in case_files_config:
        case_file_abs_path = Path(case_file_rel_path_str).resolve()

        if not case_file_abs_path.exists():
            print(f"[WARNING] Case file not found: {case_file_abs_path}. Skipping.")
            continue

        project_name_for_case = case_file_abs_path.parent.name
        case_name_stem = case_file_abs_path.stem

        output_dir_for_this_case = study_output_dir / project_name_for_case / case_name_stem
        output_dir_for_this_case.mkdir(parents=True, exist_ok=True)

        print(f"--- Processing Case: {case_file_abs_path} ---")
        print(f"Outputting to: {output_dir_for_this_case}")

        try:
            summary_json_path = execute_single_case_processing(
                case_file_path_str=str(case_file_abs_path),
                output_dir_for_case_str=str(output_dir_for_this_case),
                project_name_str=f"{project_name_for_case}_{case_name_stem}",
                global_hypothesis_text=global_hypothesis_text,
                global_hypothesis_id=global_hypothesis_id
            )
            if summary_json_path:
                all_case_analysis_summary_paths.append(summary_json_path)
                print(f"[SUCCESS] Completed processing for {case_file_abs_path}. Summary at: {summary_json_path}")
            else:
                print(f"[WARNING] Processing completed for {case_file_abs_path}, but no summary JSON path was returned.")
        except Exception as e:
            print(f"[ERROR] Failed to process case {case_file_abs_path}: {e}")
        print(f"--- Finished Case: {case_file_abs_path} ---\n")

    print(f"[INFO] All cases processed. Collected {len(all_case_analysis_summary_paths)} summary JSON paths.")

    # --- Perform Cross-Case Synthesis ---
    if all_case_analysis_summary_paths:
        print(f"[INFO] Proceeding to cross-case synthesis using {len(all_case_analysis_summary_paths)} case summaries...")
        try:
            synthesis_report_path = perform_cross_case_synthesis_and_report(
                list_of_summary_json_paths=all_case_analysis_summary_paths,
                study_output_dir=study_output_dir, # This is Path object from earlier in run_study.py
                global_hypothesis_id=global_hypothesis_id, # This can be None
                global_hypothesis_text=global_hypothesis_text, # This can be None
                study_name=study_name # From config
            )
            if synthesis_report_path:
                print(f"[SUCCESS] Cross-case synthesis report generated at: {synthesis_report_path}")
                # Optionally, try to open the report:
                # import webbrowser
                # try:
                #     webbrowser.open('file://' + str(Path(synthesis_report_path).resolve()))
                # except Exception as e_wb:
                #     print(f"[WARN] Could not open synthesis HTML report in browser: {e_wb}")
            else:
                print("[WARNING] Cross-case synthesis completed, but no report path was returned.")
        except Exception as e_synth:
            print(f"[ERROR] Failed during cross-case synthesis: {e_synth}")
    else:
        print("[INFO] No valid case summaries collected, skipping cross-case synthesis.")

    print(f"[INFO] Multi-case study run '{study_name}' finished.")

if __name__ == "__main__":
    run_study() 
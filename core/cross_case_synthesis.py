import json
from pathlib import Path
import datetime
from collections import Counter
import textwrap  # For formatting report content
from core.llm_reporting_utils import generate_narrative_summary_with_llm
from process_trace_advanced import query_llm # Added for LLM-generated emergent hypotheses

# Issue #83 Fix: Add semantic matching for cross-case hypothesis comparison
def calculate_hypothesis_similarity(hyp1_id, hyp1_desc, hyp2_id, hyp2_desc, threshold=0.4):
    """
    Calculate semantic similarity between two hypotheses for cross-case matching.
    
    Args:
        hyp1_id: ID of first hypothesis
        hyp1_desc: Description of first hypothesis  
        hyp2_id: ID of second hypothesis
        hyp2_desc: Description of second hypothesis
        threshold: Similarity threshold for matching (0.0-1.0, default 0.4)
        
    Returns:
        bool: True if hypotheses are semantically similar enough to match
    """
    # Exact ID match (legacy compatibility)
    if hyp1_id == hyp2_id:
        return True
    
    # If no descriptions available, fall back to ID comparison
    if not hyp1_desc or not hyp2_desc:
        return hyp1_id == hyp2_id
    
    # Normalize descriptions for comparison
    desc1_words = set(hyp1_desc.lower().split())
    desc2_words = set(hyp2_desc.lower().split())
    
    # Remove common stop words that don't carry semantic meaning
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
    desc1_words = desc1_words - stop_words
    desc2_words = desc2_words - stop_words
    
    # Add synonym expansion for better matching
    synonym_groups = [
        {'economic', 'financial', 'monetary', 'fiscal'},
        {'crisis', 'collapse', 'instability', 'turmoil', 'unrest'},
        {'political', 'governmental', 'regime'},
        {'causes', 'leads', 'results', 'produces', 'triggers'},
        {'change', 'transformation', 'shift', 'transition'},
        {'instability', 'unrest', 'disorder', 'chaos', 'turbulence'}
    ]
    
    # Expand words with synonyms
    desc1_expanded = desc1_words.copy()
    desc2_expanded = desc2_words.copy()
    
    for group in synonym_groups:
        if desc1_words.intersection(group):
            desc1_expanded.update(group)
        if desc2_words.intersection(group):
            desc2_expanded.update(group)
    
    # Calculate Jaccard similarity with synonym expansion
    if not desc1_expanded or not desc2_expanded:
        return False
        
    intersection = len(desc1_expanded.intersection(desc2_expanded))
    union = len(desc1_expanded.union(desc2_expanded))
    
    similarity = intersection / union if union > 0 else 0.0
    
    return similarity >= threshold

def load_and_extract_hypothesis_data(summary_json_path, global_hypothesis_id, global_hypothesis_desc=None):
    """
    Load a single case analysis summary JSON and extract the evaluation data for the global hypothesis.
    
    Args:
        summary_json_path (str): Path to the case's analysis_summary.json file
        global_hypothesis_id (str): The ID of the global hypothesis to extract
        global_hypothesis_desc (str): Description of global hypothesis for semantic matching
        
    Returns:
        dict: Dictionary with key details for the hypothesis from this case, or None if not found
    """
    try:
        with open(summary_json_path, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
        
        # Extract the case identifier from the JSON or from the filename
        case_identifier = case_data.get('case_identifier', Path(summary_json_path).stem)
        
        # Issue #83 Fix: Find specific hypothesis using semantic matching
        if 'hypotheses_evaluation' in case_data:
            for hypothesis in case_data['hypotheses_evaluation']:
                hyp_id = hypothesis.get('hypothesis_id', '')
                hyp_desc = hypothesis.get('description', '')
                
                # Use semantic matching instead of exact string comparison
                if calculate_hypothesis_similarity(global_hypothesis_id, global_hypothesis_desc, hyp_id, hyp_desc):
                    # Found matching hypothesis, extract key details
                    return {
                        'case_identifier': case_identifier,
                        'matched_hypothesis_id': hyp_id,  # Track which hypothesis actually matched
                        'assessment': hypothesis.get('assessment', 'Undetermined'),
                        'balance_score': hypothesis.get('balance_score', 0.0),
                        'supporting_evidence_count': hypothesis.get('supporting_evidence_count', 0),
                        'refuting_evidence_count': hypothesis.get('refuting_evidence_count', 0),
                        'supporting_evidence_ids': hypothesis.get('supporting_evidence_ids', []),
                        'refuting_evidence_ids': hypothesis.get('refuting_evidence_ids', [])
                    }
        
        print(f"[WARNING] Global hypothesis ID '{global_hypothesis_id}' not found in {summary_json_path}")
        return None
        
    except FileNotFoundError:
        print(f"[WARNING] Summary file not found: {summary_json_path}")
        return None
    except json.JSONDecodeError:
        print(f"[WARNING] Invalid JSON in summary file: {summary_json_path}")
        return None
    except Exception as e:
        print(f"[WARNING] Error loading hypothesis data from {summary_json_path}: {e}")
        return None

def format_synthesis_report_html(synthesis_data, study_name, global_hypothesis_text, global_hypothesis_id):
    """
    Format the cross-case synthesis data as an HTML report.
    
    Args:
        synthesis_data (dict): Aggregated data from all case summaries
        study_name (str): Name of the study
        global_hypothesis_text (str): The text of the global hypothesis
        global_hypothesis_id (str): The ID of the global hypothesis
        
    Returns:
        str: HTML report content
    """
    # LLM-powered overall summary
    overall_assessment_prompt = f"Provide an overall analytical summary regarding the global hypothesis '{global_hypothesis_text}' based on its evaluation across multiple cases, as detailed in the provided data. Mention the general trend of support or refutation and any notable patterns."
    llm_overall_summary = generate_narrative_summary_with_llm(synthesis_data, overall_assessment_prompt)
    # Basic HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cross-Case Synthesis: {study_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .supported {{ color: #27ae60; font-weight: bold; }}
            .refuted {{ color: #e74c3c; font-weight: bold; }}
            .mixed {{ color: #f39c12; font-weight: bold; }}
            .undetermined {{ color: #7f8c8d; font-style: italic; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #2980b9; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Cross-Case Synthesis Report: {study_name}</h1>
        <div class='llm-summary'><h2>LLM Analytical Synthesis Summary</h2><p>{llm_overall_summary}</p></div>
        <div class="summary">
            <h2>Global Hypothesis</h2>
            <p><strong>ID:</strong> {global_hypothesis_id}</p>
            <p><strong>Text:</strong> {global_hypothesis_text}</p>
            <h2>Overall Assessment</h2>
            <p>{synthesis_data.get('overall_assessment_summary','')}</p>
            <ul>
                <li><strong>Total cases analyzed:</strong> {synthesis_data.get('total_cases_processed','')}</li>
                <li><strong>Cases supporting hypothesis:</strong> {synthesis_data.get('cases_supporting_hypothesis','')}</li>
                <li><strong>Cases refuting hypothesis:</strong> {synthesis_data.get('cases_refuting_hypothesis','')}</li>
                <li><strong>Cases with mixed/undetermined evidence:</strong> {synthesis_data.get('cases_mixed_undetermined','')}</li>
            </ul>
        </div>
        
        <h2>Individual Case Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Case</th>
                    <th>Assessment</th>
                    <th>Evidence Balance Score</th>
                    <th>Supporting Evidence Count</th>
                    <th>Refuting Evidence Count</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add rows for each case
    for case_result in synthesis_data['case_results']:
        assessment = case_result.get('assessment', 'Undetermined')
        assessment_class = 'undetermined'
        # Issue #82 Fix: Handle new elimination-based assessment categories
        if ('Supported' in assessment or 'Confirmed' in assessment) and not ('Contested' in assessment or 'Mixed' in assessment):
            assessment_class = 'supported'
        elif ('Refuted' in assessment or 'Eliminated' in assessment) and not ('Contested' in assessment or 'Mixed' in assessment):
            assessment_class = 'refuted'
        elif 'Mixed' in assessment or 'Contested' in assessment:
            assessment_class = 'mixed'
            
        html += f"""
                <tr>
                    <td>{case_result['case_identifier']}</td>
                    <td class="{assessment_class}">{assessment}</td>
                    <td>{case_result.get('balance_score', 0.0):.2f}</td>
                    <td>{case_result.get('supporting_evidence_count', 0)}</td>
                    <td>{case_result.get('refuting_evidence_count', 0)}</td>
                </tr>
        """
    
    # Close table and document
    html += """
            </tbody>
        </table>
        
        <h2>Synthesis Methodology</h2>
        <p>This cross-case synthesis report aggregates results from multiple case studies, each analyzed using process tracing methodology.
           The overall assessment is based on the pattern of hypothesis evaluation across all cases, with consideration given to the
           strength of evidence in each case (measured by evidence balance scores and counts of supporting/refuting evidence).</p>
           
        <p><em>Report generated: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
    </body>
    </html>
    """
    
    return html

def perform_cross_case_synthesis_and_report(
    list_of_summary_json_paths, 
    study_output_dir, 
    global_hypothesis_id=None, # Now optional
    global_hypothesis_text=None, # Now optional
    study_name="UnnamedStudy"
):
    import datetime
    from collections import Counter
    print(f"[INFO] Beginning cross-case synthesis.")
    all_case_data = []
    for summary_path in list_of_summary_json_paths:
        try:
            with open(summary_path, 'r', encoding='utf-8') as f_json:
                case_summary = json.load(f_json)
                all_case_data.append(case_summary)
        except Exception as e:
            print(f"[WARNING] Could not load or parse case summary {summary_path}: {e}")
            continue
    if not all_case_data:
        print("[ERROR] No valid case summary data loaded for synthesis.")
        return None
    synthesis_data_for_report = {}
    report_html_content = ""
    if global_hypothesis_id and global_hypothesis_text:
        print(f"[INFO] Performing focused synthesis for global hypothesis ID: {global_hypothesis_id}")
        temp_case_results_focused = []
        cases_with_global_hyp = 0
        num_supporting = 0
        num_refuting = 0
        num_mixed_undetermined = 0

        for case_summary in all_case_data:
            found_hyp_data_for_case = None
            for hyp_eval in case_summary.get('hypotheses_evaluation', []):
                # Issue #83 Fix: Use semantic matching instead of exact string comparison
                hyp_id = hyp_eval.get('hypothesis_id', '')
                hyp_desc = hyp_eval.get('description', '')
                
                if calculate_hypothesis_similarity(global_hypothesis_id, global_hypothesis_text, hyp_id, hyp_desc):
                    found_hyp_data_for_case = hyp_eval
                    cases_with_global_hyp +=1
                    break
            if found_hyp_data_for_case:
                assessment = found_hyp_data_for_case.get('assessment', 'Undetermined')
                # Issue #82 Fix: Handle new elimination-based assessment categories
                if ('Supported' in assessment or 'Confirmed' in assessment) and not ('Contested' in assessment or 'Mixed' in assessment):
                    num_supporting += 1
                elif ('Refuted' in assessment or 'Eliminated' in assessment) and not ('Contested' in assessment or 'Mixed' in assessment):
                    num_refuting += 1
                else:
                    num_mixed_undetermined +=1
                temp_case_results_focused.append({
                    "case_identifier": case_summary.get('case_identifier', Path(case_summary.get('filename', 'Unknown_Case')).stem),
                    "assessment": assessment,
                    "balance_score": found_hyp_data_for_case.get('balance_score', 0.0),
                    "description": found_hyp_data_for_case.get('description', global_hypothesis_text),
                    "supporting_evidence_count": found_hyp_data_for_case.get('supporting_evidence_count',0),
                    "refuting_evidence_count": found_hyp_data_for_case.get('refuting_evidence_count',0)
                })
        synthesis_data_for_report = {
            "synthesis_type": "global_hypothesis_focused",
            "global_hypothesis_id": global_hypothesis_id,
            "global_hypothesis_text": global_hypothesis_text,
            "total_cases_processed": len(all_case_data),
            "cases_evaluating_hypothesis": cases_with_global_hyp,
            "case_results": temp_case_results_focused,
            "cases_supporting_hypothesis": num_supporting,
            "cases_refuting_hypothesis": num_refuting,
            "cases_mixed_undetermined": num_mixed_undetermined,
            "overall_assessment_summary": f"Hypothesis '{global_hypothesis_text}' was evaluated in {cases_with_global_hyp} of {len(all_case_data)} cases. Supported in {num_supporting}, Refuted in {num_refuting}, Mixed/Undetermined in {num_mixed_undetermined}."
        }
    else:
        print("[INFO] Performing exploratory synthesis of emergent hypotheses (no global hypothesis specified).")
        all_extracted_hypotheses = []
        for case_summary in all_case_data:
            for hyp_eval in case_summary.get('hypotheses_evaluation', []):
                all_extracted_hypotheses.append({
                    "case_identifier": case_summary.get('case_identifier', Path(case_summary.get('filename', 'Unknown_Case')).stem),
                    "hypothesis_id": hyp_eval.get('hypothesis_id'),
                    "description": hyp_eval.get('description'),
                    "assessment": hyp_eval.get('assessment'),
                    "balance_score": hyp_eval.get('balance_score')
                })
        
        unique_hypotheses_summary = {}
        for hyp_item in all_extracted_hypotheses:
            desc = hyp_item['description']
            if desc not in unique_hypotheses_summary:
                unique_hypotheses_summary[desc] = {
                    "description": desc,
                    "appears_in_cases_ids": [],
                    "assessments_by_case": {},
                    "assessment_summary_str": ""
                }
            unique_hypotheses_summary[desc]['appears_in_cases_ids'].append(hyp_item['case_identifier'])
            unique_hypotheses_summary[desc]['assessments_by_case'][hyp_item['case_identifier']] = hyp_item['assessment']
        
        for desc, data in unique_hypotheses_summary.items():
            assessment_counts = Counter(data['assessments_by_case'].values())
            data['assessment_summary_str'] = "; ".join([f"{k}: {v}" for k,v in assessment_counts.items()])
            data['num_cases_appeared'] = len(data['appears_in_cases_ids'])

        # Prepare summary for LLM input
        llm_input_summary_parts = [f"Summary of Emergent Themes from {len(all_case_data)} Case Analyses:"]
        for desc, data in unique_hypotheses_summary.items():
            llm_input_summary_parts.append(
                f"- Hypothesis Theme: '{data['description']}'\n"
                f"  - Appeared in {data['num_cases_appeared']} cases: {', '.join(data['appears_in_cases_ids'])}\n"
                f"  - Assessment Pattern: {data['assessment_summary_str']}"
            )
        llm_input_summary_text = "\n".join(llm_input_summary_parts)

        # LLM call to generate new overarching hypotheses
        llm_generated_emergent_hypotheses = []
        emergent_hyp_prompt = f"""You are a research analyst. Based on the following summarized findings from {len(all_case_data)} related case studies:

{llm_input_summary_text}

Your task is to:
1. Identify and formulate 2-4 overarching plausible causal hypotheses that could explain common patterns, outcomes, or processes observed across these cases. These hypotheses should be general enough to potentially span multiple cases.
2. For each hypothesis you formulate, provide a brief (1-2 sentence) justification based on the summarized findings.

Output as a JSON list with each element being a dictionary with keys 'emergent_hypothesis_text' and 'justification'. For example: [{'emergent_hypothesis_text': '...', 'justification': '...'}, ...]
"""
        print("[INFO] Calling LLM to generate overarching emergent hypotheses...")
        try:
            # Assuming query_llm handles API key internally via .env or environment
            response_text = query_llm(text_content="", schema=None, system_instruction_text=emergent_hyp_prompt)
            llm_generated_emergent_hypotheses = json.loads(response_text)
            if not isinstance(llm_generated_emergent_hypotheses, list):
                 llm_generated_emergent_hypotheses = [] # Ensure it's a list
            print(f"[SUCCESS] LLM generated {len(llm_generated_emergent_hypotheses)} new emergent hypotheses.")
        except Exception as e_llm_emergent:
            print(f"[WARNING] Failed to get or parse LLM-generated emergent hypotheses: {e_llm_emergent}")
            llm_generated_emergent_hypotheses = [{'error': f'Failed to generate/parse: {str(e_llm_emergent)}'}]

        synthesis_data_for_report = {
            "synthesis_type": "exploratory_emergent_hypotheses",
            "total_cases_processed": len(all_case_data),
            "discovered_hypotheses_summary": sorted(list(unique_hypotheses_summary.values()), key=lambda x: x['num_cases_appeared'], reverse=True),
            "llm_generated_emergent_hypotheses": llm_generated_emergent_hypotheses
        }

    report_html_content = format_synthesis_report_html(
        synthesis_data=synthesis_data_for_report,
        study_name=study_name,
        global_hypothesis_text=global_hypothesis_text,
        global_hypothesis_id=global_hypothesis_id
    )
    report_filename = f"{study_name}_cross_case_synthesis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = Path(study_output_dir) / report_filename
    try:
        with open(report_path, 'w', encoding='utf-8') as f_html:
            f_html.write(report_html_content)
        print(f"[SUCCESS] Cross-case synthesis report saved to: {report_path}")
        return str(report_path)
    except Exception as e:
        print(f"[ERROR] Could not write synthesis report: {e}")
        return None 
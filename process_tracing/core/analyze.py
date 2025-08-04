def generate_json_summary(analysis_results, G, case_filename):
    """
    Generate a structured JSON summary of the analysis, focusing on hypothesis evaluations and key metrics.
    """
    from datetime import datetime
    summary = {
        "case_identifier": case_filename,
        "timestamp": datetime.now().isoformat(),
        "network_overview": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_type_distribution": analysis_results.get('metrics', {}).get('node_type_distribution', {}),
            "edge_type_distribution": analysis_results.get('metrics', {}).get('edge_type_distribution', {})
        },
        "hypotheses_evaluation": [],
        "causal_chains_summary": {
            "count": len(analysis_results.get('causal_chains', [])),
            "top_chains_details": []
        },
        "mechanisms_summary": {
            "count": len(analysis_results.get('mechanisms', [])),
            "evaluated_mechanisms": []
        }
    }
    # Hypotheses evaluation
    for hyp_id, hypothesis_data in analysis_results.get('evidence_analysis', {}).items():
        summary["hypotheses_evaluation"].append({
            "hypothesis_id": hyp_id,
            "description": hypothesis_data.get('description', 'N/A'),
            "assessment": hypothesis_data.get('assessment', 'Undetermined'),
            "balance_score": hypothesis_data.get('balance', 0.0),
            "prior_probability": hypothesis_data.get('prior_probability'),
            "posterior_probability": hypothesis_data.get('posterior_probability'),
            "supporting_evidence_count": len(hypothesis_data.get('supporting_evidence', [])),
            "refuting_evidence_count": len(hypothesis_data.get('refuting_evidence', [])),
            "supporting_evidence_ids": [ev['id'] for ev in hypothesis_data.get('supporting_evidence', [])],
            "refuting_evidence_ids": [ev['id'] for ev in hypothesis_data.get('refuting_evidence', [])]
        })
    # Causal chains summary (top 3 chains as example)
    for chain in analysis_results.get('causal_chains', [])[:3]:
        summary["causal_chains_summary"]["top_chains_details"].append({
            "path": chain.get('path', []),
            "length": chain.get('length', len(chain.get('path', []))),
            "edge_types": chain.get('edge_types', [])
        })
    # Mechanisms summary
    for mech in analysis_results.get('mechanisms', []):
        summary["mechanisms_summary"]["evaluated_mechanisms"].append({
            "mechanism_id": mech.get('id'),
            "name": mech.get('name'),
            "completeness": mech.get('completeness'),
            "confidence": mech.get('confidence'),
            "num_contributing_factors": len(mech.get('contributing_factors', [])),
            "num_effects": len(mech.get('effects', []))
        })
    return summary

def main():
    # ... existing code ...
    # After analysis_results and theoretical_insights are generated
    json_summary_data = generate_json_summary(analysis_results, G, args.json_file)
    # Determine output path for JSON summary
    json_summary_output_filename = f"{project_name}_analysis_summary_{now_str}.json"
    json_summary_output_path = output_path.parent / json_summary_output_filename
    import json
    try:
        with open(json_summary_output_path, 'w', encoding='utf-8') as f_json:
            json.dump(json_summary_data, f_json, indent=2)
        print(f"[SUCCESS] Analysis JSON summary saved to {json_summary_output_path}")
    except Exception as e:
        print(f"Error writing JSON summary to {json_summary_output_path}: {e}")
    # ... existing code ... 
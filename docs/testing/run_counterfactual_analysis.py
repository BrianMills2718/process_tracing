import argparse
import json
from pathlib import Path
from core.counterfactual_analyzer import explore_counterfactual_with_llm

def main():
    parser = argparse.ArgumentParser(description="Run LLM-powered counterfactual analysis on a process tracing graph.")
    parser.add_argument('--graph_json', required=True, help='Path to the case graph JSON file')
    parser.add_argument('--premise', required=True, help='Counterfactual premise (e.g., "What if Event X did not happen?")')
    parser.add_argument('--outcome_id', required=True, help='Node ID of the key outcome event')
    parser.add_argument('--output', help='Optional path to save the LLM analysis result')
    args = parser.parse_args()

    with open(args.graph_json, 'r', encoding='utf-8') as f:
        graph_data_json_str = f.read()

    result = explore_counterfactual_with_llm(
        graph_data_json_str=graph_data_json_str,
        counterfactual_premise_text=args.premise,
        key_outcome_node_id=args.outcome_id
    )

    print("\n--- LLM Counterfactual Analysis Result ---\n")
    print(result)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n[INFO] Result saved to {args.output}")

if __name__ == "__main__":
    main() 
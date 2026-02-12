import pytest
import json
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_trace_advanced import execute_single_case_processing

def test_evidence_analysis_gets_source_context():
    """FAILING TEST: Evidence analysis should have access to original text"""
    
    # Test with a simple historical text
    test_text = """The French and Indian War ended in 1763 with Britain's victory over France. 
    To pay for the war debt, Britain imposed new taxes on the American colonies including the Stamp Act.
    The colonists protested that taxation without representation violated their rights as Englishmen.
    This led to increasing tensions and eventually the Boston Tea Party in 1773."""
    
    # Save test text to file
    test_file = "test_input.txt"
    with open(test_file, "w") as f:
        f.write(test_text)
    
    try:
        # Extract graph and analyze
        analysis_file = execute_single_case_processing(
            case_file_path_str=test_file,
            output_dir_for_case_str="test_output",
            project_name_str="test_evidence"
        )
        
        # Load the analysis results
        if analysis_file and os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            # Check that hypotheses evaluation exists and has meaningful data
            hypotheses = analysis.get('hypotheses_evaluation', [])
            
            if len(hypotheses) > 0:
                for hypothesis in hypotheses:
                    evidence_assessments = hypothesis.get('evidence_assessments', [])
                    
                    for evidence in evidence_assessments:
                        # THESE WILL FAIL NOW - evidence should have meaningful probative values
                        probative_value = evidence.get('probative_value', 0.0)
                        assert probative_value > 0.0, f"Evidence has zero probative value: {evidence}"
                        
                        # Evidence should have source quotes
                        source_quote = evidence.get('source_text_quote', '')
                        assert len(source_quote) > 0, f"Evidence missing source quote: {evidence}"
                        
                        # Evidence should have meaningful reasoning
                        reasoning = evidence.get('reasoning_for_type', '')
                        assert len(reasoning) > 20, f"Evidence reasoning too short: {reasoning}"
            else:
                # If no hypotheses found, that's also a problem we need to address
                pytest.skip("No hypotheses found in analysis - extraction issue")
                
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        # Clean up test output directory
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")

def test_extraction_produces_meaningful_descriptions():
    """FAILING TEST: Descriptions should come from source text"""
    
    test_text = """The Stamp Act of 1765 required colonists to pay taxes on printed materials.
    The Boston Massacre occurred in 1770 when British soldiers fired on colonial protesters.
    The Boston Tea Party in 1773 was a protest against British tea taxes."""
    
    test_file = "test_descriptions.txt"
    with open(test_file, "w") as f:
        f.write(test_text)
    
    try:
        analysis_file = execute_single_case_processing(
            case_file_path_str=test_file,
            output_dir_for_case_str="test_output_desc",
            project_name_str="test_descriptions"
        )
        
        # Find the graph file - it should be in the same directory
        if analysis_file and os.path.exists(analysis_file):
            # Graph file name pattern: project_timestamp_graph.json
            graph_file = analysis_file.replace('_analysis_summary_', '_').replace('.json', '_graph.json')
            # Try different patterns if needed
            import glob
            graph_pattern = os.path.join("test_output_desc", "*_graph.json")
            graph_files = glob.glob(graph_pattern)
            if graph_files:
                graph_file = graph_files[0]
        
        if graph_file and os.path.exists(graph_file):
            with open(graph_file, 'r') as f:
                graph = json.load(f)
            
            events = [n for n in graph['nodes'] if n['type'] == 'Event']
            assert len(events) >= 3, f"Should extract minimum 3 events, got {len(events)}"
            
            for event in events:
                desc = event['properties'].get('description', '')
                
                # THESE WILL FAIL NOW:
                assert desc != "N/A", f"Event {event['id']} has placeholder description"
                assert "Description_Not_Found" not in desc, f"Event {event['id']} has broken description"
                assert len(desc) >= 20, f"Event {event['id']} description too short: '{desc}'"
                
                # Check description contains actual historical content
                historical_terms = ['Stamp Act', 'Boston', 'colonial', 'British', 'tax', 'protest']
                has_historical_content = any(term in desc for term in historical_terms)
                assert has_historical_content, f"Event {event['id']} description lacks historical content: '{desc}'"
        else:
            pytest.fail("Graph file not created or not found")
            
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        import shutil
        if os.path.exists("test_output_desc"):
            shutil.rmtree("test_output_desc")

if __name__ == "__main__":
    # Run the tests to see current failures
    pytest.main([__file__, "-v"])
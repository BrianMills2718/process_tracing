"""
Structured Process Tracing Extractor
Modern approach using LiteLLM + Pydantic schemas instead of prompt engineering
Replaces manual JSON parsing with schema-enforced extraction
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import our schemas and UniversalLLM
from core.structured_schema import (
    ProcessTracingGraph, 
    ProcessTracingNode, 
    ProcessTracingEdge,
    StructuredExtractionResult,
    ExtractionMetadata
)
import litellm

# Van Evera Process Tracing Ontology-Aware Extraction Prompt
STRUCTURED_EXTRACTION_PROMPT = """Extract causal relationships from this text using Van Evera process tracing methodology with the following ontology:

## NODE TYPES (8 types):

**Event**: Specific occurrences in time/space
- Properties: description (required), timestamp, location, certainty (0.0-1.0), type (triggering/intermediate/outcome/unspecified), date, start_date, end_date, is_point_in_time

**Hypothesis**: Testable causal claims or theoretical propositions  
- Properties: description (required), prior_probability (0.0-1.0), posterior_probability (0.0-1.0), status (active/supported/partially_supported/refuted/undetermined)

**Evidence**: Facts that support/refute claims or confirm/deny events
- Properties: description (required), type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/bayesian/general), certainty (0.0-1.0), source, credibility (0.0-1.0)

**Causal_Mechanism**: Step-by-step explanations of HOW causation works
- Properties: description (required), confidence (0.0-1.0), status (hypothetical/supported/refuted/unspecified), level_of_detail

**Alternative_Explanation**: Competing theories or rival explanations
- Properties: description (required), probability (0.0-1.0), status (active/eliminated/supported/undetermined), key_predictions

**Actor**: People, groups, entities with agency and intentions
- Properties: name (required), role, intentions, beliefs, constraints, capabilities, credibility (0.0-1.0)

**Condition**: Enabling/constraining factors that affect other entities
- Properties: description (required), type (background/enabling/constraining/scope), necessity (0.0-1.0), temporal_scope, spatial_scope, certainty (0.0-1.0)

**Data_Source**: Origins of evidence (documents, witnesses, records)
- Properties: type (interview/document/observation/artifact/general), credibility (0.0-1.0), bias_risk (0.0-1.0)

## EDGE TYPES (21 types with domain→range constraints):

**causes**: Event → Event (causal chains)
- Properties: certainty (0.0-1.0), mechanism_id, type (direct/indirect)

**part_of_mechanism**: Event → Causal_Mechanism (mechanism components)
- Properties: role, sequence_position, necessity (0.0-1.0)

**tests_hypothesis**: Evidence,Event → Hypothesis (hypothesis testing)
- Properties: probative_value (0.0-1.0), test_result (passed/failed/ambiguous), diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general)

**tests_mechanism**: Evidence,Event → Causal_Mechanism (mechanism testing)
- Properties: probative_value (0.0-1.0), test_result (passed/failed/ambiguous), diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general)

**supports**: Evidence,Event → Hypothesis,Event,Causal_Mechanism,Actor (positive evidential support)
- Use when: evidence/events strengthen or bolster claims, when text uses "supports", "demonstrates", "shows", "indicates", "provides evidence for", "strengthens the case that"
- Properties: probative_value (0.0-1.0), certainty (0.0-1.0), diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general), target_type (event_occurrence/causal_relationship/mechanism_operation/general), source_text_quote

**refutes**: Evidence,Event → Hypothesis,Event,Causal_Mechanism (refutation relationships)
- Properties: probative_value (0.0-1.0), certainty (0.0-1.0), diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general), target_type (event_occurrence/causal_relationship/mechanism_operation/general), source_text_quote

**confirms_occurrence**: Evidence → Event (event confirmation)
- Properties: certainty (0.0-1.0), diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general), source_text_quote

**disproves_occurrence**: Evidence → Event (event disproof)
- Properties: certainty (0.0-1.0), diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general), source_text_quote

**explains_mechanism**: Hypothesis → Causal_Mechanism (mechanism explanation)
- Properties: certainty (0.0-1.0), type_of_claim (existence/operation/significance/component_role/operational_details/unspecified)

**provides_evidence_for**: Event,Evidence → Hypothesis,Causal_Mechanism,Actor,Alternative_Explanation (evidence provision)
- Properties: probative_value (0.0-1.0), reasoning, diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general)

**supports_alternative**: Evidence → Alternative_Explanation (alternative support)
- Properties: probative_value (0.0-1.0), certainty (0.0-1.0)

**refutes_alternative**: Evidence → Alternative_Explanation (alternative refutation)
- Properties: probative_value (0.0-1.0), certainty (0.0-1.0)

**tests_alternative**: Evidence,Event → Alternative_Explanation (alternative testing)
- Properties: probative_value (0.0-1.0), diagnostic_type (hoop/smoking_gun/straw_in_the_wind/doubly_decisive/general), test_result (supports/refutes/inconclusive)

**initiates**: Actor → Event (actor agency)
- Properties: certainty (0.0-1.0), intention, agency

**enables**: Condition → Event,Causal_Mechanism,Hypothesis (enabling conditions)
- Properties: necessity (0.0-1.0), certainty (0.0-1.0), type

**constrains**: Condition → Event,Causal_Mechanism,Actor (constraining conditions)
- Properties: certainty (0.0-1.0), type

**provides_evidence**: Data_Source → Evidence (evidence sourcing)
- Properties: credibility (0.0-1.0), bias_risk (0.0-1.0), certainty (0.0-1.0)

**updates_probability**: Evidence → Hypothesis (Bayesian updating)
- Properties: prior_probability (0.0-1.0), posterior_probability (0.0-1.0), Bayes_factor

**contradicts**: Evidence → Evidence (evidence contradiction)
- Properties: certainty (0.0-1.0), reason

**weighs_evidence**: Evidence → Evidence,Hypothesis,Causal_Mechanism (evidence quality affecting theoretical credibility)
- Use when: evidence quality impacts hypothesis/mechanism credibility ("primary sources strengthen the hypothesis", "archaeological evidence validates the mechanism", "documentary evidence tests mechanism operation", "source reliability affects theory"), also evidence-to-evidence quality comparisons ("more reliable than", "primary vs secondary source", "cross-validated", "independent confirmation")
- Properties: comparison_strength (0.0-1.0), comparison_type (stronger_than/weaker_than/equivalent_to/complements/contradicts), reasoning

**infers**: Evidence,Event,Inference_Rule → Hypothesis,Causal_Mechanism (inferential reasoning from evidence/events to theory)
- Properties: certainty (0.0-1.0), logic_type

## VAN EVERA DIAGNOSTIC TYPES:
- **hoop**: Necessary but not sufficient (if absent, hypothesis fails)
- **smoking_gun**: Sufficient but not necessary (if present, hypothesis confirmed)  
- **doubly_decisive**: Both necessary and sufficient (definitive test)
- **straw_in_the_wind**: Neither necessary nor sufficient (weak evidence)

## EXTRACTION PRIORITIES:
Focus especially on these methodologically critical edge types:
- **infers**: Evidence/Events that allow logical inference to hypotheses/mechanisms (Evidence→Hypothesis, Event→Mechanism)
- **confirms_occurrence**: Evidence that establishes events actually happened (Evidence→Event)
- **refutes**: Evidence/Events that falsify hypotheses/mechanisms (Evidence→Hypothesis)
- **tests_alternative**: Evidence/Events that test competing explanations (Evidence→Alternative_Explanation)
- **weighs_evidence**: Evidence quality affecting theoretical credibility - prioritize Evidence→Hypothesis and Evidence→Mechanism relationships where source reliability/strength impacts theory validity, not just Evidence→Evidence comparisons

Extract comprehensive causal networks following these constraints. Pay special attention to domain→range validity and diagnostic properties.

## REQUIRED JSON OUTPUT STRUCTURE:

You MUST return JSON with exactly this structure (use "type" not "node_type" or "edge_type"):

```json
{{
    "nodes": [
        {{
            "id": "unique_id",
            "type": "Event|Hypothesis|Evidence|Causal_Mechanism|Alternative_Explanation|Actor|Condition|Data_Source",
            "properties": {{
                "description": "required description"
            }}
        }}
    ],
    "edges": [
        {{
            "id": "unique_edge_id", 
            "source_id": "source_node_id",
            "target_id": "target_node_id",
            "type": "causes|supports|refutes|tests_hypothesis|etc",
            "properties": {{
                
            }}
        }}
    ]
}}
```

Text to analyze:
{text}"""

class StructuredProcessTracingExtractor:
    """
    Clean extraction using structured output instead of prompt engineering
    Eliminates manual JSON parsing and validation issues
    """
    
    def __init__(self, model_name: str = "gpt-5-mini"):
        """Initialize with direct LiteLLM for structured extraction"""
        self.model_name = model_name
        
        # Get API key - prioritize OpenAI for GPT-5-mini
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found in environment variables. Set OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY")
        
    def extract_graph(self, text: str, project_name: str = "default") -> StructuredExtractionResult:
        """
        Extract process tracing graph using structured output
        
        Args:
            text: Input text to analyze
            project_name: Project identifier for metadata
            
        Returns:
            StructuredExtractionResult with graph and metadata
        """
        print(f"Extracting with structured output (model: {self.model_name})")
        
        # Format the prompt with input text
        prompt = STRUCTURED_EXTRACTION_PROMPT.format(text=text)
        
        try:
            # Use LiteLLM structured output with Pydantic schema
            start_time = datetime.now()
            graph = self._extract_with_structured_output(prompt)
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            print(f"Extraction completed in {extraction_time:.2f}s")
            
            # Create metadata
            metadata = ExtractionMetadata(
                extraction_method="structured_output",
                model_used=self.model_name,
                prompt_length=len(prompt),
                text_length=len(text)
            )
            
            # Create complete result
            result = StructuredExtractionResult(
                graph=graph,
                metadata=metadata
            )
            
            # Print summary
            result.print_summary()
            
            return result
            
        except Exception as e:
            print(f"ERROR: Structured extraction failed: {str(e)}")
            print("FAIL FAST: Fix the underlying issue - no empty graph fallbacks")
            raise e
    
    def _extract_with_structured_output(self, prompt: str) -> ProcessTracingGraph:
        """Use LiteLLM structured output with Pydantic schema - fail fast if it doesn't work"""
        try:
            # Use direct LiteLLM completion with router parameters for GPT-5-mini compatibility
            # PHASE 17B: Apply exact router parameters discovered in 17A
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You must respond with valid JSON following the specified schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                api_key=self.api_key,
                max_completion_tokens=16384,
                use_in_pass_through=False,
                use_litellm_proxy=False,
                merge_reasoning_content_in_choices=False
            )
            
            # Get the JSON content and parse it with our Pydantic model
            json_content = response.choices[0].message.content
            if json_content:
                # Clean and parse the JSON response
                cleaned_response = self._clean_json_response(json_content)
                return ProcessTracingGraph.model_validate_json(cleaned_response)
            else:
                raise ValueError("No content returned from LiteLLM")
                
        except Exception as e:
            print(f"ERROR: LiteLLM structured output failed: {str(e)}")
            print("FAIL FAST: No fallbacks, no degradation - fix the LiteLLM configuration")
            raise e
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from control characters and formatting issues"""
        import re
        
        # Remove control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)
        
        # Remove any markdown code block markers
        cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Strip whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def save_results(self, result: StructuredExtractionResult, output_path: str) -> None:
        """Save extraction results to JSON file"""
        try:
            # Convert to dict for JSON serialization
            output_data = {
                "nodes": [node.model_dump() for node in result.graph.nodes],
                "edges": [edge.model_dump() for edge in result.graph.edges],
                "metadata": result.metadata.model_dump()
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Results saved to: {output_path}")
            
        except Exception as e:
            print(f"ERROR: Failed to save results: {str(e)}")
    
    def compare_with_current_approach(self, text: str, current_result_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare structured output approach with current manual parsing
        Returns comparison metrics
        """
        print("🔍 Comparing structured vs current approach...")
        
        # Extract with structured approach
        structured_result = self.extract_graph(text)
        structured_stats = structured_result.graph.get_coverage_stats()
        
        comparison = {
            "structured_approach": {
                "node_coverage": structured_stats["nodes"]["coverage_percent"],
                "edge_coverage": structured_stats["edges"]["coverage_percent"],
                "node_types": structured_stats["nodes"]["types"],
                "edge_types": structured_stats["edges"]["types"],
                "total_nodes": structured_stats["nodes"]["total"],
                "total_edges": structured_stats["edges"]["total"]
            }
        }
        
        # Compare with current approach if available
        if current_result_path and os.path.exists(current_result_path):
            try:
                with open(current_result_path, 'r') as f:
                    current_data = json.load(f)
                
                current_node_types = list(set(node.get('type') for node in current_data.get('nodes', [])))
                current_edge_types = list(set(edge.get('type') for edge in current_data.get('edges', [])))
                
                comparison["current_approach"] = {
                    "node_coverage": (len(current_node_types) / 8) * 100,
                    "edge_coverage": (len(current_edge_types) / 21) * 100,
                    "node_types": sorted(current_node_types),
                    "edge_types": sorted(current_edge_types),
                    "total_nodes": len(current_data.get('nodes', [])),
                    "total_edges": len(current_data.get('edges', []))
                }
                
                # Calculate improvements
                comparison["improvement"] = {
                    "node_coverage_gain": comparison["structured_approach"]["node_coverage"] - comparison["current_approach"]["node_coverage"],
                    "edge_coverage_gain": comparison["structured_approach"]["edge_coverage"] - comparison["current_approach"]["edge_coverage"],
                    "new_node_types": list(set(structured_stats["nodes"]["types"]) - set(current_node_types)),
                    "new_edge_types": list(set(structured_stats["edges"]["types"]) - set(current_edge_types))
                }
                
            except Exception as e:
                print(f"⚠️ Could not load current results for comparison: {str(e)}")
        
        return comparison

def extract_with_structured_output(text: str, project_name: str = "test", 
                                 output_dir: str = "output_data") -> StructuredExtractionResult:
    """
    Convenience function for structured extraction
    
    Args:
        text: Input text to analyze
        project_name: Project name for output files
        output_dir: Directory for output files
        
    Returns:
        StructuredExtractionResult
    """
    # Create extractor
    extractor = StructuredProcessTracingExtractor()
    
    # Extract graph
    result = extractor.extract_graph(text, project_name)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, project_name, f"{project_name}_{timestamp}_structured_graph.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    extractor.save_results(result, output_path)
    
    return result

if __name__ == "__main__":
    # Test the structured extractor
    test_text = """
    The Tea Act of 1773 led to the Boston Tea Party. Colonists, motivated by 
    taxation without representation, organized resistance. Samuel Adams coordinated 
    the Sons of Liberty who dumped tea into Boston Harbor. This event caused 
    King George III to pass the Intolerable Acts, which closed Boston Harbor.
    The escalating conflict eventually led to the American Revolution.
    """
    
    print("🧪 Testing structured extraction...")
    result = extract_with_structured_output(test_text, "test_structured")
    
    print("\n📊 Coverage Analysis:")
    stats = result.graph.get_coverage_stats()
    print(f"Node coverage: {stats['nodes']['coverage_percent']:.1f}% ({stats['nodes']['types_found']}/8)")
    print(f"Edge coverage: {stats['edges']['coverage_percent']:.1f}% ({stats['edges']['types_found']}/21)")
    print(f"Node types: {stats['nodes']['types']}")
    print(f"Edge types: {stats['edges']['types']}")
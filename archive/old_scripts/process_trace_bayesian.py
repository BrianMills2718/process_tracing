#!/usr/bin/env python3
"""
Bayesian Process Tracing Entry Point

Enhanced process tracing with integrated Bayesian confidence assessment,
uncertainty analysis, and probabilistic reporting. Builds upon the existing
process_trace_advanced.py pipeline while adding Phase 6 Bayesian capabilities.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Load environment configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check dependencies
try:
    import google.generativeai as genai
except ImportError:
    print("[ERROR] google-generativeai is not installed. Please install it.")
    sys.exit(1)

# Core process tracing components  
from core.extract import query_gemini, parse_json
from core.analyze import analyze_graph
from core.enhance_evidence import refine_evidence_assessment_with_llm

# Phase 6 Bayesian components
from core.bayesian_integration import (
    BayesianProcessTracingIntegrator,
    integrate_bayesian_analysis
)
from core.bayesian_reporting import BayesianReportConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gemini_api():
    """Setup and validate Gemini API configuration."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not found in environment variables.")
        print("Please set GOOGLE_API_KEY in your .env file or environment.")
        sys.exit(1)
    
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to configure Gemini API: {e}")
        sys.exit(1)


def load_text_file(filepath: str) -> str:
    """Load text content from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Loaded text file: {filepath} ({len(content)} characters)")
        return content
    except Exception as e:
        logger.error(f"Failed to load text file {filepath}: {e}")
        raise


def run_extraction_phase(text_content: str, 
                        output_dir: str,
                        project_name: str) -> Dict[str, Any]:
    """Run the graph extraction phase."""
    logger.info("Starting graph extraction phase")
    
    try:
        # Extract initial graph from text using Gemini
        raw_response = query_gemini(text_content)
        
        # Parse the JSON response
        if isinstance(raw_response, dict):
            graph_data = raw_response
        else:
            graph_data = parse_json(raw_response)
        
        # Save extracted graph
        graph_filename = f"{project_name}_extracted_graph.json"
        graph_filepath = Path(output_dir) / graph_filename
        
        with open(graph_filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted graph to: {graph_filepath}")
        return graph_data
        
    except Exception as e:
        logger.error(f"Extraction phase failed: {e}")
        raise


def run_analysis_phase(graph_data: Dict[str, Any],
                      output_dir: str,
                      project_name: str) -> Dict[str, Any]:
    """Run the traditional process tracing analysis phase."""
    logger.info("Starting traditional analysis phase")
    
    try:
        # Convert graph data to NetworkX graph for analysis
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.get('nodes', []):
            G.add_node(node['id'], **node)
        
        # Add edges
        for edge in graph_data.get('edges', []):
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Run analysis
        analysis_results = analyze_graph(G)
        
        # Save analysis results
        analysis_filename = f"{project_name}_analysis.json"
        analysis_filepath = Path(output_dir) / analysis_filename
        
        with open(analysis_filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved analysis results to: {analysis_filepath}")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Analysis phase failed: {e}")
        raise


def run_evidence_assessment_phase(text_content: str,
                                 graph_data: Dict[str, Any],
                                 analysis_results: Dict[str, Any],
                                 output_dir: str,
                                 project_name: str) -> List[Any]:
    """Run the Van Evera evidence assessment phase."""
    logger.info("Starting Van Evera evidence assessment phase")
    
    try:
        # Extract evidence from analysis results
        evidence_pieces = []
        
        # Get evidence from graph nodes
        for node in graph_data.get('nodes', []):
            if node.get('type') == 'evidence':
                evidence_pieces.append({
                    'id': node['id'],
                    'description': node.get('description', ''),
                    'text': node.get('label', '')
                })
        
        # Enhance evidence with Van Evera assessment
        enhanced_evidence = []
        for evidence in evidence_pieces:
            try:
                assessment = refine_evidence_assessment_with_llm(
                    evidence['description'],
                    text_content,
                    context_info=f"Analysis context: {analysis_results.get('narrative_summary', '')}"
                )
                enhanced_evidence.append(assessment)
            except Exception as e:
                logger.warning(f"Failed to assess evidence {evidence['id']}: {e}")
                continue
        
        # Save evidence assessments
        evidence_filename = f"{project_name}_evidence_assessments.json"
        evidence_filepath = Path(output_dir) / evidence_filename
        
        # Convert assessments to serializable format
        serializable_evidence = []
        for assessment in enhanced_evidence:
            if hasattr(assessment, 'dict'):
                serializable_evidence.append(assessment.dict())
            else:
                serializable_evidence.append(str(assessment))
        
        with open(evidence_filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_evidence, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evidence assessments to: {evidence_filepath}")
        logger.info(f"Assessed {len(enhanced_evidence)} pieces of evidence")
        
        return enhanced_evidence
        
    except Exception as e:
        logger.error(f"Evidence assessment phase failed: {e}")
        raise


def run_bayesian_analysis_phase(analysis_results: Dict[str, Any],
                               evidence_assessments: List[Any],
                               output_dir: str,
                               project_name: str,
                               config: Optional[BayesianReportConfig] = None) -> Dict[str, Any]:
    """Run the Bayesian confidence assessment phase."""
    logger.info("Starting Bayesian analysis phase")
    
    try:
        # Create Bayesian configuration if not provided
        if config is None:
            config = BayesianReportConfig(
                include_visualizations=True,
                uncertainty_simulations=1000
            )
        
        # Integrate Bayesian analysis with traditional results
        enhanced_analysis = integrate_bayesian_analysis(
            analysis_results,
            evidence_assessments,
            output_dir=output_dir,
            config=config
        )
        
        # Save enhanced analysis
        enhanced_filename = f"{project_name}_bayesian_analysis.json"
        enhanced_filepath = Path(output_dir) / enhanced_filename
        
        with open(enhanced_filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved Bayesian analysis to: {enhanced_filepath}")
        
        # Extract and log key metrics
        bayesian_section = enhanced_analysis.get('bayesian_analysis', {})
        if bayesian_section:
            confidence_assessments = bayesian_section.get('confidence_assessments', {})
            uncertainty_analysis = bayesian_section.get('uncertainty_analysis', {})
            
            logger.info("Bayesian Analysis Summary:")
            for hyp_id, assessment in confidence_assessments.items():
                confidence = assessment.get('overall_confidence', 0)
                level = assessment.get('confidence_level', 'unknown')
                logger.info(f"  {hyp_id}: {confidence:.1%} confidence ({level})")
            
            for hyp_id, uncertainty in uncertainty_analysis.items():
                std = uncertainty.get('confidence_std', 0)
                robustness = uncertainty.get('robustness_score', 0)
                logger.info(f"  {hyp_id}: ±{std:.1%} uncertainty, {robustness:.1%} robustness")
        
        return enhanced_analysis
        
    except Exception as e:
        logger.error(f"Bayesian analysis phase failed: {e}")
        # Return original analysis if Bayesian enhancement fails
        logger.warning("Returning original analysis without Bayesian enhancement")
        return analysis_results


def create_output_directory(base_output_dir: str, project_name: str) -> str:
    """Create output directory for the project."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"{project_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return str(output_dir)


def run_complete_bayesian_analysis(input_file: str,
                                 output_dir: str,
                                 project_name: Optional[str] = None,
                                 config: Optional[BayesianReportConfig] = None) -> Dict[str, Any]:
    """
    Run complete Bayesian process tracing analysis pipeline.
    
    Args:
        input_file: Path to text file for analysis
        output_dir: Base output directory
        project_name: Optional project name (inferred from filename if not provided)
        config: Optional Bayesian report configuration
        
    Returns:
        Complete analysis results with Bayesian enhancement
    """
    # Setup project name
    if project_name is None:
        project_name = Path(input_file).stem
    
    # Create output directory
    output_path = create_output_directory(output_dir, project_name)
    
    try:
        # Phase 1: Load text content
        logger.info(f"Phase 1: Loading text content from {input_file}")
        text_content = load_text_file(input_file)
        
        # Phase 2: Extract graph
        logger.info("Phase 2: Extracting causal graph")
        graph_data = run_extraction_phase(text_content, output_path, project_name)
        
        # Phase 3: Traditional analysis
        logger.info("Phase 3: Running traditional process tracing analysis")
        analysis_results = run_analysis_phase(graph_data, output_path, project_name)
        
        # Phase 4: Van Evera evidence assessment
        logger.info("Phase 4: Running Van Evera evidence assessment")
        evidence_assessments = run_evidence_assessment_phase(
            text_content, graph_data, analysis_results, output_path, project_name
        )
        
        # Phase 5: Bayesian analysis integration
        logger.info("Phase 5: Running Bayesian confidence assessment")
        final_results = run_bayesian_analysis_phase(
            analysis_results, evidence_assessments, output_path, project_name, config
        )
        
        logger.info("Complete Bayesian process tracing analysis finished successfully")
        logger.info(f"Results saved to: {output_path}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Complete analysis failed: {e}")
        raise


def main():
    """Main entry point for Bayesian process tracing."""
    parser = argparse.ArgumentParser(
        description="Bayesian Process Tracing - Enhanced probabilistic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file with Bayesian enhancement
  python process_trace_bayesian.py input.txt --output results/

  # Custom project name and configuration  
  python process_trace_bayesian.py input.txt --output results/ --project my_analysis --simulations 2000

  # High-quality analysis with visualizations
  python process_trace_bayesian.py input.txt --output results/ --visualizations --confidence-level 0.99
        """
    )
    
    parser.add_argument('input_file', help='Input text file for analysis')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    parser.add_argument('--project', '-p', help='Project name (default: input filename)')
    
    # Bayesian configuration options
    parser.add_argument('--simulations', type=int, default=1000, 
                       help='Number of Monte Carlo simulations (default: 1000)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                       help='Confidence level for intervals (default: 0.95)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Disable visualization generation')
    parser.add_argument('--no-uncertainty', action='store_true',
                       help='Skip uncertainty analysis')
    parser.add_argument('--no-sensitivity', action='store_true',
                       help='Skip sensitivity analysis')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal logging output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Setup API
    setup_gemini_api()
    
    # Create Bayesian configuration
    config = BayesianReportConfig(
        include_uncertainty_analysis=not args.no_uncertainty,
        include_sensitivity_analysis=not args.no_sensitivity,
        include_visualizations=not args.no_visualizations,
        uncertainty_simulations=args.simulations,
        confidence_level=args.confidence_level
    )
    
    try:
        # Run complete analysis
        logger.info("Starting Bayesian Process Tracing Analysis")
        logger.info(f"Input: {args.input_file}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Configuration: {args.simulations} simulations, {args.confidence_level:.1%} confidence level")
        
        results = run_complete_bayesian_analysis(
            args.input_file,
            args.output,
            args.project,
            config
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BAYESIAN PROCESS TRACING ANALYSIS COMPLETE")
        print("="*60)
        
        if 'bayesian_analysis' in results:
            bayesian_data = results['bayesian_analysis']
            confidence_data = bayesian_data.get('confidence_assessments', {})
            
            if confidence_data:
                print("\nConfidence Assessment Summary:")
                for hyp_id, assessment in confidence_data.items():
                    confidence = assessment.get('overall_confidence', 0)
                    level = assessment.get('confidence_level', 'unknown')
                    evidence_count = assessment.get('evidence_count', 0)
                    print(f"  {hyp_id}: {confidence:.1%} ({level}) - {evidence_count} evidence pieces")
            
            uncertainty_data = bayesian_data.get('uncertainty_analysis', {})
            if uncertainty_data:
                print("\nUncertainty Analysis Summary:")
                for hyp_id, uncertainty in uncertainty_data.items():
                    std = uncertainty.get('confidence_std', 0)
                    robustness = uncertainty.get('robustness_score', 0)
                    print(f"  {hyp_id}: ±{std:.1%} uncertainty, {robustness:.1%} robustness")
        
        print(f"\nResults saved to: {Path(args.output).absolute()}")
        print("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n[ERROR] Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
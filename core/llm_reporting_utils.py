import json
import logging
from typing import Dict, Any, Optional
from process_trace_advanced import query_llm
from .structured_models import NarrativeSummary

# Phase 6C Bayesian Integration
try:
    from .bayesian_reporting import BayesianReporter, BayesianReportConfig
    from .bayesian_models import BayesianHypothesisSpace
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logging.warning("Bayesian reporting components not available")

logger = logging.getLogger(__name__)

def generate_narrative_summary_with_llm(structured_data_dict, summary_focus_prompt):
    """
    Use an LLM to generate a concise analytical narrative summary for a given structured data dict and focus prompt.
    Uses structured output with Pydantic models for reliable formatting.
    
    Args:
        structured_data_dict (dict): The data to be summarized.
        summary_focus_prompt (str): Instruction for the summary focus.
    Returns:
        NarrativeSummary: Structured summary from LLM with key findings and confidence assessment.
    """
    data_str = json.dumps(structured_data_dict, indent=2)
    
    # Extract evidence counts for structured response
    supporting_count = 0
    refuting_count = 0
    
    if isinstance(structured_data_dict, dict):
        # Try to extract evidence counts from various data structures
        if 'supporting_evidence' in structured_data_dict:
            supporting_count = len(structured_data_dict.get('supporting_evidence', []))
        if 'refuting_evidence' in structured_data_dict:
            refuting_count = len(structured_data_dict.get('refuting_evidence', []))
        
        # Alternative structure for evidence analysis
        if 'evidence_analysis' in structured_data_dict:
            for hyp_data in structured_data_dict['evidence_analysis'].values():
                supporting_count += len(hyp_data.get('supporting_evidence', []))
                refuting_count += len(hyp_data.get('refuting_evidence', []))
    
    prompt = f"""
You are an expert academic writer. Based on the following structured data:

{data_str}

Please provide a comprehensive analytical summary focusing on: {summary_focus_prompt}

Your response should include:
1. A concise analytical narrative (2-5 sentences) that is objective and directly supported by the data
2. Key findings extracted from the analysis (3-5 main points)
3. Your confidence level in the conclusions (high/medium/low)
4. Accurate counts of supporting and refuting evidence

Use clear, analytical language appropriate for academic writing.
"""
    
    # Use structured output with NarrativeSummary model
    llm_response = query_llm(
        text_content="",  # No main text, just use prompt
        schema=NarrativeSummary,
        system_instruction_text=prompt,
        use_structured_output=True
    )
    
    return llm_response

def generate_simple_narrative_summary(structured_data_dict, summary_focus_prompt):
    """
    Generate a simple text summary (backward compatibility function).
    
    Args:
        structured_data_dict (dict): The data to be summarized.
        summary_focus_prompt (str): Instruction for the summary focus.
    Returns:
        str: Simple text summary.
    """
    structured_summary = generate_narrative_summary_with_llm(structured_data_dict, summary_focus_prompt)
    return structured_summary.summary_text 


def generate_enhanced_report_with_bayesian(analysis_results: Dict[str, Any], 
                                         output_dir: str,
                                         project_name: str,
                                         bayesian_config: Optional[BayesianReportConfig] = None) -> Dict[str, Any]:
    """
    Generate enhanced HTML report with Bayesian analysis integration.
    
    This function integrates traditional process tracing reports with Bayesian
    confidence assessment, uncertainty analysis, and probabilistic reporting.
    
    Args:
        analysis_results: Traditional process tracing analysis results
        output_dir: Directory for output files
        project_name: Project name for file naming
        bayesian_config: Optional Bayesian reporting configuration
        
    Returns:
        Enhanced analysis results with integrated Bayesian reporting
    """
    if not BAYESIAN_AVAILABLE:
        logger.warning("Bayesian components not available, returning traditional report")
        return analysis_results
    
    try:
        # Check if analysis already contains Bayesian data
        bayesian_section = analysis_results.get('bayesian_analysis', {})
        
        if not bayesian_section:
            logger.info("No Bayesian analysis found, generating traditional report only")
            return analysis_results
        
        # Create Bayesian reporter
        config = bayesian_config or BayesianReportConfig(
            include_visualizations=True,
            uncertainty_simulations=1000
        )
        reporter = BayesianReporter(config)
        
        # Extract Bayesian report from analysis results
        bayesian_report = bayesian_section.get('bayesian_report', {})
        
        if bayesian_report:
            # Enhance traditional report with Bayesian sections
            enhanced_html = _integrate_bayesian_into_traditional_report(
                analysis_results, bayesian_report, project_name
            )
            
            # Save enhanced HTML report
            from pathlib import Path
            from datetime import datetime
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"{project_name}_enhanced_bayesian_report_{timestamp}.html"
            html_filepath = output_path / html_filename
            
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(enhanced_html)
            
            logger.info(f"Generated enhanced Bayesian report: {html_filepath}")
            
            # Add report path to analysis results
            analysis_results['enhanced_report_path'] = str(html_filepath)
            analysis_results['enhanced_report_type'] = 'bayesian_integrated'
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Failed to generate enhanced Bayesian report: {e}")
        return analysis_results


def _integrate_bayesian_into_traditional_report(analysis_results: Dict[str, Any],
                                              bayesian_report: Dict[str, Any],
                                              project_name: str) -> str:
    """
    Integrate Bayesian report sections into traditional HTML report.
    
    Creates a unified HTML report that combines traditional process tracing
    analysis with Bayesian confidence assessment and uncertainty analysis.
    """
    # Extract key components
    bayesian_analysis = analysis_results.get('bayesian_analysis', {})
    confidence_assessments = bayesian_analysis.get('confidence_assessments', {})
    uncertainty_analysis = bayesian_analysis.get('uncertainty_analysis', {})
    
    # Get traditional summary information
    narrative_summary = analysis_results.get('narrative_summary', 'No summary available')
    evidence_assessment = analysis_results.get('evidence_assessment', {})
    
    # Get Bayesian HTML content
    bayesian_html_content = bayesian_report.get('html_content', '')
    bayesian_sections = bayesian_report.get('sections', [])
    
    # Create integrated HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Process Tracing Analysis - {project_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8fafc;
            color: #1e293b;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }}
        .header p {{
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            padding: 2rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            color: #1e40af;
            margin-top: 0;
            font-size: 1.8rem;
            font-weight: 500;
        }}
        .section h3 {{
            color: #3730a3;
            margin-top: 1.5rem;
            font-size: 1.3rem;
        }}
        .confidence-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
            margin-left: 0.5rem;
        }}
        .confidence-high {{ background-color: #dcfce7; color: #166534; }}
        .confidence-medium {{ background-color: #fef3c7; color: #92400e; }}
        .confidence-low {{ background-color: #fee2e2; color: #991b1b; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric-card {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e40af;
        }}
        .metric-label {{
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}
        .traditional-section {{
            background: #fafafa;
            border-left: 4px solid #10b981;
        }}
        .bayesian-section {{
            background: #f0f4ff;
            border-left: 4px solid #3b82f6;
        }}
        .summary-text {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 1rem 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Process Tracing Analysis</h1>
            <p>Integrated Traditional & Bayesian Analysis - {project_name}</p>
        </div>
        
        <div class="section traditional-section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-text">
                {narrative_summary}
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{evidence_assessment.get('supporting_evidence_count', 'N/A')}</div>
                    <div class="metric-label">Supporting Evidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{evidence_assessment.get('contradicting_evidence_count', 'N/A')}</div>
                    <div class="metric-label">Contradicting Evidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{evidence_assessment.get('evidence_quality', 'N/A')}</div>
                    <div class="metric-label">Evidence Quality</div>
                </div>
            </div>
        </div>
        
        <div class="section bayesian-section">
            <h2>🎯 Bayesian Confidence Assessment</h2>
            
            <h3>Hypothesis Confidence Levels</h3>
            <div class="metrics-grid">
    """
    
    # Add confidence assessments
    for hyp_id, assessment in confidence_assessments.items():
        confidence = assessment.get('overall_confidence', 0)
        confidence_pct = f"{confidence:.1%}"
        level = assessment.get('confidence_level', 'unknown')
        evidence_count = assessment.get('evidence_count', 0)
        
        # Determine confidence badge class
        if level == 'high':
            badge_class = 'confidence-high'
        elif level == 'medium':
            badge_class = 'confidence-medium'
        else:
            badge_class = 'confidence-low'
        
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-value">{confidence_pct}
                        <span class="confidence-badge {badge_class}">{level}</span>
                    </div>
                    <div class="metric-label">{hyp_id.replace('_', ' ').title()} ({evidence_count} evidence)</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    """
    
    # Add uncertainty analysis if available
    if uncertainty_analysis:
        html_content += """
        <div class="section bayesian-section">
            <h2>📈 Uncertainty Analysis</h2>
            <div class="metrics-grid">
        """
        
        for hyp_id, uncertainty in uncertainty_analysis.items():
            std = uncertainty.get('confidence_std', 0)
            robustness = uncertainty.get('robustness_score', 0)
            
            html_content += f"""
                <div class="metric-card">
                    <div class="metric-value">±{std:.1%}</div>
                    <div class="metric-label">{hyp_id.replace('_', ' ').title()} Uncertainty</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{robustness:.1%}</div>
                    <div class="metric-label">{hyp_id.replace('_', ' ').title()} Robustness</div>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    # Embed full Bayesian report if available
    if bayesian_html_content:
        # Extract just the body content from Bayesian report
        if '<body>' in bayesian_html_content and '</body>' in bayesian_html_content:
            start = bayesian_html_content.find('<body>') + 6
            end = bayesian_html_content.find('</body>')
            bayesian_body = bayesian_html_content[start:end]
        else:
            bayesian_body = bayesian_html_content
        
        html_content += f"""
        <div class="section bayesian-section">
            <h2>📋 Detailed Bayesian Analysis</h2>
            {bayesian_body}
        </div>
        """
    
    # Close HTML
    html_content += """
    </div>
</body>
</html>
    """
    
    return html_content


def check_bayesian_availability() -> bool:
    """
    Check if Bayesian reporting components are available.
    
    Returns:
        bool: True if Bayesian components are available, False otherwise
    """
    return BAYESIAN_AVAILABLE


def get_bayesian_report_metadata(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Bayesian report metadata from analysis results.
    
    Args:
        analysis_results: Analysis results containing Bayesian data
        
    Returns:
        Metadata dictionary with Bayesian analysis summary
    """
    if not BAYESIAN_AVAILABLE:
        return {'bayesian_available': False}
    
    bayesian_section = analysis_results.get('bayesian_analysis', {})
    
    if not bayesian_section:
        return {'bayesian_available': False, 'bayesian_analysis_present': False}
    
    confidence_assessments = bayesian_section.get('confidence_assessments', {})
    uncertainty_analysis = bayesian_section.get('uncertainty_analysis', {})
    
    metadata = {
        'bayesian_available': True,
        'bayesian_analysis_present': True,
        'timestamp': bayesian_section.get('timestamp'),
        'integration_method': bayesian_section.get('integration_method'),
        'hypothesis_count': len(confidence_assessments),
        'uncertainty_analysis_count': len(uncertainty_analysis)
    }
    
    # Add primary hypothesis data if available
    if 'primary_explanation' in confidence_assessments:
        primary = confidence_assessments['primary_explanation']
        metadata['primary_confidence'] = primary.get('overall_confidence', 0)
        metadata['primary_confidence_level'] = primary.get('confidence_level', 'unknown')
        metadata['primary_evidence_count'] = primary.get('evidence_count', 0)
    
    return metadata
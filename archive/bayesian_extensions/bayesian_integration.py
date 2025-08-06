"""
Bayesian Integration Bridge for Process Tracing Analysis.

Integrates Phase 6C Bayesian confidence assessment, uncertainty analysis, and 
reporting capabilities with the existing process tracing analysis pipeline.
Provides seamless bridge between traditional Van Evera analysis and advanced
Bayesian probabilistic assessment.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

# Phase 6C Bayesian Components
from .confidence_calculator import CausalConfidenceCalculator, ConfidenceAssessment
from .uncertainty_analysis import UncertaintyAnalyzer, UncertaintyAnalysisResult
from .bayesian_reporting import BayesianReporter, BayesianReportConfig

# Phase 6B Van Evera Bayesian Integration
from .van_evera_bayesian import VanEveraBayesianBridge
from .evidence_weighting import EvidenceStrengthQuantifier

# Phase 6A Bayesian Models
from .bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace,
    HypothesisType, EvidenceType
)

# Existing Process Tracing Infrastructure
from .enhance_evidence import EvidenceAssessment
from .structured_models import VanEveraEvidenceType


logger = logging.getLogger(__name__)


class BayesianProcessTracingIntegrator:
    """
    Integrates Bayesian analysis capabilities with existing process tracing pipeline.
    
    Provides seamless bridge between Van Evera evidence assessment and Bayesian
    confidence quantification, enabling probabilistic analysis within the existing
    workflow without disrupting traditional process tracing functionality.
    """
    
    def __init__(self, config: Optional[BayesianReportConfig] = None):
        """Initialize the Bayesian integration bridge."""
        self.van_evera_bridge = VanEveraBayesianBridge()
        self.confidence_calculator = CausalConfidenceCalculator()
        self.uncertainty_analyzer = UncertaintyAnalyzer()
        self.evidence_quantifier = EvidenceStrengthQuantifier()
        
        # Configure reporting
        self.report_config = config or BayesianReportConfig(
            include_visualizations=True,
            uncertainty_simulations=1000
        )
        self.reporter = BayesianReporter(self.report_config)
        
        logger.info("Bayesian Process Tracing Integrator initialized")
    
    def enhance_analysis_with_bayesian(self, 
                                     graph_analysis: Dict[str, Any],
                                     evidence_assessments: List[EvidenceAssessment],
                                     hypothesis_descriptions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Enhance existing graph analysis with Bayesian confidence assessment.
        
        Args:
            graph_analysis: Existing process tracing analysis results
            evidence_assessments: Van Evera evidence assessments from LLM
            hypothesis_descriptions: Optional hypothesis descriptions
            
        Returns:
            Enhanced analysis with Bayesian confidence metrics
        """
        logger.info("Starting Bayesian enhancement of process tracing analysis")
        
        try:
            # Step 1: Create Bayesian hypothesis space from graph analysis
            hypothesis_space = self._create_hypothesis_space_from_analysis(
                graph_analysis, hypothesis_descriptions
            )
            
            # Step 2: Convert Van Evera evidence to Bayesian evidence
            bayesian_evidence = self._convert_evidence_assessments(
                evidence_assessments, hypothesis_space
            )
            
            # Step 3: Calculate confidence assessments for all hypotheses
            confidence_assessments = self._calculate_hypothesis_confidence(
                hypothesis_space, bayesian_evidence
            )
            
            # Step 4: Perform uncertainty analysis
            uncertainty_results = self._perform_uncertainty_analysis(
                hypothesis_space, bayesian_evidence
            )
            
            # Step 5: Generate Bayesian report
            bayesian_report = self._generate_bayesian_report(
                hypothesis_space, confidence_assessments, uncertainty_results
            )
            
            # Step 6: Integrate results with existing analysis
            enhanced_analysis = self._integrate_bayesian_results(
                graph_analysis, confidence_assessments, uncertainty_results, bayesian_report
            )
            
            logger.info("Bayesian enhancement completed successfully")
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Error in Bayesian enhancement: {e}")
            # Return original analysis if Bayesian enhancement fails
            graph_analysis['bayesian_enhancement_error'] = str(e)
            return graph_analysis
    
    def _create_hypothesis_space_from_analysis(self,
                                             graph_analysis: Dict[str, Any],
                                             hypothesis_descriptions: Optional[List[str]] = None) -> BayesianHypothesisSpace:
        """Create Bayesian hypothesis space from traditional analysis."""
        # Extract hypothesis information from graph analysis
        space = BayesianHypothesisSpace(
            hypothesis_space_id="process_tracing_analysis",
            description="Hypotheses extracted from process tracing analysis"
        )
        
        # Extract alternative explanations as hypotheses
        alt_explanations = graph_analysis.get('alternative_explanations', {})
        
        # Create primary hypothesis (main causal explanation)
        main_hypothesis = BayesianHypothesis(
            hypothesis_id="primary_explanation",
            description=self._extract_primary_hypothesis_description(graph_analysis),
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5,  # Default neutral prior
            posterior_probability=0.5  # Will be updated based on evidence
        )
        space.add_hypothesis(main_hypothesis)
        
        # Create alternative hypotheses
        for i, (alt_id, alt_info) in enumerate(alt_explanations.items()):
            alt_hypothesis = BayesianHypothesis(
                hypothesis_id=f"alternative_{i+1}",
                description=alt_info.get('description', f"Alternative explanation {i+1}"),
                hypothesis_type=HypothesisType.ALTERNATIVE,
                prior_probability=0.3,  # Lower prior for alternatives
                posterior_probability=0.3
            )
            space.add_hypothesis(alt_hypothesis)
        
        # Create null hypothesis
        null_hypothesis = BayesianHypothesis(
            hypothesis_id="null_hypothesis",
            description="No systematic causal relationship exists",
            hypothesis_type=HypothesisType.NULL,
            prior_probability=0.2,
            posterior_probability=0.2
        )
        space.add_hypothesis(null_hypothesis)
        
        # Normalize priors to sum to 1.0
        self._normalize_hypothesis_priors(space)
        
        return space
    
    def _normalize_hypothesis_priors(self, space: BayesianHypothesisSpace):
        """Normalize hypothesis priors to sum to 1.0."""
        hypotheses = list(space.hypotheses.values())
        if not hypotheses:
            return
        
        total_prior = sum(h.prior_probability for h in hypotheses)
        if total_prior > 0:
            for hypothesis in hypotheses:
                hypothesis.prior_probability = hypothesis.prior_probability / total_prior
                hypothesis.posterior_probability = hypothesis.prior_probability  # Reset to normalized prior
    
    def _extract_primary_hypothesis_description(self, graph_analysis: Dict[str, Any]) -> str:
        """Extract primary hypothesis description from graph analysis."""
        # Try to extract from various analysis sections
        if 'narrative_summary' in graph_analysis:
            return graph_analysis['narrative_summary'][:200] + "..."
        
        if 'causal_mechanisms' in graph_analysis:
            mechanisms = graph_analysis['causal_mechanisms']
            if mechanisms:
                return f"Primary causal mechanism: {list(mechanisms.keys())[0]}"
        
        return "Primary causal explanation extracted from process tracing analysis"
    
    def _convert_evidence_assessments(self,
                                    evidence_assessments: List[EvidenceAssessment],
                                    hypothesis_space: BayesianHypothesisSpace) -> List[BayesianEvidence]:
        """Convert Van Evera evidence assessments to Bayesian evidence."""
        bayesian_evidence = []
        
        primary_hypothesis = hypothesis_space.get_hypothesis("primary_explanation")
        
        for i, assessment in enumerate(evidence_assessments):
            try:
                # Convert using Van Evera Bayesian bridge
                evidence = self.van_evera_bridge.convert_evidence_assessment(
                    assessment,
                    hypothesis_context=primary_hypothesis.description,
                    source_node_id=f"evidence_node_{i+1}"
                )
                
                # Link evidence to hypothesis
                if evidence.get_likelihood_ratio() > 1.0:
                    primary_hypothesis.supporting_evidence.add(evidence.evidence_id)
                else:
                    primary_hypothesis.contradicting_evidence.add(evidence.evidence_id)
                
                hypothesis_space.add_evidence(evidence)
                bayesian_evidence.append(evidence)
                
            except Exception as e:
                logger.warning(f"Failed to convert evidence assessment {i}: {e}")
                continue
        
        logger.info(f"Converted {len(bayesian_evidence)} evidence assessments to Bayesian evidence")
        return bayesian_evidence
    
    def _calculate_hypothesis_confidence(self,
                                       hypothesis_space: BayesianHypothesisSpace,
                                       bayesian_evidence: List[BayesianEvidence]) -> Dict[str, ConfidenceAssessment]:
        """Calculate confidence assessments for all hypotheses."""
        confidence_assessments = {}
        
        for hypothesis_id, hypothesis in hypothesis_space.hypotheses.items():
            # Get evidence relevant to this hypothesis
            relevant_evidence = [
                e for e in bayesian_evidence 
                if e.evidence_id in hypothesis.supporting_evidence.union(hypothesis.contradicting_evidence)
            ]
            
            # Calculate confidence assessment
            assessment = self.confidence_calculator.calculate_confidence(
                hypothesis, hypothesis_space, relevant_evidence
            )
            
            confidence_assessments[hypothesis_id] = assessment
        
        logger.info(f"Calculated confidence for {len(confidence_assessments)} hypotheses")
        return confidence_assessments
    
    def _perform_uncertainty_analysis(self,
                                    hypothesis_space: BayesianHypothesisSpace,
                                    bayesian_evidence: List[BayesianEvidence]) -> Dict[str, UncertaintyAnalysisResult]:
        """Perform uncertainty analysis for key hypotheses."""
        uncertainty_results = {}
        
        # Focus on primary hypothesis for uncertainty analysis
        primary_hypothesis = hypothesis_space.get_hypothesis("primary_explanation")
        if primary_hypothesis:
            relevant_evidence = [
                e for e in bayesian_evidence 
                if e.evidence_id in primary_hypothesis.supporting_evidence.union(primary_hypothesis.contradicting_evidence)
            ]
            
            if relevant_evidence:
                result = self.uncertainty_analyzer.analyze_uncertainty(
                    primary_hypothesis,
                    hypothesis_space,
                    relevant_evidence,
                    n_simulations=self.report_config.uncertainty_simulations
                )
                uncertainty_results[primary_hypothesis.hypothesis_id] = result
        
        logger.info(f"Performed uncertainty analysis for {len(uncertainty_results)} hypotheses")
        return uncertainty_results
    
    def _generate_bayesian_report(self,
                                hypothesis_space: BayesianHypothesisSpace,
                                confidence_assessments: Dict[str, ConfidenceAssessment],
                                uncertainty_results: Dict[str, UncertaintyAnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive Bayesian report."""
        # Generate report for primary hypothesis
        primary_hypothesis_id = "primary_explanation"
        
        if primary_hypothesis_id in hypothesis_space.hypotheses:
            report = self.reporter.generate_comprehensive_report(
                hypothesis_space, primary_hypothesis_id
            )
            
            # Add confidence and uncertainty results to report metadata
            report['confidence_assessments'] = {
                hyp_id: {
                    'overall_confidence': assessment.overall_confidence,
                    'confidence_level': assessment.confidence_level.label,
                    'evidence_count': assessment.evidence_count,
                    'evidence_quality': assessment.evidence_quality_score
                }
                for hyp_id, assessment in confidence_assessments.items()
            }
            
            report['uncertainty_analysis'] = {
                hyp_id: {
                    'baseline_confidence': result.baseline_confidence,
                    'confidence_std': result.confidence_std,
                    'robustness_score': result.robustness_score,
                    'stability_score': result.stability_score
                }
                for hyp_id, result in uncertainty_results.items()
            }
            
            logger.info("Generated comprehensive Bayesian report")
            return report
        
        return {}
    
    def _integrate_bayesian_results(self,
                                  graph_analysis: Dict[str, Any],
                                  confidence_assessments: Dict[str, ConfidenceAssessment],
                                  uncertainty_results: Dict[str, UncertaintyAnalysisResult],
                                  bayesian_report: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate Bayesian results with existing analysis."""
        enhanced_analysis = graph_analysis.copy()
        
        # Add Bayesian section to analysis
        enhanced_analysis['bayesian_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'confidence_assessments': self._serialize_confidence_assessments(confidence_assessments),
            'uncertainty_analysis': self._serialize_uncertainty_results(uncertainty_results),
            'bayesian_report': bayesian_report,
            'integration_method': 'Phase6C_Integration'
        }
        
        # Add summary metrics to main analysis
        if confidence_assessments:
            primary_assessment = confidence_assessments.get('primary_explanation')
            if primary_assessment:
                enhanced_analysis['overall_confidence'] = primary_assessment.overall_confidence
                enhanced_analysis['confidence_level'] = primary_assessment.confidence_level.label
                enhanced_analysis['evidence_quality_score'] = primary_assessment.evidence_quality_score
        
        # Add uncertainty metrics
        if uncertainty_results:
            primary_uncertainty = uncertainty_results.get('primary_explanation')
            if primary_uncertainty:
                enhanced_analysis['confidence_uncertainty'] = primary_uncertainty.confidence_std
                enhanced_analysis['robustness_score'] = primary_uncertainty.robustness_score
        
        logger.info("Successfully integrated Bayesian results with process tracing analysis")
        return enhanced_analysis
    
    def _serialize_confidence_assessments(self, assessments: Dict[str, ConfidenceAssessment]) -> Dict[str, Any]:
        """Serialize confidence assessments for JSON output."""
        serialized = {}
        
        for hyp_id, assessment in assessments.items():
            serialized[hyp_id] = {
                'overall_confidence': assessment.overall_confidence,
                'confidence_level': assessment.confidence_level.label,
                'confidence_components': {
                    ct.value: score for ct, score in assessment.confidence_components.items()
                },
                'evidence_count': assessment.evidence_count,
                'evidence_quality_score': assessment.evidence_quality_score,
                'confidence_interval': assessment.confidence_interval,
                'interpretation': assessment.get_interpretation(),
                'recommendations': assessment.get_recommendations(),
                'assessment_timestamp': assessment.assessment_timestamp.isoformat()
            }
        
        return serialized
    
    def _serialize_uncertainty_results(self, results: Dict[str, UncertaintyAnalysisResult]) -> Dict[str, Any]:
        """Serialize uncertainty analysis results for JSON output."""
        serialized = {}
        
        for hyp_id, result in results.items():
            serialized[hyp_id] = {
                'baseline_confidence': result.baseline_confidence,
                'confidence_mean': result.confidence_mean,
                'confidence_std': result.confidence_std,
                'robustness_score': result.robustness_score,
                'stability_score': result.stability_score,
                'confidence_percentiles': result.confidence_percentiles,
                'n_simulations': result.n_simulations,
                'convergence_achieved': result.convergence_achieved,
                'uncertainty_interpretation': result._get_interpretation(),
                'most_sensitive_parameters': result._get_most_sensitive_parameters()[:5]
            }
        
        return serialized
    
    def save_bayesian_report(self, 
                           bayesian_report: Dict[str, Any], 
                           output_dir: str,
                           filename_prefix: str = "bayesian_analysis") -> str:
        """Save Bayesian report to HTML file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"{filename_prefix}_{timestamp}.html"
        html_filepath = output_path / html_filename
        
        # Write HTML content
        html_content = bayesian_report.get('html_content', '')
        if html_content:
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Saved Bayesian report to {html_filepath}")
            return str(html_filepath)
        
        return ""


def integrate_bayesian_analysis(graph_analysis: Dict[str, Any],
                               evidence_assessments: List[EvidenceAssessment],
                               output_dir: Optional[str] = None,
                               config: Optional[BayesianReportConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to integrate Bayesian analysis with process tracing results.
    
    Args:
        graph_analysis: Existing process tracing analysis results
        evidence_assessments: Van Evera evidence assessments from LLM
        output_dir: Optional directory to save Bayesian report
        config: Optional Bayesian report configuration
        
    Returns:
        Enhanced analysis with Bayesian confidence metrics
    """
    integrator = BayesianProcessTracingIntegrator(config)
    
    enhanced_analysis = integrator.enhance_analysis_with_bayesian(
        graph_analysis, evidence_assessments
    )
    
    # Save Bayesian report if output directory provided
    if output_dir and 'bayesian_analysis' in enhanced_analysis:
        bayesian_report = enhanced_analysis['bayesian_analysis'].get('bayesian_report', {})
        if bayesian_report:
            integrator.save_bayesian_report(bayesian_report, output_dir)
    
    return enhanced_analysis
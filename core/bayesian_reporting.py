"""
Bayesian Reporting for HTML Dashboard Integration.

Generates comprehensive probabilistic analysis reports for integration with 
the existing HTML dashboard. Provides rich visualizations, interactive elements,
and detailed narrative reporting for Bayesian process tracing results.
"""

import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import base64
import io

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace,
    BayesianProcessTracingModel, EvidenceType
)
from .confidence_calculator import (
    ConfidenceAssessment, CausalConfidenceCalculator, ConfidenceLevel, ConfidenceType
)
from .uncertainty_analysis import UncertaintyAnalysisResult, UncertaintyAnalyzer
from .van_evera_bayesian import VanEveraBayesianBridge
from .evidence_weighting import EvidenceStrengthQuantifier


@dataclass
class BayesianReportConfig:
    """Configuration for Bayesian report generation."""
    include_confidence_analysis: bool = True
    include_uncertainty_analysis: bool = True
    include_sensitivity_analysis: bool = True
    include_evidence_details: bool = True
    include_visualizations: bool = True
    
    # Visualization settings
    figure_width: int = 10
    figure_height: int = 6
    figure_dpi: int = 100
    color_scheme: str = "viridis"
    
    # Analysis settings
    uncertainty_simulations: int = 1000
    confidence_level: float = 0.95
    
    # Output settings
    embed_images: bool = True
    generate_json_data: bool = True


@dataclass
class BayesianReportSection:
    """Individual section of a Bayesian report."""
    section_id: str
    title: str
    content_html: str
    data: Dict[str, Any]
    visualizations: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.visualizations is None:
            self.visualizations = []


class BayesianReporter:
    """
    Generates comprehensive Bayesian analysis reports for HTML dashboard integration.
    
    Creates rich, interactive reports combining confidence assessment, uncertainty
    analysis, and evidence evaluation with visualizations and narrative summaries.
    """
    
    def __init__(self, config: Optional[BayesianReportConfig] = None):
        self.config = config or BayesianReportConfig()
        self.confidence_calculator = CausalConfidenceCalculator()
        self.uncertainty_analyzer = UncertaintyAnalyzer()
        self.van_evera_bridge = VanEveraBayesianBridge()
        self.evidence_quantifier = EvidenceStrengthQuantifier()
        
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            sns.set_palette(self.config.color_scheme)
    
    def generate_comprehensive_report(self,
                                    hypothesis_space: BayesianHypothesisSpace,
                                    target_hypothesis_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive Bayesian analysis report.
        
        Args:
            hypothesis_space: Hypothesis space to analyze
            target_hypothesis_id: Specific hypothesis to focus on (optional)
            
        Returns:
            Complete report with HTML, data, and visualizations
        """
        report_sections = []
        report_data = {}
        
        # Get target hypothesis or most likely one
        if target_hypothesis_id:
            target_hypothesis = hypothesis_space.get_hypothesis(target_hypothesis_id)
        else:
            # Find hypothesis with highest posterior
            all_hypotheses = hypothesis_space.hypotheses
            if all_hypotheses:
                target_hypothesis = max(
                    all_hypotheses.values(),
                    key=lambda h: h.posterior_probability
                )
            else:
                raise ValueError("No hypotheses found in hypothesis space")
        
        # Get relevant evidence
        evidence_list = self._get_hypothesis_evidence(target_hypothesis, hypothesis_space)
        
        # 1. Executive Summary
        if self.config.include_confidence_analysis:
            summary_section = self._generate_executive_summary(
                target_hypothesis, hypothesis_space, evidence_list
            )
            report_sections.append(summary_section)
            report_data['executive_summary'] = summary_section.data
        
        # 2. Confidence Analysis
        if self.config.include_confidence_analysis:
            confidence_section = self._generate_confidence_analysis(
                target_hypothesis, hypothesis_space, evidence_list
            )
            report_sections.append(confidence_section)
            report_data['confidence_analysis'] = confidence_section.data
        
        # 3. Uncertainty Analysis
        if self.config.include_uncertainty_analysis:
            uncertainty_section = self._generate_uncertainty_analysis(
                target_hypothesis, hypothesis_space, evidence_list
            )
            report_sections.append(uncertainty_section)
            report_data['uncertainty_analysis'] = uncertainty_section.data
        
        # 4. Evidence Analysis
        if self.config.include_evidence_details:
            evidence_section = self._generate_evidence_analysis(
                target_hypothesis, evidence_list
            )
            report_sections.append(evidence_section)
            report_data['evidence_analysis'] = evidence_section.data
        
        # 5. Hypothesis Comparison
        hypothesis_comparison_section = self._generate_hypothesis_comparison(
            hypothesis_space, target_hypothesis.hypothesis_id
        )
        report_sections.append(hypothesis_comparison_section)
        report_data['hypothesis_comparison'] = hypothesis_comparison_section.data
        
        # 6. Methodology and Assumptions
        methodology_section = self._generate_methodology_section(
            target_hypothesis, evidence_list
        )
        report_sections.append(methodology_section)
        report_data['methodology'] = methodology_section.data
        
        # Combine sections into complete report
        complete_report = self._assemble_complete_report(
            report_sections, target_hypothesis, hypothesis_space
        )
        
        # Add metadata
        complete_report['metadata'] = {
            'report_generated': datetime.now().isoformat(),
            'target_hypothesis': target_hypothesis.hypothesis_id,
            'hypothesis_count': len(hypothesis_space.hypotheses),
            'evidence_count': len(evidence_list),
            'config': asdict(self.config)
        }
        
        if self.config.generate_json_data:
            complete_report['raw_data'] = report_data
        
        return complete_report
    
    def _generate_executive_summary(self,
                                  hypothesis: BayesianHypothesis,
                                  hypothesis_space: BayesianHypothesisSpace,
                                  evidence_list: List[BayesianEvidence]) -> BayesianReportSection:
        """Generate executive summary section."""
        # Calculate confidence assessment
        confidence_assessment = self.confidence_calculator.calculate_confidence(
            hypothesis, hypothesis_space, evidence_list
        )
        
        # Get key metrics
        posterior_prob = hypothesis.posterior_probability
        confidence_level = confidence_assessment.confidence_level
        evidence_count = len(evidence_list)
        evidence_quality = confidence_assessment.evidence_quality_score
        
        # Generate narrative summary
        summary_html = f"""
        <div class="bayesian-executive-summary">
            <h3>Executive Summary: Bayesian Analysis</h3>
            
            <div class="key-findings">
                <h4>Key Findings</h4>
                <div class="finding-cards">
                    <div class="finding-card hypothesis-strength">
                        <h5>Hypothesis Support</h5>
                        <div class="metric-value">{posterior_prob:.1%}</div>
                        <div class="metric-label">Posterior Probability</div>
                    </div>
                    
                    <div class="finding-card confidence-level">
                        <h5>Analysis Confidence</h5>
                        <div class="metric-value confidence-{confidence_level.label}">{confidence_assessment.overall_confidence:.1%}</div>
                        <div class="metric-label">{confidence_level.label.replace('_', ' ').title()}</div>
                    </div>
                    
                    <div class="finding-card evidence-summary">
                        <h5>Evidence Base</h5>
                        <div class="metric-value">{evidence_count}</div>
                        <div class="metric-label">Evidence Pieces (Quality: {evidence_quality:.1%})</div>
                    </div>
                </div>
            </div>
            
            <div class="summary-interpretation">
                <h4>Interpretation</h4>
                <p>{confidence_assessment.get_interpretation()}</p>
                
                <h5>Key Recommendations</h5>
                <ul>
                {"".join(f"<li>{rec}</li>" for rec in confidence_assessment.get_recommendations())}
                </ul>
            </div>
        </div>
        """
        
        # Generate visualization if available
        visualizations = []
        if PLOTTING_AVAILABLE and self.config.include_visualizations:
            summary_viz = self._create_summary_visualization(confidence_assessment)
            if summary_viz:
                visualizations.append(summary_viz)
        
        return BayesianReportSection(
            section_id="executive_summary",
            title="Executive Summary",
            content_html=summary_html,
            data={
                'posterior_probability': posterior_prob,
                'confidence_score': confidence_assessment.overall_confidence,
                'confidence_level': confidence_level.label,
                'evidence_count': evidence_count,
                'evidence_quality_score': evidence_quality,
                'interpretation': confidence_assessment.get_interpretation(),
                'recommendations': confidence_assessment.get_recommendations()
            },
            visualizations=visualizations
        )
    
    def _generate_confidence_analysis(self,
                                    hypothesis: BayesianHypothesis,
                                    hypothesis_space: BayesianHypothesisSpace,
                                    evidence_list: List[BayesianEvidence]) -> BayesianReportSection:
        """Generate detailed confidence analysis section."""
        confidence_assessment = self.confidence_calculator.calculate_confidence(
            hypothesis, hypothesis_space, evidence_list
        )
        
        # Create detailed analysis HTML
        components_html = ""
        for conf_type, score in confidence_assessment.confidence_components.items():
            confidence_bar_width = score * 100
            components_html += f"""
            <div class="confidence-component">
                <div class="component-label">{conf_type.value.replace('_', ' ').title()}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_bar_width}%"></div>
                </div>
                <div class="component-score">{score:.2%}</div>
            </div>
            """
        
        confidence_html = f"""
        <div class="bayesian-confidence-analysis">
            <h3>Confidence Analysis</h3>
            
            <div class="overall-confidence">
                <h4>Overall Confidence: {confidence_assessment.overall_confidence:.1%}</h4>
                <p class="confidence-interpretation">{confidence_assessment.get_interpretation()}</p>
            </div>
            
            <div class="confidence-components">
                <h4>Confidence Components</h4>
                {components_html}
            </div>
            
            <div class="confidence-details">
                <h4>Supporting Metrics</h4>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Evidence Quality:</span>
                        <span class="metric-value">{confidence_assessment.evidence_quality_score:.1%}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Logical Coherence:</span>
                        <span class="metric-value">{confidence_assessment.logical_coherence_score:.1%}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Robustness:</span>
                        <span class="metric-value">{confidence_assessment.robustness_score:.1%}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Evidence Diversity:</span>
                        <span class="metric-value">{confidence_assessment.evidence_diversity_score:.1%}</span>
                    </div>
                </div>
            </div>
            
            <div class="confidence-interval">
                <h4>Confidence Interval</h4>
                <p>Estimated range: {confidence_assessment.confidence_interval[0]:.1%} - {confidence_assessment.confidence_interval[1]:.1%}</p>
            </div>
        </div>
        """
        
        # Generate visualizations
        visualizations = []
        if PLOTTING_AVAILABLE and self.config.include_visualizations:
            confidence_viz = self._create_confidence_breakdown_visualization(confidence_assessment)
            if confidence_viz:
                visualizations.append(confidence_viz)
        
        return BayesianReportSection(
            section_id="confidence_analysis",
            title="Confidence Analysis",
            content_html=confidence_html,
            data={
                'overall_confidence': confidence_assessment.overall_confidence,
                'confidence_components': {ct.value: score for ct, score in confidence_assessment.confidence_components.items()},
                'evidence_quality_score': confidence_assessment.evidence_quality_score,
                'logical_coherence_score': confidence_assessment.logical_coherence_score,
                'robustness_score': confidence_assessment.robustness_score,
                'evidence_diversity_score': confidence_assessment.evidence_diversity_score,
                'confidence_interval': confidence_assessment.confidence_interval
            },
            visualizations=visualizations
        )
    
    def _generate_uncertainty_analysis(self,
                                     hypothesis: BayesianHypothesis,
                                     hypothesis_space: BayesianHypothesisSpace,
                                     evidence_list: List[BayesianEvidence]) -> BayesianReportSection:
        """Generate uncertainty analysis section."""
        # Perform uncertainty analysis
        uncertainty_result = self.uncertainty_analyzer.analyze_uncertainty(
            hypothesis, hypothesis_space, evidence_list,
            n_simulations=self.config.uncertainty_simulations
        )
        
        # Generate uncertainty report
        uncertainty_report = self.uncertainty_analyzer.generate_uncertainty_report(uncertainty_result)
        
        # Create HTML content
        percentiles = uncertainty_result.confidence_percentiles
        uncertainty_html = f"""
        <div class="bayesian-uncertainty-analysis">
            <h3>Uncertainty Analysis</h3>
            
            <div class="uncertainty-summary">
                <h4>Uncertainty Summary</h4>
                <div class="uncertainty-metrics">
                    <div class="metric-item">
                        <span class="metric-label">Baseline Confidence:</span>
                        <span class="metric-value">{uncertainty_result.baseline_confidence:.1%}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Mean Confidence:</span>
                        <span class="metric-value">{uncertainty_result.confidence_mean:.1%}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Standard Deviation:</span>
                        <span class="metric-value">{uncertainty_result.confidence_std:.1%}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">95% Confidence Interval:</span>
                        <span class="metric-value">{percentiles.get('2.5', 0):.1%} - {percentiles.get('97.5', 1):.1%}</span>
                    </div>
                </div>
            </div>
            
            <div class="robustness-analysis">
                <h4>Robustness Assessment</h4>
                <div class="robustness-metrics">
                    <div class="metric-item">
                        <span class="metric-label">Robustness Score:</span>
                        <span class="metric-value">{uncertainty_result.robustness_score:.1%}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Stability Score:</span>
                        <span class="metric-value">{uncertainty_result.stability_score:.1%}</span>
                    </div>
                </div>
                <p class="uncertainty-interpretation">{uncertainty_result._get_interpretation()}</p>
            </div>
            
            <div class="sensitivity-summary">
                <h4>Most Sensitive Parameters</h4>
                <ul>
                {"".join(f"<li>{param}: {sens:.1%} sensitivity</li>" for param, sens in uncertainty_result._get_most_sensitive_parameters()[:5])}
                </ul>
            </div>
            
            <div class="uncertainty-recommendations">
                <h4>Recommendations</h4>
                <ul>
                {"".join(f"<li>{rec}</li>" for rec in uncertainty_report['recommendations'])}
                </ul>
            </div>
        </div>
        """
        
        # Generate visualizations
        visualizations = []
        if PLOTTING_AVAILABLE and self.config.include_visualizations:
            uncertainty_viz = self._create_uncertainty_visualization(uncertainty_result)
            if uncertainty_viz:
                visualizations.append(uncertainty_viz)
        
        return BayesianReportSection(
            section_id="uncertainty_analysis",
            title="Uncertainty Analysis",
            content_html=uncertainty_html,
            data=uncertainty_report,
            visualizations=visualizations
        )
    
    def _generate_evidence_analysis(self,
                                  hypothesis: BayesianHypothesis,
                                  evidence_list: List[BayesianEvidence]) -> BayesianReportSection:
        """Generate detailed evidence analysis section."""
        # Analyze each piece of evidence
        evidence_analyses = []
        
        for evidence in evidence_list:
            evidence_type_name = evidence.evidence_type.value.replace('_', ' ').title()
            likelihood_ratio = evidence.get_likelihood_ratio()
            
            # Determine support/contradiction
            if evidence.evidence_id in hypothesis.supporting_evidence:
                relationship = "Supporting"
                relationship_class = "supporting"
            elif evidence.evidence_id in hypothesis.contradicting_evidence:
                relationship = "Contradicting"
                relationship_class = "contradicting"
            else:
                relationship = "Neutral"
                relationship_class = "neutral"
            
            evidence_analyses.append({
                'evidence_id': evidence.evidence_id,
                'description': evidence.description,
                'evidence_type': evidence_type_name,
                'likelihood_ratio': likelihood_ratio,
                'reliability': evidence.reliability,
                'strength': evidence.strength,
                'relationship': relationship,
                'relationship_class': relationship_class
            })
        
        # Create evidence table HTML
        evidence_rows = ""
        for analysis in evidence_analyses:
            lr_display = f"{analysis['likelihood_ratio']:.2f}" if analysis['likelihood_ratio'] != float('inf') else "∞"
            evidence_rows += f"""
            <tr class="evidence-row {analysis['relationship_class']}">
                <td class="evidence-id">{analysis['evidence_id']}</td>
                <td class="evidence-type">{analysis['evidence_type']}</td>
                <td class="likelihood-ratio">{lr_display}</td>
                <td class="reliability">{analysis['reliability']:.1%}</td>
                <td class="strength">{analysis['strength']:.1%}</td>
                <td class="relationship">{analysis['relationship']}</td>
            </tr>
            """
        
        # Calculate evidence diversity
        evidence_diversity = self.evidence_quantifier.calculate_evidence_diversity(evidence_list)
        
        evidence_html = f"""
        <div class="bayesian-evidence-analysis">
            <h3>Evidence Analysis</h3>
            
            <div class="evidence-summary">
                <h4>Evidence Summary</h4>
                <div class="evidence-stats">
                    <div class="stat-item">
                        <span class="stat-label">Total Evidence:</span>
                        <span class="stat-value">{len(evidence_list)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Supporting:</span>
                        <span class="stat-value">{len(hypothesis.supporting_evidence)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Contradicting:</span>
                        <span class="stat-value">{len(hypothesis.contradicting_evidence)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Evidence Diversity:</span>
                        <span class="stat-value">{evidence_diversity:.1%}</span>
                    </div>
                </div>
            </div>
            
            <div class="evidence-table">
                <h4>Detailed Evidence Breakdown</h4>
                <table class="evidence-details-table">
                    <thead>
                        <tr>
                            <th>Evidence ID</th>
                            <th>Van Evera Type</th>
                            <th>Likelihood Ratio</th>
                            <th>Reliability</th>
                            <th>Strength</th>
                            <th>Relationship</th>
                        </tr>
                    </thead>
                    <tbody>
                        {evidence_rows}
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        # Generate visualizations
        visualizations = []
        if PLOTTING_AVAILABLE and self.config.include_visualizations:
            evidence_viz = self._create_evidence_visualization(evidence_list, hypothesis)
            if evidence_viz:
                visualizations.append(evidence_viz)
        
        return BayesianReportSection(
            section_id="evidence_analysis",
            title="Evidence Analysis",
            content_html=evidence_html,
            data={
                'evidence_analyses': evidence_analyses,
                'evidence_diversity': evidence_diversity,
                'supporting_count': len(hypothesis.supporting_evidence),
                'contradicting_count': len(hypothesis.contradicting_evidence),
                'total_count': len(evidence_list)
            },
            visualizations=visualizations
        )
    
    def _generate_hypothesis_comparison(self,
                                      hypothesis_space: BayesianHypothesisSpace,
                                      target_hypothesis_id: str) -> BayesianReportSection:
        """Generate hypothesis comparison section."""
        all_hypotheses = list(hypothesis_space.hypotheses.values())
        
        # Sort by posterior probability
        sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.posterior_probability, reverse=True)
        
        # Create comparison table
        comparison_rows = ""
        for i, hypothesis in enumerate(sorted_hypotheses):
            is_target = hypothesis.hypothesis_id == target_hypothesis_id
            row_class = "target-hypothesis" if is_target else ""
            
            # Calculate evidence counts
            supporting_count = len(hypothesis.supporting_evidence)
            contradicting_count = len(hypothesis.contradicting_evidence)
            
            comparison_rows += f"""
            <tr class="hypothesis-row {row_class}">
                <td class="hypothesis-rank">#{i+1}</td>
                <td class="hypothesis-id">{hypothesis.hypothesis_id}</td>
                <td class="hypothesis-type">{hypothesis.hypothesis_type.value.replace('_', ' ').title()}</td>
                <td class="posterior-prob">{hypothesis.posterior_probability:.1%}</td>
                <td class="prior-prob">{hypothesis.prior_probability:.1%}</td>
                <td class="evidence-count">{supporting_count} / {contradicting_count}</td>
                <td class="confidence">{hypothesis.confidence_level:.1%}</td>
            </tr>
            """
        
        comparison_html = f"""
        <div class="bayesian-hypothesis-comparison">
            <h3>Hypothesis Comparison</h3>
            
            <div class="comparison-summary">
                <p>Comparing {len(all_hypotheses)} hypotheses based on posterior probabilities and supporting evidence.</p>
            </div>
            
            <div class="hypothesis-table">
                <table class="hypothesis-comparison-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Hypothesis ID</th>
                            <th>Type</th>
                            <th>Posterior Prob.</th>
                            <th>Prior Prob.</th>
                            <th>Evidence (Support/Contradict)</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comparison_rows}
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        # Generate visualizations
        visualizations = []
        if PLOTTING_AVAILABLE and self.config.include_visualizations:
            comparison_viz = self._create_hypothesis_comparison_visualization(sorted_hypotheses)
            if comparison_viz:
                visualizations.append(comparison_viz)
        
        return BayesianReportSection(
            section_id="hypothesis_comparison",
            title="Hypothesis Comparison",
            content_html=comparison_html,
            data={
                'hypotheses': [
                    {
                        'hypothesis_id': h.hypothesis_id,
                        'hypothesis_type': h.hypothesis_type.value,
                        'posterior_probability': h.posterior_probability,
                        'prior_probability': h.prior_probability,
                        'supporting_evidence_count': len(h.supporting_evidence),
                        'contradicting_evidence_count': len(h.contradicting_evidence),
                        'confidence_level': h.confidence_level
                    }
                    for h in sorted_hypotheses
                ],
                'target_hypothesis_id': target_hypothesis_id
            },
            visualizations=visualizations
        )
    
    def _generate_methodology_section(self,
                                    hypothesis: BayesianHypothesis,
                                    evidence_list: List[BayesianEvidence]) -> BayesianReportSection:
        """Generate methodology and assumptions section."""
        # Analyze Van Evera evidence types
        evidence_types = {}
        for evidence in evidence_list:
            ev_type = evidence.evidence_type.value
            if ev_type not in evidence_types:
                evidence_types[ev_type] = 0
            evidence_types[ev_type] += 1
        
        # Create methodology description
        methodology_html = f"""
        <div class="bayesian-methodology">
            <h3>Methodology and Assumptions</h3>
            
            <div class="analysis-approach">
                <h4>Analysis Approach</h4>
                <p>This analysis employs Bayesian process tracing combined with Van Evera diagnostic tests 
                to evaluate causal hypotheses. The methodology integrates:</p>
                <ul>
                    <li><strong>Van Evera Diagnostic Tests:</strong> Evidence classified as hoop tests (necessity), 
                    smoking gun tests (sufficiency), doubly decisive tests (both), or straw-in-the-wind tests (weak evidence)</li>
                    <li><strong>Bayesian Inference:</strong> Systematic belief updating using likelihood ratios and prior probabilities</li>
                    <li><strong>Confidence Assessment:</strong> Multi-dimensional confidence evaluation including 
                    evidential, causal, coherence, robustness, and sensitivity factors</li>
                    <li><strong>Uncertainty Analysis:</strong> Monte Carlo simulation to quantify uncertainty and test robustness</li>
                </ul>
            </div>
            
            <div class="evidence-methodology">
                <h4>Evidence Classification</h4>
                <p>Evidence pieces were classified according to Van Evera diagnostic tests:</p>
                <ul>
                {"".join(f"<li><strong>{ev_type.replace('_', ' ').title()}:</strong> {count} pieces</li>" for ev_type, count in evidence_types.items())}
                </ul>
            </div>
            
            <div class="key-assumptions">
                <h4>Key Assumptions</h4>
                <ul>
                    <li><strong>Evidence Independence:</strong> Evidence pieces are assumed to be conditionally independent unless specified otherwise</li>
                    <li><strong>Likelihood Assessments:</strong> Probability assessments are based on Van Evera templates adjusted for evidence strength and reliability</li>
                    <li><strong>Prior Probabilities:</strong> Initial probabilities assigned based on available information and expert judgment</li>
                    <li><strong>Causal Mechanisms:</strong> Analysis assumes identifiable causal mechanisms linking causes to effects</li>
                </ul>
            </div>
            
            <div class="limitations">
                <h4>Limitations and Considerations</h4>
                <ul>
                    <li>Analysis is limited by the quality and completeness of available evidence</li>
                    <li>Probability assessments involve subjective judgment and may vary across analysts</li>
                    <li>Causal inference remains probabilistic and cannot establish absolute certainty</li>
                    <li>Results should be interpreted in context of broader theoretical and empirical knowledge</li>
                </ul>
            </div>
        </div>
        """
        
        return BayesianReportSection(
            section_id="methodology",
            title="Methodology and Assumptions",
            content_html=methodology_html,
            data={
                'evidence_type_distribution': evidence_types,
                'total_evidence_count': len(evidence_list),
                'analysis_date': datetime.now().isoformat(),
                'methodology_version': "Van Evera Bayesian Process Tracing v1.0"
            }
        )
    
    def _assemble_complete_report(self,
                                sections: List[BayesianReportSection],
                                hypothesis: BayesianHypothesis,
                                hypothesis_space: BayesianHypothesisSpace) -> Dict[str, Any]:
        """Assemble complete report from sections."""
        # Generate table of contents
        toc_html = "<div class='report-toc'><h3>Table of Contents</h3><ul>"
        for section in sections:
            toc_html += f"<li><a href='#{section.section_id}'>{section.title}</a></li>"
        toc_html += "</ul></div>"
        
        # Combine all section HTML
        sections_html = ""
        for section in sections:
            sections_html += f"""
            <section id="{section.section_id}" class="report-section">
                {section.content_html}
            </section>
            """
        
        # Create complete HTML
        complete_html = f"""
        <div class="bayesian-analysis-report">
            <header class="report-header">
                <h1>Bayesian Process Tracing Analysis Report</h1>
                <div class="report-metadata">
                    <p><strong>Target Hypothesis:</strong> {hypothesis.hypothesis_id}</p>
                    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Hypothesis Space:</strong> {hypothesis_space.hypothesis_space_id}</p>
                </div>
            </header>
            
            {toc_html}
            
            <main class="report-content">
                {sections_html}
            </main>
        </div>
        """
        
        # Collect all visualizations
        all_visualizations = []
        for section in sections:
            all_visualizations.extend(section.visualizations)
        
        return {
            'html_content': complete_html,
            'sections': [asdict(section) for section in sections],
            'visualizations': all_visualizations,
            'target_hypothesis': hypothesis.hypothesis_id,
            'hypothesis_space_id': hypothesis_space.hypothesis_space_id
        }
    
    def _get_hypothesis_evidence(self,
                               hypothesis: BayesianHypothesis,
                               hypothesis_space: BayesianHypothesisSpace) -> List[BayesianEvidence]:
        """Get evidence relevant to a hypothesis."""
        evidence_ids = hypothesis.supporting_evidence.union(hypothesis.contradicting_evidence)
        evidence_list = []
        
        for evidence_id in evidence_ids:
            evidence = hypothesis_space.get_evidence(evidence_id)
            if evidence:
                evidence_list.append(evidence)
        
        return evidence_list
    
    # Visualization methods (require matplotlib)
    def _create_summary_visualization(self, confidence_assessment: ConfidenceAssessment) -> Optional[Dict[str, str]]:
        """Create executive summary visualization."""
        if not PLOTTING_AVAILABLE:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Overall confidence gauge
        self._create_confidence_gauge(ax1, confidence_assessment.overall_confidence, "Overall Confidence")
        
        # Evidence breakdown pie chart
        supporting = confidence_assessment.evidence_count
        contradicting = confidence_assessment.contradicting_evidence_count
        if supporting + contradicting > 0:
            ax2.pie([supporting, contradicting], labels=['Supporting', 'Contradicting'], 
                   autopct='%1.0f', startangle=90)
            ax2.set_title("Evidence Distribution")
        
        # Confidence components radar chart
        self._create_confidence_radar(ax3, confidence_assessment.confidence_components)
        
        # Uncertainty visualization
        ci_lower, ci_upper = confidence_assessment.confidence_interval
        ax4.barh(['Confidence'], [confidence_assessment.overall_confidence], 
                xerr=[[confidence_assessment.overall_confidence - ci_lower], 
                      [ci_upper - confidence_assessment.overall_confidence]])
        ax4.set_xlim(0, 1)
        ax4.set_title("Confidence with Uncertainty")
        
        plt.tight_layout()
        
        return self._save_figure_to_dict(fig, "summary_visualization")
    
    def _create_confidence_gauge(self, ax, confidence_value: float, title: str):
        """Create a confidence gauge visualization."""
        if not PLOTTING_AVAILABLE:
            return
        
        # Create semi-circular gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=10)
        
        # Confidence arc
        confidence_theta = theta[:int(confidence_value * 100)]
        ax.plot(np.cos(confidence_theta), np.sin(confidence_theta), 'green', linewidth=10)
        
        # Needle
        needle_angle = confidence_value * np.pi
        ax.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f"{title}: {confidence_value:.1%}")
        ax.axis('off')
    
    def _create_confidence_radar(self, ax, confidence_components: Dict[ConfidenceType, float]):
        """Create radar chart for confidence components."""
        if not PLOTTING_AVAILABLE or not confidence_components:
            return
        
        # Get component names and values
        components = list(confidence_components.keys())
        values = list(confidence_components.values())
        
        # Number of variables
        N = len(components)
        
        # Angle for each component
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Values for plotting
        values += values[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.value.replace('_', '\n').title() for c in components])
        ax.set_ylim(0, 1)
        ax.set_title("Confidence Components")
    
    def _save_figure_to_dict(self, fig: Figure, name: str) -> Dict[str, str]:
        """Save matplotlib figure to base64 encoded dictionary."""
        if not self.config.embed_images:
            return {"name": name, "type": "reference", "path": f"{name}.png"}
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.config.figure_dpi, bbox_inches='tight')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "name": name,
            "type": "embedded",
            "data": f"data:image/png;base64,{image_data}",
            "format": "png"
        }
    
    # Additional visualization methods would be implemented here
    def _create_confidence_breakdown_visualization(self, confidence_assessment: ConfidenceAssessment) -> Optional[Dict[str, str]]:
        """Create detailed confidence breakdown visualization."""
        # Implementation would create detailed confidence visualizations
        return None
    
    def _create_uncertainty_visualization(self, uncertainty_result: UncertaintyAnalysisResult) -> Optional[Dict[str, str]]:
        """Create uncertainty analysis visualization."""
        # Implementation would create uncertainty distribution plots
        return None
    
    def _create_evidence_visualization(self, evidence_list: List[BayesianEvidence], hypothesis: BayesianHypothesis) -> Optional[Dict[str, str]]:
        """Create evidence analysis visualization."""
        # Implementation would create evidence strength and type visualizations
        return None
    
    def _create_hypothesis_comparison_visualization(self, hypotheses: List[BayesianHypothesis]) -> Optional[Dict[str, str]]:
        """Create hypothesis comparison visualization."""
        # Implementation would create hypothesis comparison charts
        return None
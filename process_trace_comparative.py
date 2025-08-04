"""
Comparative Process Tracing Analysis Pipeline

Entry point for multi-case comparative process tracing analysis including
case management, cross-case alignment, mechanism detection, and MSS/MDS analysis.

Author: Claude Code Implementation
Date: August 2025
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.case_manager import CaseManager
from core.graph_alignment import GraphAligner
from core.mechanism_detector import MechanismDetector
from core.mss_mds_analysis import MSS_MDS_Analyzer
from core.comparative_models import (
    CaseSelectionCriteria, ComparisonType, MultiCaseAnalysisResult,
    ComparativeAnalysisError
)


class ComparativeProcessTracer:
    """
    Main class for conducting comparative process tracing analysis.
    """
    
    def __init__(self, case_directory: Optional[str] = None, 
                 output_directory: str = "comparative_output"):
        """
        Initialize comparative process tracer.
        
        Args:
            case_directory: Directory containing case files
            output_directory: Directory for analysis outputs
        """
        self.case_directory = case_directory
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.case_manager = CaseManager(case_directory)
        self.graph_aligner = GraphAligner()
        self.mechanism_detector = MechanismDetector()
        self.mss_mds_analyzer = MSS_MDS_Analyzer()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Analysis results
        self.analysis_results: Optional[MultiCaseAnalysisResult] = None
    
    def load_cases(self, case_files: Optional[List[str]] = None, 
                  pattern: str = "*.json") -> List[str]:
        """
        Load cases for comparative analysis.
        
        Args:
            case_files: Specific case files to load (optional)
            pattern: File pattern for automatic discovery
            
        Returns:
            List of loaded case IDs
        """
        self.logger.info("Loading cases for comparative analysis")
        
        if case_files:
            # Load specific case files
            loaded_cases = []
            for case_file in case_files:
                try:
                    case_id = self.case_manager.load_case(case_file)
                    loaded_cases.append(case_id)
                except ComparativeAnalysisError as e:
                    self.logger.error(f"Failed to load case {case_file}: {e}")
            return loaded_cases
        else:
            # Auto-discover and load cases
            return self.case_manager.load_cases_from_directory(pattern=pattern)
    
    def conduct_comparative_analysis(self, 
                                   case_selection_criteria: Optional[CaseSelectionCriteria] = None,
                                   comparison_types: List[ComparisonType] = None) -> MultiCaseAnalysisResult:
        """
        Conduct comprehensive comparative analysis.
        
        Args:
            case_selection_criteria: Criteria for case selection
            comparison_types: Types of comparisons to perform
            
        Returns:
            Complete comparative analysis results
        """
        self.logger.info("Starting comprehensive comparative analysis")
        
        if comparison_types is None:
            comparison_types = [ComparisonType.MOST_SIMILAR_SYSTEMS, 
                             ComparisonType.MOST_DIFFERENT_SYSTEMS,
                             ComparisonType.DIVERSE_CASE]
        
        # Step 1: Case selection
        case_ids = self._select_cases(case_selection_criteria)
        if len(case_ids) < 2:
            raise ComparativeAnalysisError("Need at least 2 cases for comparative analysis")
        
        self.logger.info(f"Selected {len(case_ids)} cases for analysis: {case_ids}")
        
        # Step 2: Cross-case graph alignment
        graphs = {case_id: self.case_manager.get_case(case_id) for case_id in case_ids}
        node_mappings = self._align_graphs(graphs)
        
        # Step 3: Mechanism detection
        mechanism_patterns = self._detect_mechanisms(graphs, node_mappings)
        
        # Step 4: MSS/MDS analysis
        mss_results, mds_results = self._conduct_mss_mds_analysis(
            graphs, case_ids, mechanism_patterns, comparison_types
        )
        
        # Step 5: Cross-case evidence analysis
        cross_case_evidence = self._analyze_cross_case_evidence(
            graphs, mechanism_patterns, node_mappings
        )
        
        # Step 6: Compile results
        analysis_results = self._compile_analysis_results(
            case_ids, node_mappings, mechanism_patterns, 
            mss_results, mds_results, cross_case_evidence
        )
        
        self.analysis_results = analysis_results
        self.logger.info("Comparative analysis completed successfully")
        
        return analysis_results
    
    def generate_comparative_report(self, output_format: str = "html") -> str:
        """
        Generate comprehensive comparative analysis report.
        
        Args:
            output_format: Output format ('html', 'json', 'md')
            
        Returns:
            Path to generated report
        """
        if not self.analysis_results:
            raise ComparativeAnalysisError("No analysis results available. Run analysis first.")
        
        self.logger.info(f"Generating comparative report in {output_format} format")
        
        if output_format == "html":
            return self._generate_html_report()
        elif output_format == "json":
            return self._generate_json_report()
        elif output_format == "md":
            return self._generate_markdown_report()
        else:
            raise ComparativeAnalysisError(f"Unsupported output format: {output_format}")
    
    def _select_cases(self, criteria: Optional[CaseSelectionCriteria]) -> List[str]:
        """
        Select cases based on criteria.
        
        Args:
            criteria: Case selection criteria
            
        Returns:
            List of selected case IDs
        """
        if criteria:
            return self.case_manager.select_cases(criteria)
        else:
            # Use all loaded cases
            return self.case_manager.list_cases()
    
    def _align_graphs(self, graphs: Dict[str, Any]) -> List[Any]:
        """
        Align graphs across cases to find corresponding nodes.
        
        Args:
            graphs: Dictionary of case graphs
            
        Returns:
            List of node mappings
        """
        self.logger.info("Aligning graphs across cases")
        
        all_mappings = self.graph_aligner.align_multiple_graphs(graphs)
        
        # Flatten mappings
        node_mappings = []
        for (case1, case2), mappings in all_mappings.items():
            node_mappings.extend(mappings)
        
        self.logger.info(f"Created {len(node_mappings)} cross-case node mappings")
        return node_mappings
    
    def _detect_mechanisms(self, graphs: Dict[str, Any], node_mappings: List[Any]) -> List[Any]:
        """
        Detect recurring mechanisms across cases.
        
        Args:
            graphs: Dictionary of case graphs
            node_mappings: Cross-case node mappings
            
        Returns:
            List of mechanism patterns
        """
        self.logger.info("Detecting recurring mechanisms")
        
        mechanisms = self.mechanism_detector.detect_recurring_mechanisms(
            graphs, node_mappings
        )
        
        # Classify mechanisms
        universal_mechanisms = self.mechanism_detector.identify_universal_mechanisms(
            graphs, mechanisms
        )
        conditional_mechanisms = self.mechanism_detector.identify_conditional_mechanisms(
            graphs, mechanisms
        )
        
        self.logger.info(f"Detected {len(mechanisms)} mechanisms ({len(universal_mechanisms)} universal, {len(conditional_mechanisms)} conditional)")
        
        return mechanisms
    
    def _conduct_mss_mds_analysis(self, graphs: Dict[str, Any], case_ids: List[str],
                                 mechanisms: List[Any], comparison_types: List[ComparisonType]) -> tuple:
        """
        Conduct MSS and MDS analyses.
        
        Args:
            graphs: Dictionary of case graphs
            case_ids: List of case IDs
            mechanisms: Mechanism patterns
            comparison_types: Types of comparisons to perform
            
        Returns:
            Tuple of (MSS results, MDS results)
        """
        self.logger.info("Conducting MSS/MDS comparative analysis")
        
        # Get case metadata
        metadata = {}
        for case_id in case_ids:
            metadata[case_id] = self.case_manager.get_case_metadata(case_id)
        
        mss_results = []
        mds_results = []
        
        if ComparisonType.MOST_SIMILAR_SYSTEMS in comparison_types:
            mss_results = self.mss_mds_analyzer.conduct_mss_analysis(
                graphs, metadata, mechanisms
            )
        
        if ComparisonType.MOST_DIFFERENT_SYSTEMS in comparison_types:
            mds_results = self.mss_mds_analyzer.conduct_mds_analysis(
                graphs, metadata, mechanisms
            )
        
        self.logger.info(f"Completed MSS ({len(mss_results)}) and MDS ({len(mds_results)}) analyses")
        
        return mss_results, mds_results
    
    def _analyze_cross_case_evidence(self, graphs: Dict[str, Any], 
                                   mechanisms: List[Any], node_mappings: List[Any]) -> List[Any]:
        """
        Analyze evidence patterns across cases.
        
        Args:
            graphs: Dictionary of case graphs
            mechanisms: Mechanism patterns
            node_mappings: Cross-case node mappings
            
        Returns:
            List of cross-case evidence patterns
        """
        self.logger.info("Analyzing cross-case evidence patterns")
        
        cross_case_evidence = []
        
        # For each mechanism, analyze evidence consistency across cases
        for mechanism in mechanisms:
            if len(mechanism.participating_cases) >= 2:
                evidence_pattern = self._extract_evidence_pattern(mechanism, graphs)
                if evidence_pattern:
                    cross_case_evidence.append(evidence_pattern)
        
        self.logger.info(f"Identified {len(cross_case_evidence)} cross-case evidence patterns")
        return cross_case_evidence
    
    def _extract_evidence_pattern(self, mechanism: Any, graphs: Dict[str, Any]) -> Optional[Any]:
        """
        Extract evidence pattern for a mechanism across cases.
        
        Args:
            mechanism: Mechanism pattern
            graphs: Dictionary of case graphs
            
        Returns:
            Cross-case evidence pattern or None
        """
        from core.comparative_models import CrossCaseEvidence
        
        case_evidence = {}
        evidence_strength = {}
        
        for case_id in mechanism.participating_cases:
            graph = graphs[case_id]
            
            # Extract evidence from node descriptions (simplified)
            evidence_list = []
            for node in graph.nodes():
                node_data = graph.nodes[node]
                if node_data.get('type') in mechanism.core_nodes:
                    description = node_data.get('description', '')
                    if description:
                        evidence_list.append(description)
            
            if evidence_list:
                case_evidence[case_id] = "; ".join(evidence_list)
                evidence_strength[case_id] = len(evidence_list) / len(mechanism.core_nodes)
        
        if len(case_evidence) >= 2:
            return CrossCaseEvidence(
                evidence_id=f"evidence_{mechanism.pattern_id}",
                evidence_type="mechanism_support",
                description=f"Evidence for mechanism: {mechanism.pattern_name}",
                case_evidence=case_evidence,
                evidence_strength=evidence_strength,
                pattern_consistency=mechanism.consistency_score,
                aggregate_support=sum(evidence_strength.values()) / len(evidence_strength)
            )
        
        return None
    
    def _compile_analysis_results(self, case_ids: List[str], node_mappings: List[Any],
                                mechanisms: List[Any], mss_results: List[Any],
                                mds_results: List[Any], cross_case_evidence: List[Any]) -> MultiCaseAnalysisResult:
        """
        Compile all analysis results into comprehensive summary.
        
        Args:
            case_ids: List of case IDs
            node_mappings: Cross-case node mappings
            mechanisms: Mechanism patterns
            mss_results: MSS analysis results
            mds_results: MDS analysis results
            cross_case_evidence: Cross-case evidence patterns
            
        Returns:
            Complete analysis results
        """
        # Get case metadata
        case_metadata = {}
        for case_id in case_ids:
            case_metadata[case_id] = self.case_manager.get_case_metadata(case_id)
        
        # Calculate mapping coverage
        total_nodes = sum(len(self.case_manager.get_case(case_id).nodes()) for case_id in case_ids)
        mapping_coverage = len(node_mappings) / total_nodes if total_nodes > 0 else 0.0
        
        # Classify mechanisms
        universal_patterns = [m.pattern_id for m in mechanisms if hasattr(m, 'mechanism_type') and 
                            str(m.mechanism_type) == 'MechanismType.UNIVERSAL']
        conditional_patterns = [m.pattern_id for m in mechanisms if hasattr(m, 'mechanism_type') and 
                              str(m.mechanism_type) == 'MechanismType.CONDITIONAL']
        
        # Convert MSS/MDS results to comparison results
        from core.comparative_models import ComparisonResult
        
        pairwise_comparisons = []
        
        # Convert MSS results
        for mss in mss_results:
            comparison = ComparisonResult(
                comparison_id=mss.comparison_id,
                comparison_type=ComparisonType.MOST_SIMILAR_SYSTEMS,
                primary_case=mss.primary_case,
                comparison_cases=[mss.comparison_case],
                overall_similarity=mss.context_similarity,
                context_similarity=mss.context_similarity,
                outcome_similarity=1.0 - mss.outcome_difference,
                different_mechanisms=mss.differentiating_mechanisms,
                analysis_confidence=mss.confidence_level
            )
            pairwise_comparisons.append(comparison)
        
        # Convert MDS results
        for mds in mds_results:
            comparison = ComparisonResult(
                comparison_id=mds.comparison_id,
                comparison_type=ComparisonType.MOST_DIFFERENT_SYSTEMS,
                primary_case=mds.primary_case,
                comparison_cases=[mds.comparison_case],
                overall_similarity=1.0 - mds.context_difference,
                context_similarity=1.0 - mds.context_difference,
                outcome_similarity=mds.outcome_similarity,
                shared_mechanisms=mds.shared_mechanisms,
                analysis_confidence=mds.confidence_level
            )
            pairwise_comparisons.append(comparison)
        
        # Calculate overall confidence
        if pairwise_comparisons:
            overall_confidence = sum(c.analysis_confidence for c in pairwise_comparisons) / len(pairwise_comparisons)
        else:
            overall_confidence = 0.5
        
        # Create complete results
        analysis_results = MultiCaseAnalysisResult(
            analysis_id=f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            analysis_name="Comparative Process Tracing Analysis",
            description=f"Multi-case comparative analysis of {len(case_ids)} cases",
            total_cases=len(case_ids),
            case_metadata=case_metadata,
            node_mappings=node_mappings,
            mapping_coverage=mapping_coverage,
            mechanism_patterns=mechanisms,
            universal_patterns=universal_patterns,
            conditional_patterns=conditional_patterns,
            pairwise_comparisons=pairwise_comparisons,
            mss_analyses=[c for c in pairwise_comparisons if c.comparison_type == ComparisonType.MOST_SIMILAR_SYSTEMS],
            mds_analyses=[c for c in pairwise_comparisons if c.comparison_type == ComparisonType.MOST_DIFFERENT_SYSTEMS],
            cross_case_evidence=cross_case_evidence,
            overall_confidence=overall_confidence,
            methodological_rigor=0.8  # Fixed high value for now
        )
        
        return analysis_results
    
    def _generate_html_report(self) -> str:
        """
        Generate HTML comparative analysis report.
        
        Returns:
            Path to HTML report
        """
        results = self.analysis_results
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparative Process Tracing Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container-fluid">
        <h1 class="mt-4 mb-4">Comparative Process Tracing Analysis Report</h1>
        
        <!-- Executive Summary -->
        <div class="card mb-4">
            <div class="card-header">
                <h2>Executive Summary</h2>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-primary">{results.total_cases}</h3>
                            <p>Total Cases</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-success">{len(results.mechanism_patterns)}</h3>
                            <p>Mechanisms Detected</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-info">{len(results.pairwise_comparisons)}</h3>
                            <p>Comparisons</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-warning">{results.overall_confidence:.2f}</h3>
                            <p>Overall Confidence</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Case Summary -->
        <div class="card mb-4">
            <div class="card-header">
                <h2>Case Summary</h2>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Case ID</th>
                                <th>Name</th>
                                <th>Primary Outcome</th>
                                <th>Data Quality</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        for case_id, metadata in results.case_metadata.items():
            html_content += f"""
                            <tr>
                                <td>{case_id}</td>
                                <td>{metadata.case_name}</td>
                                <td>{metadata.primary_outcome or 'N/A'}</td>
                                <td>{metadata.data_quality_score:.2f}</td>
                            </tr>
"""
        
        html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Mechanism Patterns -->
        <div class="card mb-4">
            <div class="card-header">
                <h2>Mechanism Patterns</h2>
            </div>
            <div class="card-body">
"""
        
        for mechanism in results.mechanism_patterns[:10]:  # Show top 10
            html_content += f"""
                <div class="card mb-3">
                    <div class="card-header">
                        <h5>{mechanism.pattern_name}</h5>
                        <span class="badge bg-secondary">{mechanism.mechanism_type.value if hasattr(mechanism, 'mechanism_type') else 'Unknown'}</span>
                    </div>
                    <div class="card-body">
                        <p>{mechanism.description}</p>
                        <p><strong>Participating Cases:</strong> {', '.join(mechanism.participating_cases)}</p>
                        <p><strong>Pattern Strength:</strong> {mechanism.pattern_strength:.2f}</p>
                        <p><strong>Consistency:</strong> {mechanism.consistency_score:.2f}</p>
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <!-- MSS/MDS Analysis -->
        <div class="card mb-4">
            <div class="card-header">
                <h2>MSS/MDS Analysis Results</h2>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h4>Most Similar Systems (MSS)</h4>
"""
        
        mss_analyses = results.mss_analyses
        if mss_analyses:
            for analysis in mss_analyses[:5]:  # Show top 5
                html_content += f"""
                        <div class="card mb-2">
                            <div class="card-body">
                                <h6>{analysis.primary_case} vs {', '.join(analysis.comparison_cases)}</h6>
                                <p>Context Similarity: {analysis.context_similarity:.2f}</p>
                                <p>Outcome Similarity: {analysis.outcome_similarity:.2f}</p>
                                <p>Confidence: {analysis.analysis_confidence:.2f}</p>
                            </div>
                        </div>
"""
        else:
            html_content += "<p>No MSS analyses available.</p>"
        
        html_content += """
                    </div>
                    <div class="col-md-6">
                        <h4>Most Different Systems (MDS)</h4>
"""
        
        mds_analyses = results.mds_analyses
        if mds_analyses:
            for analysis in mds_analyses[:5]:  # Show top 5
                html_content += f"""
                        <div class="card mb-2">
                            <div class="card-body">
                                <h6>{analysis.primary_case} vs {', '.join(analysis.comparison_cases)}</h6>
                                <p>Context Similarity: {analysis.context_similarity:.2f}</p>
                                <p>Outcome Similarity: {analysis.outcome_similarity:.2f}</p>
                                <p>Confidence: {analysis.analysis_confidence:.2f}</p>
                            </div>
                        </div>
"""
        else:
            html_content += "<p>No MDS analyses available.</p>"
        
        html_content += """
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Cross-Case Evidence -->
        <div class="card mb-4">
            <div class="card-header">
                <h2>Cross-Case Evidence Patterns</h2>
            </div>
            <div class="card-body">
"""
        
        if results.cross_case_evidence:
            for evidence in results.cross_case_evidence[:10]:  # Show top 10
                html_content += f"""
                <div class="card mb-3">
                    <div class="card-body">
                        <h6>{evidence.description}</h6>
                        <p><strong>Pattern Consistency:</strong> {evidence.pattern_consistency:.2f}</p>
                        <p><strong>Aggregate Support:</strong> {evidence.aggregate_support:.2f}</p>
                        <p><strong>Cases:</strong> {', '.join(evidence.case_evidence.keys())}</p>
                    </div>
                </div>
"""
        else:
            html_content += "<p>No cross-case evidence patterns identified.</p>"
        
        html_content += f"""
            </div>
        </div>
        
        <!-- Analysis Metadata -->
        <div class="card mb-4">
            <div class="card-header">
                <h2>Analysis Metadata</h2>
            </div>
            <div class="card-body">
                <p><strong>Analysis ID:</strong> {results.analysis_id}</p>
                <p><strong>Analysis Date:</strong> {results.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Mapping Coverage:</strong> {results.mapping_coverage:.2f}</p>
                <p><strong>Methodological Rigor:</strong> {results.methodological_rigor:.2f}</p>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        
        # Save HTML report
        report_path = self.output_directory / f"comparative_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {report_path}")
        return str(report_path)
    
    def _generate_json_report(self) -> str:
        """
        Generate JSON comparative analysis report.
        
        Returns:
            Path to JSON report
        """
        # Convert analysis results to JSON-serializable format
        results_dict = {
            'analysis_id': self.analysis_results.analysis_id,
            'analysis_name': self.analysis_results.analysis_name,
            'description': self.analysis_results.description,
            'total_cases': self.analysis_results.total_cases,
            'mapping_coverage': self.analysis_results.mapping_coverage,
            'overall_confidence': self.analysis_results.overall_confidence,
            'methodological_rigor': self.analysis_results.methodological_rigor,
            'analysis_date': self.analysis_results.analysis_date.isoformat(),
            'mechanism_count': len(self.analysis_results.mechanism_patterns),
            'comparison_count': len(self.analysis_results.pairwise_comparisons),
            'universal_patterns': self.analysis_results.universal_patterns,
            'conditional_patterns': self.analysis_results.conditional_patterns
        }
        
        # Save JSON report
        report_path = self.output_directory / f"comparative_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON report generated: {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self) -> str:
        """
        Generate Markdown comparative analysis report.
        
        Returns:
            Path to Markdown report
        """
        results = self.analysis_results
        
        # Generate Markdown content
        md_content = f"""# Comparative Process Tracing Analysis Report

**Analysis ID:** {results.analysis_id}  
**Date:** {results.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}  
**Cases:** {results.total_cases}  
**Overall Confidence:** {results.overall_confidence:.2f}

## Executive Summary

This comparative process tracing analysis examined {results.total_cases} cases and identified {len(results.mechanism_patterns)} recurring mechanisms through systematic cross-case comparison.

### Key Metrics

- **Total Cases:** {results.total_cases}
- **Mechanisms Detected:** {len(results.mechanism_patterns)}
- **Cross-Case Mappings:** {len(results.node_mappings)}
- **Mapping Coverage:** {results.mapping_coverage:.2f}
- **Pairwise Comparisons:** {len(results.pairwise_comparisons)}
- **Overall Confidence:** {results.overall_confidence:.2f}
- **Methodological Rigor:** {results.methodological_rigor:.2f}

## Case Summary

| Case ID | Name | Primary Outcome | Data Quality |
|---------|------|-----------------|--------------|
"""
        
        for case_id, metadata in results.case_metadata.items():
            md_content += f"| {case_id} | {metadata.case_name} | {metadata.primary_outcome or 'N/A'} | {metadata.data_quality_score:.2f} |\n"
        
        md_content += f"""
## Mechanism Patterns

Identified {len(results.mechanism_patterns)} recurring mechanisms:

"""
        
        for i, mechanism in enumerate(results.mechanism_patterns[:10], 1):
            md_content += f"""### {i}. {mechanism.pattern_name}

- **Type:** {mechanism.mechanism_type.value if hasattr(mechanism, 'mechanism_type') else 'Unknown'}
- **Description:** {mechanism.description}
- **Participating Cases:** {', '.join(mechanism.participating_cases)}
- **Pattern Strength:** {mechanism.pattern_strength:.2f}
- **Consistency Score:** {mechanism.consistency_score:.2f}

"""
        
        md_content += f"""
## Comparative Analysis Results

### Most Similar Systems (MSS) Analysis

{len(results.mss_analyses)} MSS comparisons conducted:

"""
        
        for analysis in results.mss_analyses[:5]:
            md_content += f"""- **{analysis.primary_case} vs {', '.join(analysis.comparison_cases)}**
  - Context Similarity: {analysis.context_similarity:.2f}
  - Outcome Similarity: {analysis.outcome_similarity:.2f}
  - Confidence: {analysis.analysis_confidence:.2f}

"""
        
        md_content += f"""
### Most Different Systems (MDS) Analysis

{len(results.mds_analyses)} MDS comparisons conducted:

"""
        
        for analysis in results.mds_analyses[:5]:
            md_content += f"""- **{analysis.primary_case} vs {', '.join(analysis.comparison_cases)}**
  - Context Similarity: {analysis.context_similarity:.2f}
  - Outcome Similarity: {analysis.outcome_similarity:.2f}
  - Confidence: {analysis.analysis_confidence:.2f}

"""
        
        md_content += f"""
## Cross-Case Evidence Patterns

{len(results.cross_case_evidence)} cross-case evidence patterns identified:

"""
        
        for evidence in results.cross_case_evidence[:10]:
            md_content += f"""### {evidence.description}

- **Pattern Consistency:** {evidence.pattern_consistency:.2f}
- **Aggregate Support:** {evidence.aggregate_support:.2f}
- **Cases Involved:** {', '.join(evidence.case_evidence.keys())}

"""
        
        md_content += """
## Methodology

This analysis employed a systematic comparative process tracing methodology:

1. **Case Selection:** Cases were selected based on theoretical and methodological criteria
2. **Graph Alignment:** Cross-case node and edge mappings were identified using similarity algorithms
3. **Mechanism Detection:** Recurring causal mechanisms were detected through pattern analysis
4. **MSS/MDS Analysis:** Most Similar Systems and Most Different Systems comparisons were conducted
5. **Evidence Synthesis:** Cross-case evidence patterns were identified and validated

## Conclusions

[Analysis conclusions would be inserted here based on specific findings]

---
*Report generated by Comparative Process Tracing Toolkit*
"""
        
        # Save Markdown report
        report_path = self.output_directory / f"comparative_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"Markdown report generated: {report_path}")
        return str(report_path)
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for comparative analysis.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger('comparative_process_tracer')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_directory / f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger


def main():
    """Main entry point for comparative process tracing analysis."""
    parser = argparse.ArgumentParser(description="Comparative Process Tracing Analysis")
    parser.add_argument('--case-directory', type=str, help='Directory containing case files')
    parser.add_argument('--case-files', nargs='+', help='Specific case files to analyze')
    parser.add_argument('--output-directory', type=str, default='comparative_output',
                       help='Output directory for results')
    parser.add_argument('--output-format', choices=['html', 'json', 'md'], default='html',
                       help='Report output format')
    parser.add_argument('--comparison-types', nargs='+', 
                       choices=['mss', 'mds', 'diverse'],
                       default=['mss', 'mds'],
                       help='Types of comparisons to perform')
    
    args = parser.parse_args()
    
    # Map comparison type strings to enums
    comparison_type_map = {
        'mss': ComparisonType.MOST_SIMILAR_SYSTEMS,
        'mds': ComparisonType.MOST_DIFFERENT_SYSTEMS,
        'diverse': ComparisonType.DIVERSE_CASE
    }
    comparison_types = [comparison_type_map[ct] for ct in args.comparison_types]
    
    try:
        # Initialize comparative tracer
        tracer = ComparativeProcessTracer(
            case_directory=args.case_directory,
            output_directory=args.output_directory
        )
        
        # Load cases
        case_ids = tracer.load_cases(case_files=args.case_files)
        print(f"Loaded {len(case_ids)} cases: {case_ids}")
        
        # Conduct analysis
        results = tracer.conduct_comparative_analysis(comparison_types=comparison_types)
        print(f"Analysis completed. Detected {len(results.mechanism_patterns)} mechanisms.")
        
        # Generate report
        report_path = tracer.generate_comparative_report(output_format=args.output_format)
        print(f"Report generated: {report_path}")
        
    except Exception as e:
        print(f"Error during comparative analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
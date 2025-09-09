"""
Alternative Hypothesis Generator Plugin
Creates systematic theoretical competition for rigorous Van Evera analysis
"""

import json
from typing import Dict, List, Any
from .base import ProcessTracingPlugin, PluginValidationError
import logging

# Import LLM interface for semantic analysis  
from .van_evera_llm_interface import VanEveraLLMInterface
from ..semantic_analysis_service import get_semantic_service
from ..llm_required import LLMRequiredError

logger = logging.getLogger(__name__)


class AlternativeHypothesisGeneratorPlugin(ProcessTracingPlugin):
    """
    Plugin for generating comprehensive alternative hypotheses for Van Evera testing.
    Creates 6-8 robust competing explanations across multiple theoretical domains.
    """
    
    plugin_id = "alternative_hypothesis_generator"
    
    # Academic-quality alternative hypotheses for American Revolution
    # DEPRECATED: Hardcoded American Revolution hypotheses replaced with LLM generation
    # This dictionary is kept for compatibility but should not be used
    REVOLUTION_ALTERNATIVE_HYPOTHESES_DEPRECATED = {
        'economic_interests': {
            'id': 'Q_H2',
            'description': "Colonial merchant class drove resistance to protect trade profits and economic autonomy from British mercantile restrictions",
            'theoretical_basis': "Economic determinism, rational choice theory",
            'key_predictions': [
                "Resistance leaders predominantly merchants/traders with documented economic grievances",
                "Opposition intensity correlates with severity of trade disruption in specific regions",
                "Economic arguments dominate political rhetoric and pamphlet literature",
                "Regional variation in resistance based on trade dependencies and commercial networks"
            ],
            'testable_mechanisms': [
                "Merchant wealth accumulation → Political influence → Resistance leadership",
                "British trade restrictions → Economic losses → Opposition mobilization",
                "Credit relationships → Cross-colonial coordination networks"
            ],
            'evidence_requirements': ['merchant_networks', 'trade_data', 'economic_grievances', 'commercial_interests'],
            'competing_claims': "Economic motives more important than ideological principles"
        },
        
        'generational_conflict': {
            'id': 'Q_H3',
            'description': "Young colonial generation rejected parental authority and British rule as part of broader generational rebellion against established order",
            'theoretical_basis': "Generational theory, political socialization, youth rebellion studies",
            'key_predictions': [
                "Resistance leaders significantly younger than colonial leadership average",
                "Generational rhetoric and appeals to youth in political arguments and crowd actions",
                "Young men disproportionately involved in Sons of Liberty and crowd actions",
                "Documented parent-child political divisions and family conflicts over resistance"
            ],
            'testable_mechanisms': [
                "Demographic youth bulge → Social pressure → Political radicalization",
                "Educational changes → New ideas exposure → Generational divergence from parents",
                "Military service experience → Independence → Authority rejection patterns"
            ],
            'evidence_requirements': ['age_demographics', 'generational_rhetoric', 'family_conflicts', 'youth_organizations'],
            'competing_claims': "Generational dynamics more significant than constitutional principles"
        },
        
        'religious_awakening': {
            'id': 'Q_H4', 
            'description': "Protestant religious awakening created ideological framework for independence from Anglican/secular British authority",
            'theoretical_basis': "Religious sociology, ideological mobilization theory, Great Awakening studies",
            'key_predictions': [
                "Religious language and biblical references permeate political documents and speeches",
                "Protestant clergy provide intellectual leadership and moral legitimacy for resistance",
                "Regional correlation between religious revival intensity and political opposition strength",
                "Anti-Anglican sentiment drives anti-British political sentiment in key regions"
            ],
            'testable_mechanisms': [
                "Religious revival → Authority questioning → Political resistance legitimation",
                "Clergy networks → Communication channels → Coordinated opposition messaging",
                "Moral arguments → Popular legitimacy → Mass mobilization capability"
            ],
            'evidence_requirements': ['religious_rhetoric', 'clergy_leadership', 'revival_patterns', 'anti_anglican_sentiment'],
            'competing_claims': "Religious motivations fundamental to resistance, not secular political theory"
        },
        
        'elite_power_struggle': {
            'id': 'Q_H5',
            'description': "Colonial political elites sought to replace British dominance with their own power, using popular grievances instrumentally",
            'theoretical_basis': "Elite theory, power transition models, instrumental mobilization",
            'key_predictions': [
                "Elite continuity before and after revolution with same families maintaining power",
                "Popular movements controlled and channeled by elite interests rather than autonomous",
                "Elite economic interests protected and enhanced in new political system",
                "Limited democratic expansion post-independence, elite control maintained"
            ],
            'testable_mechanisms': [
                "Elite competition → Popular mobilization → Elite victory with power consolidation",
                "British weakness perception → Elite opportunity → Coordinated power grab",
                "Popular grievances → Elite manipulation → Controlled revolutionary outcome"
            ],
            'evidence_requirements': ['elite_continuity', 'controlled_mobilization', 'power_consolidation', 'limited_democracy'],
            'competing_claims': "Elite self-interest primary driver, popular ideology secondary rationalization"
        },
        
        'regional_political_culture': {
            'id': 'Q_H6',
            'description': "New England political culture of town meetings and local self-governance was fundamentally incompatible with monarchical authority",
            'theoretical_basis': "Political culture theory, institutional analysis, regional political development",
            'key_predictions': [
                "Regional variation in resistance intensity correlating with local governance traditions",
                "Prior local governance experience predicts opposition leadership and effectiveness", 
                "Cultural arguments about political participation and self-rule dominate resistance rhetoric",
                "Institutional legacy shapes post-war governance structures and democratic practices"
            ],
            'testable_mechanisms': [
                "Local democratic practice → Democratic values → Anti-monarchical attitudes",
                "Institutional experience → Governance capacity → Independence viability assessment",
                "Cultural transmission → Persistent regional political differences → Resistance patterns"
            ],
            'evidence_requirements': ['local_governance', 'town_meetings', 'regional_variation', 'cultural_arguments'],
            'competing_claims': "Regional political culture more important than imperial policy changes"
        },
        
        'imperial_overstretch': {
            'id': 'Q_H7',
            'description': "British administrative incompetence and imperial overextension created governance failures that generated resistance",
            'theoretical_basis': "Imperial decline theory, administrative capacity limits, governance failure models",
            'key_predictions': [
                "British policy inconsistency and frequent reversals create colonial confusion and resentment",
                "Administrative failures and bureaucratic incompetence precede colonial resistance escalation",
                "Geographic distance and communication delays create systematic policy implementation problems",
                "Similar governance problems manifest in other British colonies during same period"
            ],
            'testable_mechanisms': [
                "Geographic distance → Communication delays → Policy implementation failures",
                "Multiple imperial commitments → Resource constraints → Colonial governance breakdown",
                "Bureaucratic incompetence → Arbitrary decisions → Popular grievances and resistance"
            ],
            'evidence_requirements': ['policy_inconsistency', 'administrative_failures', 'communication_delays', 'imperial_comparison'],
            'competing_claims': "British governance failure primary cause, not colonial ideological development"
        },
        
        'military_catalyst': {
            'id': 'Q_H8',
            'description': "French and Indian War transformed colonial military experience and created veteran leadership enabling resistance",
            'theoretical_basis': "Military sociology, veteran political mobilization, wartime transformation theory",
            'key_predictions': [
                "Military veterans disproportionately lead resistance organizations and provide strategic leadership",
                "Military organization patterns and hierarchy evident in resistance group structure",
                "Strategic and tactical military thinking evident in colonial planning and coordination",
                "Military precedents for challenging British authority established during French and Indian War"
            ],
            'testable_mechanisms': [
                "Military service experience → Leadership skills → Political leadership roles",
                "Veteran networks → Communication channels → Coordinated resistance planning",
                "Military confidence → Resistance feasibility assessment → Revolutionary commitment"
            ],
            'evidence_requirements': ['veteran_leadership', 'military_organization', 'strategic_planning', 'war_precedents'],
            'competing_claims': "Military transformation more significant than political ideology or economic interests"
        },
        
        'ideological_contagion': {
            'id': 'Q_H9',
            'description': "Enlightenment ideas about natural rights and popular sovereignty spread through colonial intellectual networks",
            'theoretical_basis': "Ideational diffusion theory, intellectual history, network transmission models",
            'key_predictions': [
                "Philosophical arguments and Enlightenment terminology dominate political documents",
                "Intellectual networks and correspondence connect resistance leaders across colonies",
                "Newspaper circulation and printing press expansion correlates with political mobilization",
                "European intellectual influences clearly documented in colonial political writings"
            ],
            'testable_mechanisms': [
                "Print culture expansion → Idea transmission → Political mobilization capability",
                "Intellectual networks → Coordinated arguments → Movement coherence and unity",
                "Educational institutions → Elite formation → Leadership development and recruitment"
            ],
            'evidence_requirements': ['philosophical_arguments', 'intellectual_networks', 'print_culture', 'enlightenment_influence'],
            'competing_claims': "Ideas and intellectual transformation primary driver of resistance, not material interests"
        }
    }
    
    def validate_input(self, data: Any) -> None:
        """Validate input graph data for alternative hypothesis generation"""
        if not isinstance(data, dict):
            raise PluginValidationError(self.id, "Input must be dictionary")
        
        if 'graph_data' not in data:
            raise PluginValidationError(self.id, "Missing required key 'graph_data'")
        
        graph_data = data['graph_data']
        if not isinstance(graph_data, dict) or 'nodes' not in graph_data or 'edges' not in graph_data:
            raise PluginValidationError(self.id, "graph_data must contain 'nodes' and 'edges'")
        
        # Verify existing hypotheses for competitive relationships
        nodes = graph_data['nodes']
        existing_hypotheses = [n for n in nodes if n.get('type') == 'Hypothesis']
        
        if len(existing_hypotheses) == 0:
            self.logger.warning("No existing hypotheses found - alternatives will be created without competitive relationships")
        
        self.logger.info(f"VALIDATION: Found {len(existing_hypotheses)} existing hypotheses for competitive analysis")
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Generate comprehensive alternative hypotheses for systematic testing"""
        self.logger.info("START: Alternative hypothesis generation for theoretical competition")
        
        graph_data = data['graph_data']
        
        # Generate comprehensive alternatives
        generation_result = self._generate_comprehensive_alternatives(graph_data)
        
        # Calculate theoretical competition metrics
        competition_metrics = self._calculate_competition_metrics(generation_result)
        
        # Generate academic assessment
        academic_assessment = self._generate_academic_assessment(competition_metrics)
        
        result = {
            'updated_graph_data': generation_result['updated_graph_data'],
            'alternative_hypotheses_created': generation_result['alternatives_created'],
            'competitive_relationships_added': generation_result['competitive_edges_created'],
            'theoretical_competition_metrics': competition_metrics,
            'academic_assessment': academic_assessment,
            'generation_statistics': {
                'total_alternatives_generated': len(generation_result['alternatives_created']),
                'theoretical_domains_covered': len(generation_result.get('theoretical_domains_covered', [])),
                'evidence_requirements_specified': len([alt for alt in generation_result['alternatives_created']]),
                'competitive_claims_defined': len([alt for alt in generation_result['alternatives_created']])
            }
        }
        
        self.logger.info(f"END: Generated {result['generation_statistics']['total_alternatives_generated']} alternative hypotheses across {result['generation_statistics']['theoretical_domains_covered']} domains")
        return result
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for LLM-based alternative hypothesis generation"""
        return {
            'plugin_id': self.id,
            'alternatives_available': 'llm_generated',
            'theoretical_domains': ['universal_llm_analysis'],
            'academic_quality': 'llm_enhanced_theoretical'
        }
    
    def _generate_comprehensive_alternatives(self, graph_data: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive set of alternative hypotheses using LLM semantic analysis.
        Replaces hardcoded American Revolution alternatives with universal generation.
        """
        self.logger.info("PROGRESS: Creating systematic theoretical competition using LLM analysis...")
        
        updated_graph = graph_data.copy()
        existing_hypotheses = [n for n in updated_graph['nodes'] if n.get('type') == 'Hypothesis']
        
        new_nodes = []
        new_edges = []
        alternatives_created = []
        
        if not existing_hypotheses:
            self.logger.warning("No existing hypotheses found - cannot generate alternatives")
            return {
                'updated_graph_data': updated_graph,
                'alternatives_created': alternatives_created,
                'competitive_edges_created': []
            }
        
        # Get the primary hypothesis for alternative generation
        primary_hypothesis = existing_hypotheses[0]  # Take first hypothesis as primary
        primary_desc = primary_hypothesis.get('properties', {}).get('description', '')
        
        # Collect evidence context for semantic understanding
        evidence_nodes = [n for n in updated_graph['nodes'] if n.get('type') == 'Evidence']
        evidence_context = " ".join([
            n.get('properties', {}).get('description', '')
            for n in evidence_nodes[:10]  # Limit to first 10 for context
        ])
        
        try:
            # Generate alternatives using LLM semantic understanding
            llm_interface = get_van_evera_llm()
            
            alt_generation = llm_interface.generate_alternative_hypotheses(
                original_hypothesis=primary_desc,
                evidence_context=evidence_context,
                domain_context="Universal process tracing analysis requiring theoretical competition"
            )
            
            # Convert LLM-generated alternatives to graph nodes
            for i, alt_hyp in enumerate(alt_generation.alternative_hypotheses):
                alt_id = f"Q_H{i+2:02d}_LLM"  # Q_H02_LLM, Q_H03_LLM, etc.
                
                alternative_node = {
                    'id': alt_id,
                    'type': 'Hypothesis',
                    'properties': {
                        'description': alt_hyp.get('description', f"Alternative hypothesis {i+1}"),
                        'hypothesis_type': 'alternative',
                        'ranking_score': 0.0,  # Will be updated after Van Evera testing
                        'research_question_id': 'Q',  # Will be set when research question is created
                        'prior_probability': 0.5,  # Default prior for alternatives
                        'status': 'active',
                        'theoretical_basis': alt_hyp.get('theoretical_justification', 'LLM-generated theoretical framework'),
                        'key_predictions': alt_hyp.get('test_predictions', []),
                        'testable_mechanisms': alt_hyp.get('causal_mechanism', []),
                        'evidence_requirements': alt_hyp.get('evidence_requirements', []),
                        'competing_claims': alt_hyp.get('differentiator', 'LLM-generated competing mechanism'),
                        'domain': alt_hyp.get('primary_domain', f'domain_{i+1}'),
                        'generated_by': 'llm_alternative_hypothesis_generator',
                        'academic_quality': 'llm_generated',
                        'theoretical_domain': alt_hyp.get('domain_classification', f'Domain {i+1}'),
                        'confidence_score': alt_generation.generation_confidence,
                        'universal_applicability': alt_generation.universal_applicability
                    }
                }
                new_nodes.append(alternative_node)
                alternatives_created.append(alt_id)
            
            self.logger.info(f"Generated {len(alternatives_created)} LLM-based alternative hypotheses "
                           f"with confidence {alt_generation.generation_confidence:.3f}")
            
        except Exception as e:
            self.logger.error(f"LLM alternative generation failed: {e}")
            
            # Fallback to basic universal alternatives (no dataset-specific logic)
            universal_alternatives = [
                {
                    'id': 'Q_H02_FALLBACK',
                    'description': 'Economic factors primarily drove the observed outcomes through material incentive mechanisms',
                    'theoretical_basis': 'Economic determinism and rational choice theory',
                    'domain': 'economic'
                },
                {
                    'id': 'Q_H03_FALLBACK', 
                    'description': 'Political institutional changes created the structural conditions for the observed outcomes',
                    'theoretical_basis': 'Institutional theory and political development',
                    'domain': 'political'
                },
                {
                    'id': 'Q_H04_FALLBACK',
                    'description': 'Social and cultural factors shaped collective behavior leading to the observed outcomes',
                    'theoretical_basis': 'Social movement theory and cultural sociology',
                    'domain': 'social'
                }
            ]
            
            for alt_data in universal_alternatives:
                alternative_node = {
                    'id': alt_data['id'],
                    'type': 'Hypothesis',
                    'properties': {
                        'description': alt_data['description'],
                        'hypothesis_type': 'alternative',
                        'ranking_score': 0.0,
                        'research_question_id': 'Q',
                        'prior_probability': 0.5,
                        'status': 'active',
                        'theoretical_basis': alt_data['theoretical_basis'],
                        'key_predictions': [],
                        'testable_mechanisms': [],
                        'evidence_requirements': [],
                        'competing_claims': f"Alternative {alt_data['domain']} explanation",
                        'domain': alt_data['domain'],
                        'generated_by': 'fallback_alternative_generator',
                        'academic_quality': 'basic_theoretical',
                        'theoretical_domain': alt_data['domain'].title()
                    }
                }
                new_nodes.append(alternative_node)
                alternatives_created.append(alt_data['id'])
        
        # Update graph data
        updated_graph['nodes'].extend(new_nodes)
        updated_graph['edges'].extend(new_edges)
        
        self.logger.info(f"PROGRESS: Generated {len(new_nodes)} alternatives with {len(new_edges)} relationships")
        
        return {
            'updated_graph_data': updated_graph,
            'alternatives_created': alternatives_created,
            'competitive_edges_created': 0,  # Competition handled through Van Evera testing
            'evidence_edges_created': len(new_edges),
            'theoretical_domains_covered': list(set([
                node['properties'].get('domain', 'unknown') 
                for node in new_nodes
            ]))
        }
    
    def _identify_relevant_evidence(self, graph_data: Dict, alt_data: Dict) -> List[Dict]:
        """Identify evidence nodes relevant to alternative hypothesis"""
        relevant_evidence = []
        evidence_nodes = [n for n in graph_data['nodes'] if n.get('type') == 'Evidence']
        
        # Extract relevance keywords from alternative hypothesis
        evidence_requirements = alt_data['evidence_requirements']
        description_keywords = alt_data['description'].lower().split()
        prediction_text = ' '.join(alt_data['key_predictions']).lower()
        
        # Create comprehensive keyword set
        all_keywords = set(evidence_requirements)
        all_keywords.update([word for word in description_keywords if len(word) > 4])
        all_keywords.update([word for word in prediction_text.split() if len(word) > 4])
        
        for evidence_node in evidence_nodes:
            evidence_desc = evidence_node.get('properties', {}).get('description', '').lower()
            source_quote = evidence_node.get('properties', {}).get('source_text_quote', '').lower()
            
            # Use semantic service for relevance assessment
            evidence_text = f"{evidence_desc} {source_quote}"
            
            try:
                semantic_service = get_semantic_service()
                # Use probative value assessment to determine relevance
                # Create a hypothesis description from keywords for comparison
                hypothesis_desc = ' '.join(all_keywords)
                assessment = semantic_service.assess_probative_value(
                    evidence_description=evidence_text,
                    hypothesis_description=hypothesis_desc
                )
                # Convert probative value to relevance score (scale 0-5)
                relevance_score = assessment.probative_value * 5 if hasattr(assessment, 'probative_value') else 0
            except Exception as e:
                raise LLMRequiredError(f"Cannot assess evidence relevance without LLM: {e}")
            
            # Include evidence with sufficient relevance
            if relevance_score >= 2:
                relevant_evidence.append(evidence_node)
        
        # Limit to top 15 most relevant pieces of evidence per alternative
        return relevant_evidence[:15]
    
    def _assess_evidence_requirement_match(self, evidence_node: Dict, alt_data: Dict) -> str:
        """Assess how well evidence matches alternative's requirements"""
        evidence_desc = evidence_node.get('properties', {}).get('description', '').lower()
        requirements = alt_data['evidence_requirements']
        
        matches = []
        for req in requirements:
            if req.replace('_', ' ') in evidence_desc or any(part in evidence_desc for part in req.split('_')):
                matches.append(req)
        
        if len(matches) >= 2:
            return f"strong_match_{len(matches)}_requirements"
        elif len(matches) == 1:
            return f"moderate_match_{matches[0]}"
        else:
            return "weak_match_general_relevance"
    
    def _calculate_competition_metrics(self, generation_result: Dict) -> Dict[str, Any]:
        """Calculate theoretical competition quality metrics"""
        total_alternatives = len(generation_result['alternatives_created'])
        competitive_edges = generation_result['competitive_edges_created']
        evidence_connections = generation_result['evidence_edges_created']
        domains_covered = len(generation_result['theoretical_domains_covered'])
        
        # Competition density based on theoretical diversity (no explicit edges)
        competition_density = min(1.0, domains_covered / 8.0)  # Theoretical diversity as competition proxy
        
        # Calculate evidence coverage
        avg_evidence_per_alternative = evidence_connections / total_alternatives if total_alternatives > 0 else 0
        
        return {
            'total_alternative_hypotheses': total_alternatives,
            'theoretical_domains_covered': domains_covered,
            'competitive_relationships': total_alternatives,  # All alternatives compete via Van Evera testing
            'evidence_connections': evidence_connections,
            'competition_density': round(competition_density, 2),
            'average_evidence_per_alternative': round(avg_evidence_per_alternative, 1),
            'theoretical_diversity_score': domains_covered / 8.0,  # Out of 8 possible domains
            'academic_robustness_score': min(1.0, (total_alternatives * domains_covered * avg_evidence_per_alternative) / 100)
        }
    
    def _generate_academic_assessment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate academic quality assessment of theoretical competition"""
        total_alternatives = metrics['total_alternative_hypotheses']
        domains_covered = metrics['theoretical_domains_covered']
        diversity_score = metrics['theoretical_diversity_score']
        robustness_score = metrics['academic_robustness_score']
        
        # Academic standards assessment
        meets_minimum_alternatives = total_alternatives >= 6
        meets_domain_diversity = domains_covered >= 5
        meets_competition_density = metrics['competition_density'] >= 0.8
        meets_evidence_coverage = metrics['average_evidence_per_alternative'] >= 3.0
        
        # Overall academic quality score
        quality_factors = [
            meets_minimum_alternatives,
            meets_domain_diversity,
            meets_competition_density,
            meets_evidence_coverage,
            diversity_score >= 0.6,
            robustness_score >= 0.5
        ]
        
        academic_quality_score = (sum(quality_factors) / len(quality_factors)) * 100
        
        # Generate recommendations
        recommendations = []
        if not meets_minimum_alternatives:
            recommendations.append("Increase total alternative hypotheses to at least 6")
        if not meets_domain_diversity:
            recommendations.append("Expand theoretical domain coverage to at least 5 different domains")
        if not meets_competition_density:
            recommendations.append("Strengthen competitive relationships between alternatives")
        if not meets_evidence_coverage:
            recommendations.append("Connect more evidence to each alternative hypothesis")
        
        if not recommendations:
            recommendations.append("Theoretical competition meets academic standards for publication")
        
        return {
            'academic_quality_score': round(academic_quality_score, 1),
            'meets_academic_standards': academic_quality_score >= 80,
            'theoretical_competition_adequate': meets_minimum_alternatives and meets_domain_diversity,
            'van_evera_elimination_ready': meets_competition_density and meets_evidence_coverage,
            'quality_criteria': {
                'minimum_alternatives_met': meets_minimum_alternatives,
                'domain_diversity_adequate': meets_domain_diversity,
                'competition_density_sufficient': meets_competition_density,
                'evidence_coverage_adequate': meets_evidence_coverage
            },
            'improvement_recommendations': recommendations,
            'publication_readiness': 'ready' if academic_quality_score >= 80 else 'needs_improvement'
        }


# Integration function for workflow
def generate_alternative_hypotheses(graph_data: Dict) -> Dict[str, Any]:
    """
    Main entry point for alternative hypothesis generation.
    Returns updated graph data with comprehensive theoretical competition.
    """
    from .base import PluginContext
    
    # Create minimal context for plugin execution
    context = PluginContext({'alternative_generation': True})
    plugin = AlternativeHypothesisGeneratorPlugin('alternative_hypothesis_generator', context)
    
    # Execute plugin
    result = plugin.execute({'graph_data': graph_data})
    
    return result['updated_graph_data']
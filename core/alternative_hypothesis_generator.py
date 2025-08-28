"""
Alternative Hypothesis Generation System
Creates systematic theoretical competition for rigorous Van Evera analysis
"""

from typing import Dict, List
import json

class AlternativeHypothesisGenerator:
    """
    Generates competing explanations for systematic Van Evera hypothesis testing.
    Creates 6-8 robust alternative hypotheses across multiple theoretical domains.
    """
    
    # Academic-quality alternative explanations for American Revolution
    REVOLUTION_ALTERNATIVE_HYPOTHESES = {
        'economic_interests': {
            'description': "Colonial merchant class drove resistance to protect trade profits and economic autonomy from British mercantile restrictions",
            'key_predictions': [
                "Resistance leaders predominantly merchants/traders",
                "Opposition correlates with trade disruption severity",
                "Economic arguments dominate political rhetoric",
                "Regional variation based on trade dependencies"
            ],
            'theoretical_basis': "Economic determinism, rational choice theory",
            'testable_mechanisms': [
                "Merchant wealth → Political influence → Resistance leadership",
                "Trade restrictions → Economic losses → Opposition mobilization",
                "Credit relationships → Cross-colonial coordination"
            ]
        },
        
        'generational_conflict': {
            'description': "Young colonial generation rejected parental authority and British rule as part of broader generational rebellion",
            'key_predictions': [
                "Resistance leaders significantly younger than average",
                "Generational rhetoric in political arguments",
                "Young men disproportionately involved in crowd actions",
                "Parent-child political divisions documented"
            ],
            'theoretical_basis': "Generational theory, political socialization",
            'testable_mechanisms': [
                "Demographic bulge → Social pressure → Political radicalization",
                "Educational changes → New ideas → Generational divergence",
                "Military service → Independence experience → Authority rejection"
            ]
        },
        
        'religious_awakening': {
            'description': "Protestant religious awakening created ideological framework for independence from Anglican/secular British authority",
            'key_predictions': [
                "Religious language permeates political documents",
                "Clergy provide intellectual leadership for resistance", 
                "Regional correlation between religious revival and political opposition",
                "Anti-Anglican sentiment drives anti-British sentiment"
            ],
            'theoretical_basis': "Religious sociology, ideological mobilization theory",
            'testable_mechanisms': [
                "Religious revival → Authority questioning → Political resistance",
                "Clergy networks → Communication channels → Coordinated opposition",
                "Moral arguments → Popular legitimacy → Mass mobilization"
            ]
        },
        
        'elite_power_struggle': {
            'description': "Colonial political elites sought to replace British dominance with their own power, using popular grievances instrumentally",
            'key_predictions': [
                "Elite continuity before/after revolution",
                "Popular movements controlled/channeled by elites",
                "Elite economic interests protected in new system",
                "Limited democratic expansion post-independence"
            ],
            'theoretical_basis': "Elite theory, power transition models",
            'testable_mechanisms': [
                "Elite competition → Popular mobilization → Elite victory",
                "British weakness → Elite opportunity → Power grab",
                "Popular grievances → Elite manipulation → Controlled revolution"
            ]
        },
        
        'regional_political_culture': {
            'description': "New England political culture of town meetings and local self-governance was fundamentally incompatible with monarchical authority",
            'key_predictions': [
                "Regional variation in resistance intensity",
                "Local governance experience predicts opposition",
                "Cultural arguments about political participation",
                "Institutional legacy shapes post-war governance"
            ],
            'theoretical_basis': "Political culture theory, institutional analysis",
            'testable_mechanisms': [
                "Local democracy → Democratic values → Anti-monarchical attitudes",
                "Institutional experience → Governance capacity → Independence viability",
                "Cultural transmission → Persistent regional differences"
            ]
        },
        
        'imperial_overstretch': {
            'description': "British administrative incompetence and imperial overextension created governance failures that generated resistance",
            'key_predictions': [
                "British policy inconsistency and frequent reversals",
                "Administrative failures precede colonial resistance",
                "Distance/communication delays create policy problems",
                "Similar problems in other British colonies"
            ],
            'theoretical_basis': "Imperial decline theory, administrative capacity",
            'testable_mechanisms': [
                "Geographic distance → Communication delays → Policy failures",
                "Multiple commitments → Resource constraints → Governance breakdown", 
                "Bureaucratic incompetence → Arbitrary decisions → Popular grievances"
            ]
        },
        
        'military_catalyst': {
            'description': "French and Indian War transformed colonial military experience and created veteran leadership that enabled resistance",
            'key_predictions': [
                "Military veterans disproportionately lead resistance",
                "Military organization patterns in resistance groups",
                "Strategic thinking evident in colonial planning",
                "Military precedents for challenging British authority"
            ],
            'theoretical_basis': "Military sociology, veteran political mobilization",
            'testable_mechanisms': [
                "Military service → Leadership skills → Political leadership",
                "Veteran networks → Communication channels → Coordinated resistance",
                "Military confidence → Resistance feasibility → Revolutionary commitment"
            ]
        },
        
        'ideological_contagion': {
            'description': "Enlightenment ideas about natural rights and popular sovereignty spread through colonial intellectual networks and newspapers",
            'key_predictions': [
                "Philosophical arguments dominate political documents",
                "Intellectual networks connect resistance leaders",
                "Newspaper circulation correlates with political mobilization",
                "European intellectual influences clearly documented"
            ],
            'theoretical_basis': "Ideational diffusion theory, intellectual history",
            'testable_mechanisms': [
                "Print culture → Idea transmission → Political mobilization",
                "Intellectual networks → Coordinated arguments → Movement coherence",
                "Educational institutions → Elite formation → Leadership development"
            ]
        }
    }
    
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
        self.existing_hypotheses = [n for n in graph_data['nodes'] if n.get('type') == 'Hypothesis']
        self.existing_alternatives = [n for n in graph_data['nodes'] if n.get('type') == 'Alternative_Explanation']
        
    def generate_comprehensive_alternatives(self) -> Dict:
        """
        Generate comprehensive set of alternative hypotheses for systematic testing.
        Returns updated graph data with robust theoretical competition.
        """
        print("[ALTERNATIVE_GENERATION] Creating systematic theoretical competition...")
        
        new_nodes = []
        new_edges = []
        hypothesis_counter = len(self.existing_hypotheses) + len(self.existing_alternatives)
        
        for alt_id, alt_data in self.REVOLUTION_ALTERNATIVE_HYPOTHESES.items():
            # Create alternative hypothesis node
            hypothesis_counter += 1
            node_id = f"ALT_H_{hypothesis_counter:03d}"
            
            alternative_node = {
                'id': node_id,
                'type': 'Alternative_Explanation',
                'properties': {
                    'description': alt_data['description'],
                    'theoretical_basis': alt_data['theoretical_basis'],
                    'key_predictions': alt_data['key_predictions'],
                    'testable_mechanisms': alt_data['testable_mechanisms'],
                    'generated_by': 'alternative_hypothesis_generator',
                    'academic_quality': True
                }
            }
            new_nodes.append(alternative_node)
            
            # Create competitive relationships with existing hypotheses
            for existing_hyp in self.existing_hypotheses:
                # Add competition edge
                competition_edge = {
                    'source_id': node_id,
                    'target_id': existing_hyp['id'],
                    'type': 'competes_with',
                    'properties': {
                        'relationship_type': 'theoretical_competition',
                        'competition_strength': 0.8,
                        'testable_difference': True,
                        'elimination_potential': True
                    }
                }
                new_edges.append(competition_edge)
            
            # Connect to relevant evidence that could test this alternative
            relevant_evidence = self._identify_relevant_evidence(alt_data)
            for evidence_node in relevant_evidence:
                evidence_edge = {
                    'source_id': evidence_node['id'],
                    'target_id': node_id,
                    'type': 'potentially_supports',
                    'properties': {
                        'diagnostic_type': 'straw_in_wind',  # Default, will be refined
                        'probative_value': 0.6,
                        'requires_testing': True,
                        'theoretical_relevance': alt_data['theoretical_basis']
                    }
                }
                new_edges.append(evidence_edge)
        
        # Update graph data
        updated_graph = self.graph_data.copy()
        updated_graph['nodes'].extend(new_nodes)
        updated_graph['edges'].extend(new_edges)
        
        print(f"[ALTERNATIVE_GENERATION] Generated {len(new_nodes)} alternative hypotheses")
        print(f"[ALTERNATIVE_GENERATION] Created {len(new_edges)} new evidence/competition relationships")
        print(f"[ALTERNATIVE_GENERATION] Total alternative explanations: {len(new_nodes) + len(self.existing_alternatives)}")
        
        return updated_graph
    
    def _identify_relevant_evidence(self, alt_data: Dict) -> List[Dict]:
        """
        Identify evidence nodes that could potentially test this alternative hypothesis.
        Uses keyword matching and thematic relevance.
        """
        relevant_evidence = []
        evidence_nodes = [n for n in self.graph_data['nodes'] if n.get('type') == 'Evidence']
        
        # Extract keywords from alternative hypothesis
        description = alt_data['description'].lower()
        predictions = ' '.join(alt_data['key_predictions']).lower()
        search_text = f"{description} {predictions}"
        
        # Keyword matching for relevance
        relevance_keywords = {
            'economic': ['merchant', 'trade', 'profit', 'economic', 'commercial', 'business', 'money'],
            'generational': ['young', 'generation', 'age', 'youth', 'elder', 'father', 'son'],
            'religious': ['religious', 'church', 'clergy', 'god', 'christian', 'protestant', 'anglican'],
            'elite': ['elite', 'leader', 'wealthy', 'class', 'aristocrat', 'gentleman'],
            'cultural': ['culture', 'tradition', 'local', 'community', 'custom'],
            'military': ['military', 'war', 'soldier', 'veteran', 'army', 'battle'],
            'ideological': ['idea', 'philosophy', 'enlightenment', 'rights', 'liberty', 'freedom'],
            'administrative': ['administration', 'policy', 'government', 'bureaucrat', 'official']
        }
        
        for evidence_node in evidence_nodes:
            evidence_desc = evidence_node.get('properties', {}).get('description', '').lower()
            
            # Check for keyword overlap
            # Use semantic analysis to determine evidence relevance
            from core.semantic_analysis_service import get_semantic_service
            semantic_service = get_semantic_service()
            
            assessment = semantic_service.assess_probative_value(
                evidence_description=evidence_desc,
                hypothesis_description=f"Evidence is relevant to: {search_text}",
                context="Determining evidence relevance for alternative hypothesis generation"
            )
            
            # Include evidence with sufficient relevance
            if assessment.confidence_score >= 0.6:
                relevant_evidence.append(evidence_node)
        
        return relevant_evidence[:10]  # Limit to top 10 most relevant pieces of evidence

# Integration function
def generate_alternative_hypotheses(graph_data: Dict) -> Dict:
    """
    Main entry point for alternative hypothesis generation.
    Returns updated graph data with comprehensive theoretical competition.
    """
    generator = AlternativeHypothesisGenerator(graph_data)
    return generator.generate_comprehensive_alternatives()
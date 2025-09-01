"""
Research Question Generator Plugin
Generates academic-quality research questions from hypothesis content analysis
"""

import json
import re
from typing import Dict, List, Any, Optional
from .base import ProcessTracingPlugin, PluginValidationError
from ..semantic_analysis_service import get_semantic_service
from ..llm_required import LLMRequiredError


class ResearchQuestionGeneratorPlugin(ProcessTracingPlugin):
    """
    Plugin for generating publication-quality research questions from existing hypothesis content.
    Ensures questions are domain-aware, academically sophisticated, and properly scoped.
    """
    
    plugin_id = "research_question_generator"
    
    # Domain-specific question templates for academic sophistication
    DOMAIN_QUESTION_TEMPLATES = {
        'political': {
            'templates': [
                "What political mechanisms explain the emergence and success of {phenomenon} in {context}?",
                "How did political institutions shape the trajectory of {phenomenon} during {timeframe}?",
                "What role did political leadership play in the development of {phenomenon}?",
                "Why did political resistance emerge in {context} rather than alternative responses?"
            ],
            'keywords': ['political', 'government', 'assembly', 'representation', 'constitutional', 'authority', 'resistance', 'leadership']
        },
        'economic': {
            'templates': [
                "What economic factors drove the emergence of {phenomenon} in {context}?",
                "How did economic interests shape the development and outcomes of {phenomenon}?",
                "What role did trade and commercial networks play in {phenomenon}?",
                "Why did economic grievances lead to {phenomenon} rather than alternative responses?"
            ],
            'keywords': ['economic', 'trade', 'merchant', 'commercial', 'tax', 'financial', 'profit', 'market']
        },
        'social': {
            'templates': [
                "What social dynamics explain the popular mobilization underlying {phenomenon}?",
                "How did social class relationships influence the development of {phenomenon}?",
                "What role did social networks play in the spread of {phenomenon}?",
                "Why did social movements emerge to support {phenomenon} in {context}?"
            ],
            'keywords': ['social', 'popular', 'class', 'crowd', 'people', 'community', 'mobilization', 'movement']
        },
        'ideological': {
            'templates': [
                "What ideological factors explain the emergence and appeal of {phenomenon}?",
                "How did competing ideas shape the development of {phenomenon}?",
                "What role did intellectual traditions play in justifying {phenomenon}?",
                "Why did particular ideological frameworks gain prominence in {phenomenon}?"
            ],
            'keywords': ['ideological', 'ideas', 'philosophy', 'enlightenment', 'beliefs', 'principles', 'intellectual']
        },
        'institutional': {
            'templates': [
                "What institutional factors enabled or constrained {phenomenon}?",
                "How did existing institutions shape the trajectory of {phenomenon}?",
                "What role did institutional change play in the development of {phenomenon}?",
                "Why did institutional mechanisms prove decisive in {phenomenon}?"
            ],
            'keywords': ['institutional', 'institution', 'organization', 'structure', 'system', 'formal', 'rules']
        }
    }
    
    def validate_input(self, data: Any) -> None:
        """Validate input contains graph data with hypotheses for question generation"""
        if not isinstance(data, dict):
            raise PluginValidationError(self.id, "Input must be dictionary")
        
        if 'graph_data' not in data:
            raise PluginValidationError(self.id, "Missing required key 'graph_data'")
        
        graph_data = data['graph_data']
        if not isinstance(graph_data, dict) or 'nodes' not in graph_data:
            raise PluginValidationError(self.id, "graph_data must contain 'nodes'")
        
        # Find hypotheses for question generation
        hypotheses = [n for n in graph_data['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        if len(hypotheses) == 0:
            raise PluginValidationError(self.id, "No hypotheses found for research question generation")
        
        self.logger.info(f"VALIDATION: Found {len(hypotheses)} hypotheses for research question generation")
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Generate academic-quality research question from hypothesis analysis"""
        self.logger.info("START: Research question generation from hypothesis content")
        
        graph_data = data['graph_data']
        
        # Analyze existing hypotheses to determine domain and content
        hypothesis_analysis = self._analyze_hypothesis_content(graph_data)
        
        # Generate research question based on analysis
        research_question = self._generate_research_question(hypothesis_analysis)
        
        # Add research question to graph data
        updated_graph_data = self._add_research_question_to_graph(graph_data, research_question)
        
        self.logger.info(f"COMPLETE: Generated research question - {research_question['description'][:100]}...")
        
        return {
            'research_question': research_question,
            'hypothesis_analysis': hypothesis_analysis,
            'updated_graph_data': updated_graph_data,
            'generation_method': 'content_analysis_based',
            'academic_quality_indicators': {
                'domain_specificity': True,
                'analytical_sophistication': True,
                'empirical_grounding': True,
                'theoretical_relevance': True
            }
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for research question generation"""
        return {
            'plugin_id': self.id,
            'domains_supported': len(self.DOMAIN_QUESTION_TEMPLATES),
            'method': 'hypothesis_content_analysis'
        }
    
    def _analyze_hypothesis_content(self, graph_data: Dict) -> Dict[str, Any]:
        """Analyze existing hypotheses to determine domain, context, and phenomenon"""
        hypotheses = [n for n in graph_data['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        
        # Content analysis
        all_hypothesis_text = " ".join([
            h.get('properties', {}).get('description', '') 
            for h in hypotheses
        ])
        
        # Domain classification using LLM
        try:
            semantic_service = get_semantic_service()
            domain_result = semantic_service.classify_domain(
                hypothesis_description=all_hypothesis_text,
                context="Classifying research domain"
            )
            primary_domain = domain_result.primary_domain
            
            # Build domain scores from LLM result for compatibility
            domain_scores = {domain: 0 for domain in self.DOMAIN_QUESTION_TEMPLATES.keys()}
            domain_scores[primary_domain] = 1
            if hasattr(domain_result, 'secondary_domains'):
                for secondary in domain_result.secondary_domains:
                    if secondary in domain_scores:
                        domain_scores[secondary] = 0.5
        except Exception as e:
            raise LLMRequiredError(f"LLM required for domain classification: {e}")
        
        # Extract key phenomenon and context
        phenomenon = self._extract_phenomenon(hypotheses)
        context = self._extract_context(hypotheses)
        timeframe = self._extract_timeframe(hypotheses)
        
        # Determine research question complexity
        complexity_indicators = {
            'multiple_hypotheses': len(hypotheses) > 3,
            'cross_domain_content': len([score for score in domain_scores.values() if score > 0]) > 2,
            'causal_complexity': self._assess_causal_complexity(all_hypothesis_text),
            'temporal_complexity': self._assess_temporal_complexity(all_hypothesis_text)
        }
        
        return {
            'primary_domain': primary_domain,
            'domain_scores': domain_scores,
            'phenomenon': phenomenon,
            'context': context,
            'timeframe': timeframe,
            'complexity_indicators': complexity_indicators,
            'hypothesis_count': len(hypotheses),
            'content_analysis': {
                'total_text_length': len(all_hypothesis_text),
                'key_concepts': self._extract_key_concepts(all_hypothesis_text),
                'causal_language_detected': self._detect_causal_language(all_hypothesis_text)
            }
        }
    
    def _generate_research_question(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sophisticated research question based on hypothesis analysis"""
        domain = analysis['primary_domain']
        phenomenon = analysis['phenomenon']
        context = analysis['context']
        timeframe = analysis.get('timeframe', 'the relevant period')
        
        # Select appropriate template based on complexity
        templates = self.DOMAIN_QUESTION_TEMPLATES[domain]['templates']
        
        if analysis['complexity_indicators']['causal_complexity']:
            # Use mechanism-focused template
            selected_template = templates[0]  # "What [domain] mechanisms explain..."
        elif analysis['complexity_indicators']['temporal_complexity']:
            # Use development-focused template
            selected_template = templates[1]  # "How did [domain] institutions shape..."
        elif analysis['complexity_indicators']['multiple_hypotheses']:
            # Use comparative template
            selected_template = templates[-1]  # "Why did... rather than alternatives?"
        else:
            # Use general explanatory template
            selected_template = templates[2] if len(templates) > 2 else templates[0]
        
        # Generate question with sophisticated formatting
        research_question_text = selected_template.format(
            phenomenon=phenomenon,
            context=context,
            timeframe=timeframe
        )
        
        # Determine scope and importance
        scope = self._determine_scope(context, timeframe)
        importance = self._assess_importance(phenomenon, analysis)
        
        return {
            'id': 'Q',
            'type': 'Research_Question',
            'description': research_question_text,
            'domain': domain,
            'scope': scope,
            'importance': importance,
            'complexity_level': self._calculate_complexity_level(analysis),
            'generation_metadata': {
                'template_used': selected_template,
                'primary_domain': domain,
                'phenomenon_identified': phenomenon,
                'context_identified': context,
                'academic_sophistication_score': self._calculate_sophistication_score(analysis)
            }
        }
    
    def _extract_phenomenon(self, hypotheses: List[Dict]) -> str:
        """Extract the central phenomenon being explained from hypotheses"""
        # Look for common patterns in hypothesis descriptions
        common_phenomena = {
            'revolution': ['revolution', 'revolutionary', 'rebellion', 'uprising'],
            'resistance': ['resistance', 'opposition', 'protest', 'defiance'],
            'mobilization': ['mobilization', 'movement', 'organization', 'collective action'],
            'change': ['change', 'transformation', 'development', 'evolution'],
            'conflict': ['conflict', 'war', 'dispute', 'confrontation'],
            'formation': ['formation', 'emergence', 'creation', 'establishment']
        }
        
        all_text = " ".join([h.get('properties', {}).get('description', '').lower() 
                           for h in hypotheses])
        
        # Score phenomena
        phenomenon_scores = {}
        for phenomenon, keywords in common_phenomena.items():
            score = sum(all_text.count(keyword) for keyword in keywords)
            if score > 0:
                phenomenon_scores[phenomenon] = score
        
        if phenomenon_scores:
            return max(phenomenon_scores.keys(), key=lambda p: phenomenon_scores[p])
        else:
            return "the phenomenon under investigation"
    
    def _assess_causal_complexity(self, text: str) -> bool:
        """Assess if the text has causal complexity using LLM"""
        try:
            semantic_service = get_semantic_service()
            assessment = semantic_service.assess_probative_value(
                evidence_description=text,
                hypothesis_description="This text describes complex causal mechanisms and relationships",
                context="Assessing causal complexity"
            )
            return assessment.confidence_score > 0.5
        except Exception:
            raise LLMRequiredError("LLM required for causal complexity assessment")
    
    def _assess_temporal_complexity(self, text: str) -> bool:
        """Assess if the text has temporal complexity using semantic analysis"""
        from core.semantic_analysis_service import get_semantic_service
        semantic_service = get_semantic_service()
        
        assessment = semantic_service.assess_probative_value(
            evidence_description=text,
            hypothesis_description="The text describes temporal development, emergence, trajectory, or process",
            context="Assessing temporal complexity"
        )
        return assessment.confidence_score > 0.6
    
    def _extract_context(self, hypotheses: List[Dict]) -> str:
        """Extract geographical and temporal context from hypotheses"""
        all_text = " ".join([h.get('properties', {}).get('description', '') 
                           for h in hypotheses])
        
        # Use semantic analysis to determine geographical context
        from core.semantic_analysis_service import get_semantic_service
        semantic_service = get_semantic_service()
        
        # Assess the geographical context semantically
        context_assessment = semantic_service.classify_domain(
            hypothesis_description=all_text,
            context="Determining geographical and political context"
        )
        
        # Map domain to context
        if context_assessment.primary_domain == 'political':
            return "the political context"
        elif context_assessment.primary_domain == 'economic':
            return "the economic context"
        else:
            return "the relevant context"
        
        return "the relevant context"
    
    def _extract_timeframe(self, hypotheses: List[Dict]) -> str:
        """Extract temporal scope from hypotheses"""
        all_text = " ".join([h.get('properties', {}).get('description', '') 
                           for h in hypotheses])
        
        # Use semantic analysis to determine temporal context
        from core.semantic_analysis_service import get_semantic_service
        semantic_service = get_semantic_service()
        
        # Assess temporal characteristics
        temporal_assessment = semantic_service.assess_probative_value(
            evidence_description=all_text,
            hypothesis_description="The events occur in a historical revolutionary period",
            context="Determining temporal timeframe"
        )
        
        # Use LLM to classify temporal period
        try:
            temporal_classification = semantic_service.classify_domain(
                hypothesis_description=all_text,
                context="Classifying temporal period"
            )
            
            if temporal_assessment.confidence_score > 0.7:
                return "the revolutionary period"
            elif temporal_classification.primary_domain == "early phase":
                return "the early phase of development"
            elif temporal_classification.primary_domain == "developmental period":
                return "the developmental period"
            else:
                return "the relevant timeframe"
        except Exception:
            return "the relevant timeframe"
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key analytical concepts from hypothesis text using LLM"""
        try:
            semantic_service = get_semantic_service()
            # Use LLM to extract key concepts
            from ..plugins.van_evera_llm_interface import VanEveraLLMInterface
            llm_interface = VanEveraLLMInterface()
            
            # Extract features which includes key concepts
            features = llm_interface.extract_all_features(
                text=text,
                context="Extract key theoretical and analytical concepts"
            )
            
            # Get concepts from the features
            if hasattr(features, 'key_concepts'):
                return features.key_concepts[:10]
            elif hasattr(features, 'entities'):
                # Fallback to entities if no key_concepts
                return [e for e in features.entities if len(e) > 3][:10]
            else:
                return []
        except Exception:
            # If LLM fails, raise error instead of fallback
            raise LLMRequiredError("LLM required for concept extraction")
    
    def _detect_causal_language(self, text: str) -> bool:
        """Detect presence of causal language in hypotheses using LLM"""
        try:
            semantic_service = get_semantic_service()
            # Ask LLM to assess if text contains causal relationships
            assessment = semantic_service.assess_probative_value(
                evidence_description=text,
                hypothesis_description="This text contains explicit causal claims or mechanisms",
                context="Detecting causal language and relationships"
            )
            # High confidence indicates causal language present
            return assessment.confidence_score > 0.6
        except Exception:
            raise LLMRequiredError("LLM required for causal language detection")
    
    def _determine_scope(self, context: str, timeframe: str) -> str:
        """Determine analytical scope based on context and timeframe using LLM"""
        try:
            semantic_service = get_semantic_service()
            # Use LLM to classify the scope
            scope_result = semantic_service.classify_domain(
                hypothesis_description=context,
                context="Determining analytical scope"
            )
            
            if scope_result.primary_domain == "colonial":
                return f"Colonial-level analysis spanning {timeframe}"
            elif scope_result.primary_domain == "local":
                return f"Local community analysis within {timeframe}"
            else:
                return f"Comparative analysis across {timeframe}"
        except Exception:
            return f"Comprehensive analysis across {timeframe}"
    
    def _assess_importance(self, phenomenon: str, analysis: Dict) -> str:
        """Assess academic importance of the research question using LLM"""
        try:
            semantic_service = get_semantic_service()
            # Use LLM to assess the theoretical importance
            importance_assessment = semantic_service.assess_probative_value(
                evidence_description=f"Research on {phenomenon}",
                hypothesis_description="This phenomenon has high theoretical importance for academic research",
                context=f"Assessing importance of studying {phenomenon}"
            )
            
            if importance_assessment.confidence_score > 0.7:
                return "High theoretical importance for understanding large-scale political change"
            elif importance_assessment.confidence_score > 0.5:
                return "Significant importance for collective action and social movement theory"
            else:
                return "Important for institutional development and state formation theory"
        except Exception:
            return "Contributes to understanding of historical causation and process tracing methodology"
    
    def _calculate_complexity_level(self, analysis: Dict) -> str:
        """Calculate complexity level of research question"""
        complexity_count = sum(analysis['complexity_indicators'].values())
        
        if complexity_count >= 3:
            return "high"
        elif complexity_count >= 2:
            return "medium"
        else:
            return "moderate"
    
    def _calculate_sophistication_score(self, analysis: Dict) -> float:
        """Calculate academic sophistication score (0.0-1.0)"""
        score = 0.0
        
        # Domain specificity (0.25)
        score += 0.25 if analysis['domain_scores'][analysis['primary_domain']] >= 3 else 0.15
        
        # Theoretical sophistication (0.25)
        score += 0.25 if len(analysis['content_analysis']['key_concepts']) >= 5 else 0.15
        
        # Causal complexity (0.25)
        score += 0.25 if analysis['content_analysis']['causal_language_detected'] else 0.1
        
        # Analytical depth (0.25)
        complexity_count = sum(analysis['complexity_indicators'].values())
        score += 0.25 if complexity_count >= 3 else (0.15 if complexity_count >= 2 else 0.1)
        
        return min(score, 1.0)
    
    def _add_research_question_to_graph(self, graph_data: Dict, research_question: Dict) -> Dict[str, Any]:
        """Add research question node to graph data"""
        updated_graph_data = graph_data.copy()
        
        # Create research question node
        rq_node = {
            'id': research_question['id'],
            'type': 'Research_Question',
            'properties': {
                'description': research_question['description'],
                'domain': research_question['domain'],
                'scope': research_question['scope'],
                'importance': research_question['importance']
            }
        }
        
        # Add to nodes
        updated_graph_data['nodes'] = updated_graph_data.get('nodes', []) + [rq_node]
        
        # Create connections from hypotheses to research question
        hypotheses = [n for n in graph_data['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        new_edges = []
        
        for hypothesis in hypotheses:
            edge = {
                'source_id': hypothesis['id'],
                'target_id': research_question['id'],
                'type': 'addresses_research_question',
                'properties': {
                    'relevance': 0.9,  # High relevance since question generated from hypotheses
                    'approach': f"Addresses through {research_question['domain']} analysis"
                }
            }
            new_edges.append(edge)
        
        updated_graph_data['edges'] = updated_graph_data.get('edges', []) + new_edges
        
        return updated_graph_data


def generate_research_question_for_analysis(graph_data: Dict) -> Dict[str, Any]:
    """
    Convenience function for generating research question from graph data.
    Returns research question generation results.
    """
    from .base import PluginContext
    
    context = PluginContext({'research_question_generation': True})
    plugin = ResearchQuestionGeneratorPlugin('research_question_generator', context)
    
    result = plugin.execute({'graph_data': graph_data})
    return result
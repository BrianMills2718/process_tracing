"""
Evidence Document System for Pre-Analysis Optimization

Pre-analyzes evidence documents once, then efficiently evaluates them
against multiple hypotheses without redundant analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from core.plugins.van_evera_llm_schemas import (
    ComprehensiveEvidenceAnalysis,
    MultiFeatureExtraction
)


@dataclass
class HypothesisEvaluation:
    """
    Lightweight evaluation result for evidence against a specific hypothesis.
    """
    hypothesis_id: str
    hypothesis_text: str
    relationship_type: str  # supports/contradicts/neutral/ambiguous
    confidence: float
    van_evera_diagnostic: str
    reasoning: str
    evaluation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvidenceDocument:
    """
    Pre-analyzed evidence document with queryable features.
    Allows efficient evaluation against multiple hypotheses.
    """
    
    id: str
    text: str
    source: Optional[str] = None
    comprehensive_analysis: Optional[ComprehensiveEvidenceAnalysis] = None
    feature_extraction: Optional[MultiFeatureExtraction] = None
    feature_index: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: Optional[datetime] = None
    hypothesis_evaluations: Dict[str, HypothesisEvaluation] = field(default_factory=dict)
    
    def analyze_once(self, semantic_service) -> ComprehensiveEvidenceAnalysis:
        """
        One-time comprehensive analysis of evidence.
        Extracts all features for future use.
        
        Args:
            semantic_service: SemanticAnalysisService instance
            
        Returns:
            Comprehensive analysis results
        """
        if not self.comprehensive_analysis:
            # Extract all features in one go
            self.feature_extraction = semantic_service.extract_all_features(
                self.text,
                context=f"Evidence document: {self.id}"
            )
            
            # Build a dummy hypothesis for standalone analysis
            # This gives us domain, quality, and feature information
            dummy_hypothesis = "General hypothesis for feature extraction"
            self.comprehensive_analysis = semantic_service.analyze_comprehensive(
                self.text,
                dummy_hypothesis,
                context="Standalone evidence analysis"
            )
            
            self.analysis_timestamp = datetime.now()
            self._build_feature_index()
            
        return self.comprehensive_analysis
    
    def evaluate_against_hypotheses_batch(self,
                                         hypotheses: List[Dict[str, str]],
                                         semantic_service) -> Dict[str, HypothesisEvaluation]:
        """
        Evaluate evidence against ALL hypotheses in a single LLM call.
        This is the LLM-first approach that provides the best quality.
        
        Args:
            hypotheses: List of dicts with 'id' and 'text' keys
            semantic_service: SemanticAnalysisService instance
            
        Returns:
            Dict mapping hypothesis_id to HypothesisEvaluation
        """
        # Filter out already evaluated hypotheses
        new_hypotheses = [h for h in hypotheses 
                         if h['id'] not in self.hypothesis_evaluations]
        
        if not new_hypotheses:
            # All hypotheses already evaluated
            return {h['id']: self.hypothesis_evaluations[h['id']] 
                   for h in hypotheses}
        
        # Get batch evaluation from LLM
        batch_result = semantic_service.evaluate_evidence_against_hypotheses_batch(
            self.id,
            self.text,
            new_hypotheses,
            context="Process tracing batch evaluation"
        )
        
        # Process batch results into individual evaluations
        for eval_result in batch_result.evaluations:
            evaluation = HypothesisEvaluation(
                hypothesis_id=eval_result.hypothesis_id,
                hypothesis_text=next(h['text'] for h in new_hypotheses 
                                   if h['id'] == eval_result.hypothesis_id),
                relationship_type=eval_result.relationship_type,
                confidence=eval_result.confidence,
                van_evera_diagnostic=eval_result.van_evera_diagnostic,
                reasoning=eval_result.reasoning
            )
            self.hypothesis_evaluations[eval_result.hypothesis_id] = evaluation
        
        # Return all requested evaluations
        return {h['id']: self.hypothesis_evaluations[h['id']] 
               for h in hypotheses}
    
    def evaluate_against_hypothesis(self, 
                                   hypothesis_id: str,
                                   hypothesis_text: str,
                                   semantic_service) -> HypothesisEvaluation:
        """
        DEPRECATED: Use evaluate_against_hypotheses_batch for better quality.
        Kept for backward compatibility.
        """
        # Check if already evaluated
        if hypothesis_id in self.hypothesis_evaluations:
            return self.hypothesis_evaluations[hypothesis_id]
        
        # Use batch evaluation with single hypothesis
        hypotheses = [{'id': hypothesis_id, 'text': hypothesis_text}]
        results = self.evaluate_against_hypotheses_batch(hypotheses, semantic_service)
        return results[hypothesis_id]
    
    def _build_feature_index(self):
        """
        Build searchable index of features for quick access.
        """
        if self.comprehensive_analysis:
            self.feature_index = {
                'domain': self.comprehensive_analysis.primary_domain,
                'secondary_domains': self.comprehensive_analysis.secondary_domains,
                'key_concepts': self.comprehensive_analysis.key_concepts,
                'probative_value': self.comprehensive_analysis.probative_value,
                'evidence_quality': self.comprehensive_analysis.evidence_quality
            }
        
        if self.feature_extraction:
            self.feature_index.update({
                'actors': self.feature_extraction.primary_actors,
                'mechanisms': self.feature_extraction.mechanisms,
                'temporal': self.feature_extraction.temporal_sequence,
                'concepts': self.feature_extraction.key_concepts
            })
    
    def get_actors(self) -> List[str]:
        """Get all actors mentioned in the evidence."""
        if 'actors' in self.feature_index:
            return self.feature_index['actors']
        return []
    
    def get_mechanisms(self) -> List[Dict[str, str]]:
        """Get all causal mechanisms identified."""
        if 'mechanisms' in self.feature_index:
            return self.feature_index['mechanisms']
        return []
    
    def get_temporal_markers(self) -> List[Dict[str, str]]:
        """Get all temporal markers and sequences."""
        # This is a structural check for dictionary key, not semantic matching
        # The key 'temporal' is a field name, not content analysis
        if 'temporal' in self.feature_index:
            return self.feature_index['temporal']
        return []
    
    def get_domain(self) -> str:
        """Get primary domain classification."""
        return self.feature_index.get('domain', 'unknown')
    
    def get_quality_score(self) -> float:
        """Get evidence quality score."""
        return self.feature_index.get('probative_value', 0.5)
    
    def has_actor(self, actor_name: str) -> bool:
        """Check if a specific actor is mentioned."""
        # Note: This should use semantic similarity in a full implementation
        # but since actors are pre-extracted entities, exact match is acceptable
        actors = self.get_actors()
        return any(actor_name.lower() in actor.lower() for actor in actors)
    
    def has_concept(self, concept: str) -> bool:
        """Check if a specific concept is present."""
        # Note: This should use semantic similarity in a full implementation
        # but since concepts are pre-extracted entities, exact match is acceptable
        concepts = self.feature_index.get('concepts', [])
        return any(concept.lower() in c.lower() for c in concepts)
    
    def get_all_hypothesis_evaluations(self) -> Dict[str, HypothesisEvaluation]:
        """Get all cached hypothesis evaluations."""
        return self.hypothesis_evaluations
    
    def clear_evaluations(self):
        """Clear cached hypothesis evaluations (useful if evidence is updated)."""
        self.hypothesis_evaluations.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'text': self.text,
            'source': self.source,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'feature_index': self.feature_index,
            'hypothesis_evaluations': {
                h_id: {
                    'relationship_type': eval.relationship_type,
                    'confidence': eval.confidence,
                    'van_evera_diagnostic': eval.van_evera_diagnostic,
                    'reasoning': eval.reasoning
                }
                for h_id, eval in self.hypothesis_evaluations.items()
            }
        }


class EvidenceCorpus:
    """
    Collection of pre-analyzed evidence documents.
    Manages batch processing and efficient hypothesis testing.
    """
    
    def __init__(self):
        """Initialize empty corpus."""
        self.documents: Dict[str, EvidenceDocument] = {}
        self.analysis_stats = {
            'total_documents': 0,
            'analyzed_documents': 0,
            'total_evaluations': 0,
            'cache_hits': 0
        }
    
    def add_document(self, doc_id: str, text: str, source: Optional[str] = None) -> EvidenceDocument:
        """
        Add a new evidence document to the corpus.
        
        Args:
            doc_id: Unique identifier
            text: Evidence text
            source: Optional source information
            
        Returns:
            Created EvidenceDocument
        """
        doc = EvidenceDocument(id=doc_id, text=text, source=source)
        self.documents[doc_id] = doc
        self.analysis_stats['total_documents'] += 1
        return doc
    
    def pre_analyze_all(self, semantic_service):
        """
        Pre-analyze all documents in the corpus.
        
        Args:
            semantic_service: SemanticAnalysisService instance
        """
        for doc in self.documents.values():
            if not doc.comprehensive_analysis:
                doc.analyze_once(semantic_service)
                self.analysis_stats['analyzed_documents'] += 1
    
    def evaluate_hypothesis(self, hypothesis_id: str, hypothesis_text: str, semantic_service) -> Dict[str, HypothesisEvaluation]:
        """
        Evaluate a hypothesis against all evidence in the corpus.
        
        Args:
            hypothesis_id: Unique hypothesis identifier
            hypothesis_text: The hypothesis text
            semantic_service: SemanticAnalysisService instance
            
        Returns:
            Dictionary of document_id -> evaluation
        """
        results = {}
        
        for doc_id, doc in self.documents.items():
            # Check if we've already evaluated this hypothesis for this document
            if hypothesis_id in doc.hypothesis_evaluations:
                results[doc_id] = doc.hypothesis_evaluations[hypothesis_id]
                self.analysis_stats['cache_hits'] += 1
            else:
                evaluation = doc.evaluate_against_hypothesis(
                    hypothesis_id, hypothesis_text, semantic_service
                )
                results[doc_id] = evaluation
                self.analysis_stats['total_evaluations'] += 1
        
        return results
    
    def get_supporting_evidence(self, hypothesis_id: str) -> List[str]:
        """
        Get IDs of documents that support a hypothesis.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            List of supporting document IDs
        """
        supporting = []
        for doc_id, doc in self.documents.items():
            if hypothesis_id in doc.hypothesis_evaluations:
                eval = doc.hypothesis_evaluations[hypothesis_id]
                if eval.relationship_type == "supports" and eval.confidence > 0.6:
                    supporting.append(doc_id)
        return supporting
    
    def get_contradicting_evidence(self, hypothesis_id: str) -> List[str]:
        """
        Get IDs of documents that contradict a hypothesis.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            List of contradicting document IDs
        """
        contradicting = []
        for doc_id, doc in self.documents.items():
            if hypothesis_id in doc.hypothesis_evaluations:
                eval = doc.hypothesis_evaluations[hypothesis_id]
                if eval.relationship_type == "contradicts" and eval.confidence > 0.6:
                    contradicting.append(doc_id)
        return contradicting
    
    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        return self.analysis_stats
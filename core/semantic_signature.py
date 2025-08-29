"""
Semantic Signature System for Smart Caching

Creates semantic fingerprints of content for intelligent caching that works
even when text is paraphrased or slightly modified.
"""

import hashlib
import json
import re
from typing import List, Set
from collections import Counter


class SemanticSignature:
    """
    Creates cacheable semantic fingerprints of content.
    Allows semantically similar texts to generate the same cache key.
    """
    
    def __init__(self):
        """Initialize with common stop words to filter."""
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "can", "will", "just", "should", "now", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "could", "would", "might", "must", "shall", "may"
        }
    
    def generate_signature(self, 
                          evidence: str, 
                          hypothesis: str,
                          operation: str) -> str:
        """
        Create semantic signature that captures meaning, not exact text.
        Similar texts will generate the same signature.
        
        Args:
            evidence: Evidence text
            hypothesis: Hypothesis text
            operation: Type of operation being performed
            
        Returns:
            Hexadecimal hash signature
        """
        # Normalize texts
        norm_evidence = self._normalize(evidence)
        norm_hypothesis = self._normalize(hypothesis)
        
        # Extract semantic components
        evidence_keywords = self._extract_keywords(norm_evidence, top_n=10)
        hypothesis_keywords = self._extract_keywords(norm_hypothesis, top_n=10)
        
        # Create structured signature
        signature_components = [
            operation,  # What analysis is being done
            sorted(evidence_keywords),  # Core evidence concepts
            sorted(hypothesis_keywords),  # Core hypothesis concepts
            len(norm_evidence) // 100,  # Size bucket (100-char buckets)
            len(norm_hypothesis) // 100
        ]
        
        # Hash the semantic signature
        return hashlib.sha256(
            json.dumps(signature_components, sort_keys=True).encode()
        ).hexdigest()
    
    def generate_text_signature(self, text: str, operation: str) -> str:
        """
        Generate signature for single text (for feature extraction).
        
        Args:
            text: Text to analyze
            operation: Type of operation
            
        Returns:
            Hexadecimal hash signature
        """
        norm_text = self._normalize(text)
        keywords = self._extract_keywords(norm_text, top_n=15)
        
        signature_components = [
            operation,
            sorted(keywords),
            len(norm_text) // 100
        ]
        
        return hashlib.sha256(
            json.dumps(signature_components, sort_keys=True).encode()
        ).hexdigest()
    
    def _normalize(self, text: str) -> str:
        """
        Normalize text for semantic comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Lowercase and remove extra whitespace
        text = " ".join(text.lower().split())
        
        # Remove punctuation except sentence boundaries
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract top N meaningful keywords using simple term frequency.
        
        Args:
            text: Normalized text
            top_n: Number of keywords to extract
            
        Returns:
            List of top keywords
        """
        # Split into words
        words = text.split()
        
        # Filter stop words and short words
        words = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        # Get word frequencies
        word_freq = Counter(words)
        
        # Return top N words
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def calculate_similarity(self, sig1: str, sig2: str) -> float:
        """
        Calculate similarity between two signatures (for debugging).
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # For exact hash comparison, signatures are either same (1.0) or different (0.0)
        return 1.0 if sig1 == sig2 else 0.0
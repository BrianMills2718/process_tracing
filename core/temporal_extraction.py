"""
Temporal Process Tracing - Temporal Entity Extraction Module

Extracts temporal information from text for process tracing analysis,
including dates, temporal relationships, and sequence indicators.

Author: Claude Code Implementation  
Date: August 2025
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

class TemporalType(Enum):
    ABSOLUTE = "absolute"  # Specific dates/times
    RELATIVE = "relative"  # "after", "before", "during"
    DURATION = "duration"  # "for 3 months", "lasted 2 years"
    SEQUENCE = "sequence"  # "first", "then", "finally"
    UNCERTAIN = "uncertain"  # "around", "approximately"

class TemporalRelation(Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    CONCURRENT = "concurrent"
    OVERLAPS = "overlaps"
    PRECEDES = "precedes"
    FOLLOWS = "follows"

@dataclass
class TemporalExpression:
    """Represents a temporal expression extracted from text"""
    text: str
    temporal_type: TemporalType
    normalized_value: Optional[datetime]
    duration: Optional[timedelta]
    uncertainty: float  # 0.0 = certain, 1.0 = very uncertain
    confidence: float   # 0.0 = low confidence, 1.0 = high confidence
    source_span: Tuple[int, int]  # Character positions in original text
    context: str  # Surrounding text for context

@dataclass
class TemporalRelationship:
    """Represents a temporal relationship between events"""
    event1: str
    event2: str
    relation: TemporalRelation
    confidence: float
    evidence_text: str
    temporal_gap: Optional[timedelta] = None

class TemporalExtractor:
    """
    Extracts temporal information from text using pattern matching
    and LLM-assisted analysis for complex temporal expressions.
    """
    
    def __init__(self):
        # Compile regex patterns for temporal expressions
        self.absolute_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # dates
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
        ]
        
        self.relative_patterns = [
            r'\b(?:before|after|during|while|when|since|until|by)\s+',
            r'\b(?:earlier|later|previously|subsequently|afterwards|meanwhile)\b',
            r'\b(?:prior to|following|in the aftermath of|in response to)\b',
        ]
        
        self.duration_patterns = [
            r'\b(?:for|lasted|took|over|within|in)\s+(?:\d+\s+)?(?:days?|weeks?|months?|years?|hours?|minutes?)\b',
            r'\b\d+\s*(?:day|week|month|year|hour|minute)s?\b',
            r'\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|week|month|year)s?\b',
        ]
        
        self.sequence_patterns = [
            r'\b(?:first|second|third|fourth|fifth|initially|then|next|subsequently|finally|lastly)\b',
            r'\b(?:step \d+|phase \d+|stage \d+)\b',
            r'\b(?:beginning|start|onset|commencement|conclusion|end|termination)\b',
        ]
        
        # Compile patterns
        self.compiled_patterns = {
            TemporalType.ABSOLUTE: [re.compile(p, re.IGNORECASE) for p in self.absolute_patterns],
            TemporalType.RELATIVE: [re.compile(p, re.IGNORECASE) for p in self.relative_patterns],
            TemporalType.DURATION: [re.compile(p, re.IGNORECASE) for p in self.duration_patterns],
            TemporalType.SEQUENCE: [re.compile(p, re.IGNORECASE) for p in self.sequence_patterns],
        }
    
    def extract_temporal_expressions(self, text: str) -> List[TemporalExpression]:
        """
        Extract all temporal expressions from text using pattern matching.
        """
        expressions = []
        
        for temp_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    expression = self._create_temporal_expression(
                        match, temp_type, text
                    )
                    if expression:
                        expressions.append(expression)
        
        # Remove duplicates and sort by position
        expressions = self._deduplicate_expressions(expressions)
        expressions.sort(key=lambda x: x.source_span[0])
        
        return expressions
    
    def _create_temporal_expression(self, match: re.Match, temp_type: TemporalType, text: str) -> Optional[TemporalExpression]:
        """Create a TemporalExpression from a regex match"""
        try:
            matched_text = match.group()
            start, end = match.span()
            
            # Get context (50 characters before and after)
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end]
            
            # Normalize the temporal expression
            normalized_value, duration, uncertainty, confidence = self._normalize_temporal_expression(
                matched_text, temp_type, context
            )
            
            return TemporalExpression(
                text=matched_text,
                temporal_type=temp_type,
                normalized_value=normalized_value,
                duration=duration,
                uncertainty=uncertainty,
                confidence=confidence,
                source_span=(start, end),
                context=context
            )
        except Exception:
            return None
    
    def _normalize_temporal_expression(self, text: str, temp_type: TemporalType, context: str) -> Tuple[Optional[datetime], Optional[timedelta], float, float]:
        """
        Normalize temporal expression to standard format.
        Returns: (normalized_datetime, duration, uncertainty, confidence)
        """
        normalized_value = None
        duration = None
        uncertainty = 0.0
        confidence = 0.8  # Default confidence
        
        if temp_type == TemporalType.ABSOLUTE:
            normalized_value, uncertainty = self._parse_absolute_date(text)
            confidence = 0.9 if normalized_value else 0.3
            
        elif temp_type == TemporalType.DURATION:
            duration, uncertainty = self._parse_duration(text)
            confidence = 0.8 if duration else 0.4
            
        elif temp_type == TemporalType.RELATIVE:
            uncertainty = 0.3  # Relative expressions are inherently uncertain
            confidence = 0.7
            
        elif temp_type == TemporalType.SEQUENCE:
            uncertainty = 0.2  # Sequence indicators are usually clear
            confidence = 0.8
        
        # Adjust confidence based on context
        if any(word in context.lower() for word in ['approximately', 'around', 'roughly', 'about']):
            uncertainty += 0.3
            confidence -= 0.2
        
        # Ensure values are in valid ranges
        uncertainty = min(1.0, max(0.0, uncertainty))
        confidence = min(1.0, max(0.0, confidence))
        
        return normalized_value, duration, uncertainty, confidence
    
    def _parse_absolute_date(self, text: str) -> Tuple[Optional[datetime], float]:
        """Parse absolute date expressions"""
        uncertainty = 0.0
        
        # Try different date formats
        formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%B %d, %Y', '%B %d %Y', '%d %B %Y',
            '%b %d, %Y', '%b %d %Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt), uncertainty
            except ValueError:
                continue
        
        # Try parsing with natural language
        return self._parse_natural_date(text)
    
    def _parse_natural_date(self, text: str) -> Tuple[Optional[datetime], float]:
        """Parse natural language date expressions"""
        text_lower = text.lower()
        uncertainty = 0.2  # Natural language dates are less precise
        
        # Handle common natural language patterns
        if 'early' in text_lower:
            uncertainty += 0.2
        elif 'late' in text_lower:
            uncertainty += 0.2
        elif 'mid' in text_lower or 'middle' in text_lower:
            uncertainty += 0.1
        
        # For now, return None - would need more sophisticated NLP
        return None, uncertainty
    
    def _parse_duration(self, text: str) -> Tuple[Optional[timedelta], float]:
        """Parse duration expressions"""
        uncertainty = 0.1
        
        # Extract numbers and units
        number_match = re.search(r'\b(\d+)\b', text)
        if not number_match:
            # Handle written numbers
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'a': 1, 'an': 1
            }
            for word, num in word_to_num.items():
                if word in text.lower():
                    number = num
                    break
            else:
                return None, uncertainty
        else:
            number = int(number_match.group(1))
        
        # Determine unit
        text_lower = text.lower()
        if 'year' in text_lower:
            return timedelta(days=number * 365), uncertainty
        elif 'month' in text_lower:
            return timedelta(days=number * 30), uncertainty
        elif 'week' in text_lower:
            return timedelta(weeks=number), uncertainty
        elif 'day' in text_lower:
            return timedelta(days=number), uncertainty
        elif 'hour' in text_lower:
            return timedelta(hours=number), uncertainty
        elif 'minute' in text_lower:
            return timedelta(minutes=number), uncertainty
        
        return None, uncertainty
    
    def _deduplicate_expressions(self, expressions: List[TemporalExpression]) -> List[TemporalExpression]:
        """Remove duplicate temporal expressions"""
        seen = set()
        unique_expressions = []
        
        for expr in expressions:
            # Create a key based on text and position
            key = (expr.text.lower(), expr.source_span)
            if key not in seen:
                seen.add(key)
                unique_expressions.append(expr)
        
        return unique_expressions
    
    def extract_temporal_relationships(self, text: str, events: List[str]) -> List[TemporalRelationship]:
        """
        Extract temporal relationships between events using pattern matching.
        """
        relationships = []
        
        # Pattern matching for temporal relationships
        relationship_patterns = {
            TemporalRelation.BEFORE: [
                r'(.+?)\s+(?:before|prior to|ahead of)\s+(.+?)(?:\.|,|;|$)',
                r'(.+?)\s+(?:preceded|came before)\s+(.+?)(?:\.|,|;|$)',
            ],
            TemporalRelation.AFTER: [
                r'(.+?)\s+(?:after|following|subsequent to)\s+(.+?)(?:\.|,|;|$)',
                r'(.+?)\s+(?:followed|came after)\s+(.+?)(?:\.|,|;|$)',
            ],
            TemporalRelation.DURING: [
                r'(.+?)\s+(?:during|while|throughout)\s+(.+?)(?:\.|,|;|$)',
                r'(?:during|while|throughout)\s+(.+?),?\s+(.+?)(?:\.|,|;|$)',
            ],
            TemporalRelation.CONCURRENT: [
                r'(.+?)\s+(?:simultaneously|concurrently|at the same time as)\s+(.+?)(?:\.|,|;|$)',
                r'(.+?)\s+and\s+(.+?)\s+(?:occurred|happened)\s+(?:simultaneously|concurrently|together)',
            ]
        }
        
        for relation, patterns in relationship_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                    event1 = match.group(1).strip()
                    event2 = match.group(2).strip()
                    
                    # Calculate confidence based on pattern specificity
                    confidence = 0.8  # Default confidence
                    if 'simultaneously' in match.group().lower():
                        confidence = 0.9
                    elif 'concurrent' in match.group().lower():
                        confidence = 0.85
                    
                    relationships.append(TemporalRelationship(
                        event1=event1,
                        event2=event2,
                        relation=relation,
                        confidence=confidence,
                        evidence_text=match.group().strip()
                    ))
        
        return relationships
    
    def extract_llm_enhanced_temporal_info(self, text: str, llm_function=None) -> Dict[str, Any]:
        """
        Use LLM to extract complex temporal information that pattern matching might miss.
        This would integrate with the existing LLM infrastructure.
        """
        if not llm_function:
            return {"temporal_expressions": [], "relationships": [], "sequences": []}
        
        # LLM prompt for temporal extraction
        prompt = f"""
        Extract temporal information from the following text for process tracing analysis:

        TEXT: {text}

        Extract:
        1. TEMPORAL ENTITIES:
           - Specific dates and times
           - Relative temporal expressions (before, after, during)
           - Duration expressions (lasted 3 months, for 2 years)
           - Sequence indicators (first, then, finally)

        2. TEMPORAL RELATIONSHIPS:
           - Which events precede others
           - Concurrent or simultaneous events
           - Overlapping processes
           - Causal timing requirements

        3. CRITICAL JUNCTURES:
           - Key decision points with timing implications
           - Moments where timing affected outcomes
           - Time-sensitive transitions

        Provide structured output with confidence scores for each temporal claim.
        
        Output format:
        {{
            "temporal_expressions": [
                {{
                    "text": "extracted expression",
                    "type": "absolute|relative|duration|sequence",
                    "normalized_value": "ISO datetime if applicable",
                    "confidence": 0.8,
                    "uncertainty": 0.2
                }}
            ],
            "temporal_relationships": [
                {{
                    "event1": "first event",
                    "event2": "second event", 
                    "relationship": "before|after|during|concurrent",
                    "confidence": 0.9,
                    "evidence": "supporting text"
                }}
            ],
            "critical_junctures": [
                {{
                    "description": "critical moment description",
                    "timing_importance": "why timing mattered",
                    "confidence": 0.8
                }}
            ]
        }}
        """
        
        try:
            response = llm_function(prompt)
            return json.loads(response) if isinstance(response, str) else response
        except Exception as e:
            print(f"LLM temporal extraction failed: {e}")
            return {"temporal_expressions": [], "relationships": [], "sequences": []}

def test_temporal_extraction():
    """Test function for temporal extraction"""
    extractor = TemporalExtractor()
    
    test_text = """
    The crisis began on January 15, 2020, when the initial policy was announced. 
    Two months later, after widespread protests, the government reversed its decision.
    During this period, several key events occurred simultaneously. The final resolution
    came in late March, approximately 10 weeks after the initial announcement.
    """
    
    # Test temporal expression extraction
    expressions = extractor.extract_temporal_expressions(test_text)
    print("Temporal Expressions:")
    for expr in expressions:
        print(f"  {expr.text} ({expr.temporal_type.value}) - Confidence: {expr.confidence:.2f}")
    
    # Test relationship extraction
    relationships = extractor.extract_temporal_relationships(test_text, [])
    print("\nTemporal Relationships:")
    for rel in relationships:
        print(f"  {rel.event1} {rel.relation.value} {rel.event2} - Confidence: {rel.confidence:.2f}")

if __name__ == "__main__":
    test_temporal_extraction()
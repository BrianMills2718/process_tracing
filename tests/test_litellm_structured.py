#!/usr/bin/env python3
"""Test proper LiteLLM structured output with Pydantic"""

import os
import litellm
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No API key found")
    exit(1)

# Define a Pydantic model
class TestAssessment(BaseModel):
    """Test assessment structure"""
    diagnostic_type: str = Field(description="Van Evera test type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Reasoning for assessment")

# Test 1: Pass Pydantic model directly to response_format
print("Test 1: Pydantic model in response_format")
print("-" * 40)

try:
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{
            "role": "user", 
            "content": "Classify this as a Van Evera test type: 'The evidence strongly confirms the hypothesis'"
        }],
        response_format=TestAssessment,  # Pass Pydantic model directly
        api_key=api_key
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response content: {response.choices[0].message.content}")
    
    # Try to parse as Pydantic
    import json
    content = response.choices[0].message.content
    if isinstance(content, str):
        data = json.loads(content)
        assessment = TestAssessment(**data)
        print(f"Parsed successfully: {assessment}")
    else:
        print(f"Content is not a string: {type(content)}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

# Test 2: Use response_format with dict
print("Test 2: Dict-based response_format")
print("-" * 40)

try:
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{
            "role": "user", 
            "content": "Classify this as a Van Evera test type: 'The evidence is necessary but not sufficient'"
        }],
        response_format={"type": "json_object"},
        api_key=api_key
    )
    
    print(f"Response content: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)

# Test 3: What the current code does
print("Test 3: Current approach (schema in prompt)")
print("-" * 40)

schema = TestAssessment.model_json_schema()
prompt = f"""Classify this as a Van Evera test type: 'The evidence weakly suggests the hypothesis'

Respond with valid JSON matching this schema: {schema}"""

try:
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        api_key=api_key
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    # Parse and validate
    data = json.loads(response.choices[0].message.content)
    assessment = TestAssessment(**data)
    print(f"Parsed: {assessment}")
    
except Exception as e:
    print(f"Error: {e}")
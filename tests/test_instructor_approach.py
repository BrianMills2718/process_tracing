#!/usr/bin/env python3
"""Test using instructor library for proper structured output"""

import os
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pt.llm import _schema_response_format

load_dotenv()

if os.getenv("PT_RUN_LIVE_LLM_TESTS") != "1":
    pytest.skip(
        "live LLM exploratory script; set PT_RUN_LIVE_LLM_TESTS=1 to run",
        allow_module_level=True,
    )

# Check if instructor is available
try:
    import instructor
    from litellm import completion
    
    print("[OK] Instructor library available")
    
    # Patch LiteLLM with instructor
    client = instructor.from_litellm(completion)
    
    class TestAssessment(BaseModel):
        """Test assessment structure"""
        diagnostic_type: str = Field(description="Van Evera test type")
        confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
        reasoning: str = Field(description="Reasoning for assessment")
    
    # Use instructor to get actual Pydantic object
    response = client.chat.completions.create(
        model="gemini/gemini-2.5-flash",
        response_model=TestAssessment,  # This returns actual Pydantic object!
        messages=[{
            "role": "user",
            "content": "Classify this as Van Evera test: 'Evidence is necessary but not sufficient'"
        }],
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
    print(f"Direct access: diagnostic_type={response.diagnostic_type}")
    
except ImportError:
    print("[X] Instructor not installed")
    print("To get proper structured output, run: pip install instructor")
    
    # Show current workaround
    print("\nCurrent approach (JSON parsing required):")
    import litellm
    import json
    
    class TestAssessment(BaseModel):
        diagnostic_type: str = Field(description="Van Evera test type") 
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str
    
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{
            "role": "user",
            "content": f"Classify as Van Evera test. Respond with JSON: {TestAssessment.model_json_schema()}"
        }],
        response_format=_schema_response_format(TestAssessment)
    )
    
    # Must parse JSON string
    data = json.loads(response.choices[0].message.content)
    assessment = TestAssessment(**data)
    print(f"Parsed result: {assessment}")

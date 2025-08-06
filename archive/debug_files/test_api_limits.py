#!/usr/bin/env python3

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.extract import PROMPT_TEMPLATE, GEMINI_API_KEY, MODEL_NAME
import google.generativeai as genai

def test_api_with_comprehensive_prompt():
    """Test if the comprehensive prompt causes API timeouts"""
    
    # Configure API
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Short test text
    short_text = """The Boston Tea Party occurred on December 16, 1773, when colonists dumped tea into Boston Harbor. This event was caused by the Tea Act and led to increased tensions between Britain and America."""
    
    # Format the comprehensive prompt
    final_prompt = PROMPT_TEMPLATE.format(text=short_text)
    
    print(f"Testing API call with comprehensive prompt...")
    print(f"Prompt length: {len(final_prompt)} characters")
    print(f"Text length: {len(short_text)} characters")
    
    try:
        start_time = time.time()
        print("Starting API call...")
        
        # Make the API call with timeout simulation
        response = model.generate_content(final_prompt)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"API call completed in {duration:.2f} seconds")
        print(f"Response received: {len(response.text)} characters")
        
        # Check if response looks like valid JSON
        if '{' in response.text and '}' in response.text:
            print("Response appears to contain JSON structure")
        else:
            print("Response does not appear to be JSON")
            
        return True
        
    except Exception as e:
        print(f"API call failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api_with_comprehensive_prompt()
    if success:
        print("SUCCESS: Comprehensive prompt works with API")
        sys.exit(0)
    else:
        print("FAILURE: Comprehensive prompt failed with API")
        sys.exit(1)
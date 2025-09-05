"""
Universal LLM Interface - Drop into any project
Handles all major providers with automatic optimization
"""

import os
import litellm
from litellm import Router
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Suppress LiteLLM logs for cleaner output
litellm.suppress_debug_info = True


class UniversalLLM:
    """Universal interface for all LLM providers with smart routing"""
    
    def __init__(self):
        self.router = self._setup_router()
    
    def _setup_router(self) -> Router:
        """Setup router with unified GPT-5-mini configuration"""
        model_list = []
        
        # PHASE 16B: UNIFIED LLM CONFIGURATION - GPT-5-mini Only
        # Priority: OpenAI models only when available, with clear fallbacks
        if os.getenv("OPENAI_API_KEY"):
            # Use GPT-5-mini for all smart/fast operations - NOTE: GPT-5-mini uses max_completion_tokens
            model_list.extend([
                {"model_name": "smart", "litellm_params": {"model": "gpt-5-mini", "max_completion_tokens": 16384}},
                {"model_name": "fast", "litellm_params": {"model": "gpt-5-mini", "max_completion_tokens": 16384}},
                {"model_name": "reasoning", "litellm_params": {"model": "o1-preview", "max_completion_tokens": 32768}},
                {"model_name": "legacy-smart", "litellm_params": {"model": "gpt-4o", "max_tokens": 4096}},
                {"model_name": "legacy-fast", "litellm_params": {"model": "gpt-4o-mini", "max_tokens": 16384}},
            ])
            print("[INFO] UniversalLLM: Using GPT-5-mini for unified pipeline")
        
        # Fallback providers only when OpenAI not available
        elif os.getenv("ANTHROPIC_API_KEY"):
            model_list.extend([
                {"model_name": "smart", "litellm_params": {"model": "claude-3-5-sonnet-20241022", "max_tokens": 8192}},
                {"model_name": "fast", "litellm_params": {"model": "claude-3-5-haiku-20241022", "max_tokens": 8192}},
            ])
            print("[INFO] UniversalLLM: Fallback to Anthropic models")
        
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            model_list.extend([
                {"model_name": "smart", "litellm_params": {"model": "gemini/gemini-2.5-flash", "max_tokens": 65536}},
                {"model_name": "code", "litellm_params": {"model": "gemini/gemini-2.5-flash", "max_tokens": 65536, "tools": [{"codeExecution": {}}]}},
                {"model_name": "fast", "litellm_params": {"model": "gemini/gemini-2.5-flash", "max_tokens": 32768}},
            ])
            print("[INFO] UniversalLLM: Fallback to Gemini models")
        
        # OpenRouter fallback (lowest priority)
        elif os.getenv("OPENROUTER_API_KEY"):
            model_list.extend([
                {"model_name": "smart", "litellm_params": {"model": "openrouter/anthropic/claude-3.5-sonnet", "max_tokens": 8192}},
                {"model_name": "fast", "litellm_params": {"model": "openrouter/meta-llama/llama-3.1-8b-instruct:free", "max_tokens": 8192}},
            ])
            print("[INFO] UniversalLLM: Fallback to OpenRouter models")
        
        if not model_list:
            raise ValueError("No API keys found! Please set at least one API key in .env file")
        
        return Router(
            model_list=model_list,
            routing_strategy="simple-shuffle"  # Avoid cost-based routing issues
        )
    
    def chat(self, prompt: str, model_type: str = "smart", **kwargs) -> str:
        """Universal chat interface"""
        response = self.router.completion(
            model=model_type,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    def code_execution(self, prompt: str) -> str:
        """Execute code using Gemini's native code execution"""
        try:
            return self.chat(prompt, model_type="code")
        except Exception as e:
            # Fallback to regular model if code execution fails
            return self.chat(f"Please solve this step by step with detailed explanation: {prompt}", model_type="smart")
    
    def reasoning(self, prompt: str) -> str:
        """Use o1 models for complex reasoning"""
        try:
            return self.chat(prompt, model_type="reasoning")
        except Exception as e:
            # Fallback to regular model
            return self.chat(f"Think through this step by step: {prompt}", model_type="smart")
    
    def function_call(self, prompt: str, functions: List[Dict]) -> Any:
        """Universal function calling"""
        response = self.router.completion(
            model="smart",
            messages=[{"role": "user", "content": prompt}],
            tools=functions
        )
        return response
    
    def structured_output(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Get structured JSON output using router for unified model selection"""
        
        # Use the router's smart model instead of hardcoded Gemini
        # This ensures consistent routing with the rest of the system
        messages = [{"role": "user", "content": prompt}]
        
        # Add schema to prompt if provided
        if schema and hasattr(schema, 'model_json_schema'):
            import json
            schema_json = schema.model_json_schema()
            schema_prompt = f"\n\nYou must return valid JSON that matches this schema:\n```json\n{json.dumps(schema_json, indent=2)}\n```"
            messages[0]["content"] += schema_prompt
        
        # Use router instead of hardcoded Gemini model
        try:
            response = self.router.completion(
                model="smart",  # Use smart model from router (GPT-5-mini)
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            # Handle cases where the response isn't proper JSON
            if content and not content.strip().startswith('{'):
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # Look for any JSON-like structure
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
            
            return content
            
        except Exception as e:
            raise ValueError(f"Structured output generation failed: {e}")
    
    def compare_models(self, prompt: str, models: Optional[List[str]] = None) -> Dict[str, str]:
        """Compare responses across different models"""
        if models is None:
            models = ["smart", "fast", "code"]
        
        results = {}
        for model in models:
            try:
                results[model] = self.chat(prompt, model_type=model)
            except Exception as e:
                results[model] = f"Error: {str(e)}"
        
        return results


# Convenience functions for quick use
_llm = None

def get_llm() -> UniversalLLM:
    """Get singleton LLM instance"""
    global _llm
    if _llm is None:
        _llm = UniversalLLM()
    return _llm

def chat(prompt: str, model_type: str = "smart") -> str:
    """Quick chat function"""
    return get_llm().chat(prompt, model_type)

def code(prompt: str) -> str:
    """Quick code execution"""
    return get_llm().code_execution(prompt)

def reason(prompt: str) -> str:
    """Quick reasoning with o1 models"""
    return get_llm().reasoning(prompt)

def structured(prompt: str, schema: Optional[BaseModel] = None) -> str:
    """Quick structured output"""
    return get_llm().structured_output(prompt, schema)

def compare(prompt: str, models: Optional[List[str]] = None) -> Dict[str, str]:
    """Quick model comparison"""
    return get_llm().compare_models(prompt, models)


if __name__ == "__main__":
    # Quick test
    llm = UniversalLLM()
    print("🚀 Universal LLM ready!")
    print(f"Available models: {[m['model_name'] for m in llm.router.model_list]}")
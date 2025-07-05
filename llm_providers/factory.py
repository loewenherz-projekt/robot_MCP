"""
Factory for creating LLM providers based on model names.
"""

import os
from typing import Optional
from .claude_provider import ClaudeProvider
from .gemini_provider import GeminiProvider


def create_llm_provider(model: str, api_key: Optional[str] = None):
    """
    Create an LLM provider based on the model name.
    
    Args:
        model: Model name (e.g., "claude-3-7-sonnet-latest", "gemini-2.5-flash")
        api_key: Optional API key override
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If model is not supported or API key is missing
    """
    model_lower = model.lower()
    
    if "claude" in model_lower:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return ClaudeProvider(key, model)
    
    elif "gemini" in model_lower:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return GeminiProvider(key, model)
    
    else:
        raise ValueError(f"Unsupported model: {model}. Supported models: claude-*, gemini-*")

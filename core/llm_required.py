"""
LLM requirement enforcement utilities.
System MUST fail if LLM is unavailable.
NO FALLBACKS ALLOWED.
"""

import os
import logging

logger = logging.getLogger(__name__)


class LLMRequiredError(Exception):
    """Raised when LLM is required but unavailable"""
    pass


def require_llm():
    """
    Ensure LLM is available or fail immediately.
    NO FALLBACKS ALLOWED.
    
    Returns:
        LLM interface instance
        
    Raises:
        LLMRequiredError: If LLM is unavailable for any reason
    """
    # Check for explicit disable flag (for testing)
    if os.environ.get('DISABLE_LLM') == 'true':
        raise LLMRequiredError("LLM explicitly disabled via DISABLE_LLM environment variable")
    
    try:
        # Import must succeed
        from core.plugins.van_evera_llm_interface import get_van_evera_llm
    except ImportError as e:
        raise LLMRequiredError(f"Cannot import LLM interface: {e}")
    
    try:
        # Get LLM instance
        llm = get_van_evera_llm()
    except Exception as e:
        raise LLMRequiredError(f"Failed to initialize LLM interface: {e}")
    
    # Verify LLM is actually available
    if not llm:
        raise LLMRequiredError("LLM interface returned None - LLM is required")
    
    # Test that LLM can actually make calls
    try:
        # Quick test to ensure LLM is responsive
        test_result = llm.assess_probative_value(
            evidence_description="test",
            hypothesis_description="test",
            context="LLM availability check"
        )
        if not test_result:
            raise LLMRequiredError("LLM test call failed - LLM must be functional")
    except LLMRequiredError:
        raise  # Re-raise our own errors
    except Exception as e:
        raise LLMRequiredError(f"LLM test call failed: {e}")
    
    logger.info("LLM requirement satisfied - interface available and functional")
    return llm


def require_llm_lazy():
    """
    Lazy version that doesn't test on import, only when called.
    Used for module-level initialization.
    
    Returns:
        LLM interface instance
        
    Raises:
        LLMRequiredError: If LLM is unavailable
    """
    if os.environ.get('DISABLE_LLM') == 'true':
        raise LLMRequiredError("LLM explicitly disabled via DISABLE_LLM environment variable")
    
    try:
        from core.plugins.van_evera_llm_interface import get_van_evera_llm
        llm = get_van_evera_llm()
        if not llm:
            raise LLMRequiredError("LLM interface required but not available")
        return llm
    except Exception as e:
        raise LLMRequiredError(f"Cannot operate without LLM: {e}")
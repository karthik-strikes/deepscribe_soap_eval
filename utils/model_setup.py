"""
Model Setup Utilities
=====================

Centralizes initialization of DSPy and LLM clients for SOAP generation and evaluation.
"""

import dspy
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def setup_dspy_model(model_name: str, max_tokens: int = 4000, temperature: float = 0.1) -> bool:
    """
    Setup DSPy model configuration globally.

    Configures the DSPy framework with the specified language model.
    This configuration is used for all DSPy-based operations including
    field detection, SOAP generation, and evaluation.

    Args:
        model_name: Model identifier (e.g., "gemini/gemini-2.5-pro", "gpt-4")
        max_tokens: Maximum tokens for model responses
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

    Returns:
        True if setup successful, False otherwise

    Example:
        >>> success = setup_dspy_model("gemini/gemini-2.5-pro", max_tokens=4000)
        >>> if success:
        ...     # DSPy is now configured and ready to use
    """
    logger.info(f"Setting up DSPy model: {model_name}")
    logger.info(f"  max_tokens: {max_tokens}, temperature: {temperature}")

    try:
        lm = dspy.LM(model_name, max_tokens=max_tokens,
                     temperature=temperature)
        dspy.configure(lm=lm)
        logger.info("DSPy model configured successfully")
        return True

    except Exception as e:
        logger.error(f"DSPy model setup failed: {e}")
        return False


def create_llm_client(provider: str, model_name: str, api_key: Optional[str] = None,
                      **kwargs) -> Optional[Any]:
    """
    Create LLM client for direct API usage.

    Creates a client for OpenAI-compatible APIs. Used when soap_engine="llm"
    to generate SOAP notes via direct API calls instead of DSPy.

    Args:
        provider: Provider name ("openai", "anthropic", "google", etc.)
        model_name: Model identifier
        api_key: API key (if None, reads from environment)
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM client instance or None if creation failed

    Example:
        >>> client = create_llm_client("openai", "gpt-4", api_key="sk-...")
        >>> if client:
        ...     # Use client for SOAP generation
    """
    logger.info(
        f"Creating LLM client: provider={provider}, model={model_name}")

    try:
        if provider.lower() == "openai":
            return _create_openai_client(model_name, api_key, **kwargs)
        elif provider.lower() == "anthropic":
            return _create_anthropic_client(model_name, api_key, **kwargs)
        elif provider.lower() == "google":
            return _create_google_client(model_name, api_key, **kwargs)
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            return None

    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return None


def _create_openai_client(model_name: str, api_key: Optional[str] = None, **kwargs) -> Any:
    """
    Create OpenAI client.

    Args:
        model_name: OpenAI model name
        api_key: API key (reads from OPENAI_API_KEY env if None)
        **kwargs: Additional OpenAI client arguments

    Returns:
        OpenAI client instance
    """
    try:
        from openai import OpenAI
        import os

        client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            **kwargs
        )
        logger.info(f"OpenAI client created for model: {model_name}")
        return client

    except ImportError:
        logger.error("OpenAI package not installed. Run: pip install openai")
        raise
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        raise


def _create_anthropic_client(model_name: str, api_key: Optional[str] = None, **kwargs) -> Any:
    """
    Create Anthropic client.

    Args:
        model_name: Anthropic model name
        api_key: API key (reads from ANTHROPIC_API_KEY env if None)
        **kwargs: Additional Anthropic client arguments

    Returns:
        Anthropic client instance
    """
    try:
        from anthropic import Anthropic
        import os

        client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs
        )
        logger.info(f"Anthropic client created for model: {model_name}")
        return client

    except ImportError:
        logger.error(
            "Anthropic package not installed. Run: pip install anthropic")
        raise
    except Exception as e:
        logger.error(f"Failed to create Anthropic client: {e}")
        raise


def _create_google_client(model_name: str, api_key: Optional[str] = None, **kwargs) -> Any:
    """
    Create Google AI client.

    Args:
        model_name: Google model name
        api_key: API key (reads from GOOGLE_API_KEY env if None)
        **kwargs: Additional Google client arguments

    Returns:
        Google AI client instance
    """
    try:
        import google.generativeai as genai
        import os

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        client = genai.GenerativeModel(model_name)
        logger.info(f"Google AI client created for model: {model_name}")
        return client

    except ImportError:
        logger.error(
            "Google AI package not installed. Run: pip install google-generativeai")
        raise
    except Exception as e:
        logger.error(f"Failed to create Google AI client: {e}")
        raise


def validate_model_config(soap_engine: str, model_config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that model configuration is complete for selected engine.

    Args:
        soap_engine: Selected SOAP engine ("dspy" or "llm")
        model_config: Model configuration dictionary

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> config = {"name": "gpt-4", "provider": "openai"}
        >>> valid, error = validate_model_config("llm", config)
        >>> if not valid:
        ...     print(f"Config error: {error}")
    """
    if soap_engine == "dspy":
        # DSPy just needs model name
        if not model_config.get("name"):
            return False, "DSPy engine requires 'model.name' in config"
        return True, ""

    elif soap_engine == "llm":
        # LLM engine needs provider, model name, and prompt file
        if not model_config.get("provider"):
            return False, "LLM engine requires 'model.provider' in config"
        if not model_config.get("name"):
            return False, "LLM engine requires 'model.name' in config"
        if not model_config.get("prompt_file"):
            return False, "LLM engine requires 'model.prompt_file' in config"
        return True, ""

    else:
        return False, f"Unknown SOAP engine: {soap_engine}"


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration structure with all required fields.

    Returns:
        Default configuration dictionary
    """
    return {
        "model": {
            "name": "gemini/gemini-2.5-pro",
            "provider": "google",  # For LLM engine
            "prompt_file": "config/llm_prompts.yaml",  # For LLM engine
            "max_tokens": 4000,
            "temperature": 0.1
        },
        "defaults": {
            "samples": 5,
            "mode": "both",
            "output": "results/soap_results.json",
            "storage": "both",
            "soap_engine": "dspy",
            "evaluation_mode": "comprehensive",
            "batch_size": 10
        }
    }

"""
LLM Client Adapter
==================

Provides a unified interface for different LLM providers (OpenAI, Anthropic, Google).
Normalizes API differences so LLMSOAPEngine can work with any provider.
"""

import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """
    Abstract adapter for LLM clients.

    Provides a unified interface for different LLM providers.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.

        Args:
            system_prompt: System message defining assistant behavior
            user_prompt: User message with the actual request
            **kwargs: Provider-specific arguments

        Returns:
            Generated text
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for OpenAI API.

    Wraps OpenAI client to provide unified interface.
    """

    def __init__(self, client, model_name: str, temperature: float = 0.1):
        """
        Initialize OpenAI adapter.

        Args:
            client: OpenAI client instance
            model_name: Model identifier (e.g., "gpt-4")
            temperature: Sampling temperature
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.chat = self  # Compatibility shim
        self.completions = self  # Compatibility shim

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float, **kwargs) -> Any:
        """
        OpenAI-style create method for compatibility with existing code.

        Args:
            model: Model name
            messages: List of message dictionaries
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            OpenAI response object
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI API.

        Args:
            system_prompt: System message
            user_prompt: User message
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class AnthropicAdapter(LLMAdapter):
    """
    Adapter for Anthropic API.

    Wraps Anthropic client to provide unified interface.
    """

    def __init__(self, client, model_name: str, temperature: float = 0.1, max_tokens: int = 4000):
        """
        Initialize Anthropic adapter.

        Args:
            client: Anthropic client instance
            model_name: Model identifier (e.g., "claude-3-opus-20240229")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat = self  # Compatibility shim
        self.completions = self  # Compatibility shim

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float, **kwargs) -> Any:
        """
        OpenAI-style create method for compatibility.

        Converts OpenAI format to Anthropic format.

        Args:
            model: Model name (ignored, uses self.model_name)
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Mock response object with OpenAI-like structure
        """
        # Extract system message if present
        system_msg = None
        user_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                user_messages.append(msg)

        # Call Anthropic API
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=system_msg,
            messages=user_messages,
            temperature=temperature
        )

        # Create OpenAI-compatible response object
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': content})()
                })()]

        return MockResponse(response.content[0].text)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic API.

        Args:
            system_prompt: System message
            user_prompt: User message
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                **kwargs
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class GoogleAdapter(LLMAdapter):
    """
    Adapter for Google AI API.

    Wraps Google GenerativeModel to provide unified interface.
    """

    def __init__(self, model, temperature: float = 0.1):
        """
        Initialize Google adapter.

        Args:
            model: Google GenerativeModel instance
            temperature: Sampling temperature
        """
        self.model = model
        self.temperature = temperature
        self.chat = self  # Compatibility shim
        self.completions = self  # Compatibility shim

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float, **kwargs) -> Any:
        """
        OpenAI-style create method for compatibility.

        Converts OpenAI format to Google format.

        Args:
            model: Model name (ignored)
            messages: List of message dictionaries
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Mock response object with OpenAI-like structure
        """
        # Combine system and user messages
        combined_prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                combined_prompt += f"Instructions: {msg['content']}\n\n"
            elif msg['role'] == 'user':
                combined_prompt += msg['content']

        # Call Google API
        response = self.model.generate_content(
            combined_prompt,
            generation_config={'temperature': temperature}
        )

        # Create OpenAI-compatible response object
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': content})()
                })()]

        return MockResponse(response.text)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate text using Google AI API.

        Args:
            system_prompt: System message (prepended as instructions)
            user_prompt: User message
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        try:
            # Google doesn't have separate system messages, so combine them
            combined_prompt = f"Instructions: {system_prompt}\n\n{user_prompt}"

            response = self.model.generate_content(
                combined_prompt,
                generation_config={'temperature': self.temperature}
            )
            return response.text

        except Exception as e:
            logger.error(f"Google AI generation failed: {e}")
            raise


def create_adapter(provider: str, client, model_name: str, **kwargs) -> LLMAdapter:
    """
    Create appropriate adapter for the given provider.

    Args:
        provider: Provider name ("openai", "anthropic", "google")
        client: Provider-specific client instance
        model_name: Model identifier
        **kwargs: Additional arguments (temperature, max_tokens, etc.)

    Returns:
        LLMAdapter instance for the provider

    Raises:
        ValueError: If provider is unsupported

    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> adapter = create_adapter("openai", client, "gpt-4", temperature=0.1)
        >>> # adapter now has unified interface
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        return OpenAIAdapter(client, model_name, kwargs.get('temperature', 0.1))
    elif provider_lower == "anthropic":
        return AnthropicAdapter(
            client,
            model_name,
            kwargs.get('temperature', 0.1),
            kwargs.get('max_tokens', 4000)
        )
    elif provider_lower == "google":
        return GoogleAdapter(client, kwargs.get('temperature', 0.1))
    else:
        raise ValueError(f"Unsupported provider: {provider}")

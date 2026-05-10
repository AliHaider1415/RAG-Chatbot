from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """Base exception for service-level failures."""


class ConfigurationError(ServiceError):
    """Raised when required configuration is missing or invalid."""


class RetrievalError(ServiceError):
    """Raised when vector retrieval or embedding generation fails."""


class LLMError(ServiceError):
    """Raised when LLM initialization or inference fails."""

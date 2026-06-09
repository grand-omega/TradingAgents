"""Model name validators for each provider."""

from .model_catalog import get_known_models


# Providers whose model names are open-ended (custom IDs, rolling aliases, or
# user-deployed names) and therefore accept any model without validation.
_ANY_MODEL_PROVIDERS = ("ollama", "openrouter", "claude-cli", "claude_code")

VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in _ANY_MODEL_PROVIDERS
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For ollama, openrouter, and claude-cli - any model is accepted.
    """
    provider_lower = provider.lower()

    if provider_lower in _ANY_MODEL_PROVIDERS:
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]

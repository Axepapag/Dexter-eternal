from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.tool_agent_provider import resolve_provider_config


def resolve_llm_slot(
    config: Dict[str, Any],
    slot_name: Optional[str],
) -> Tuple[str, Dict[str, Any], str]:
    """
    Resolve provider + model for a logical slot name.
    Uses config["llm_slots"] as the source of truth.
    Returns (provider_name, provider_config, model).
    """
    llm_slots = config.get("llm_slots", {}) or {}
    slot_key = (slot_name or "").strip() if slot_name else ""

    if slot_key in llm_slots:
        slot_cfg = llm_slots.get(slot_key, {}) or {}
        provider_name = slot_cfg.get("provider_name")
        model = slot_cfg.get("model")
        if provider_name and model:
            _, provider_cfg = resolve_provider_config(config, provider_name)
            # Allow per-slot provider overrides (e.g., tighter timeouts for dexter chat).
            overrides = slot_cfg.get("provider_overrides") or {}
            if isinstance(overrides, dict) and overrides:
                provider_cfg = dict(provider_cfg)
                provider_cfg.update(overrides)
            return provider_name, provider_cfg, model

    # Fallback (no slot configured): use config["provider"] if present, otherwise pick a reasonable default.
    provider_name = (config.get("provider", {}) or {}).get("name")
    if not provider_name:
        providers = config.get("providers", {}) or {}
        provider_name = "ollama_cloud" if "ollama_cloud" in providers else ("nvidia" if "nvidia" in providers else (next(iter(providers.keys()), "ollama_cloud")))
    _, provider_cfg = resolve_provider_config(config, provider_name)

    model = (config.get("provider", {}) or {}).get("model") or provider_cfg.get("model")
    if not model:
        models = provider_cfg.get("models") or []
        model = models[0] if models else ""
    return provider_name, provider_cfg, model

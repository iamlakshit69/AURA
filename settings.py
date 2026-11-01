"""
Where the settings come from.

    .env          secrets only: GROQ_API_KEY, SARVAM_API_KEY, SONIOX_API_KEY, ELEVEN_API_KEY, TAVILY_API_KEY, EXA_API_KEY
    settings.env  everything else: model names, voices, timings (no secrets, safe to share)
    prices.ini    prices, keyed by the model names found in settings.env

Order of priority: a variable set in the shell > settings.env > .env.
(So `STT_PROVIDER=sarvam python Structure-II.py console` still works for a one-off test.)
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

SETTINGS_FILE = os.getenv("SETTINGS_FILE", "settings.env")

STT_PROVIDERS = ("sarvam", "soniox", "elevenlabs")

# Used when settings.env does not say. This is the ONE place these defaults live: the agent and the
# price calculator both read the models through get_models().
DEFAULT_MODELS = {
    "GROQ_LLM_MODEL": "qwen/qwen3.8-27b",
    "TTS_MODEL": "bulbul:v4-flash",
}
# One STT model per provider, so switching STT_PROVIDER never leaves you with another provider's model name.
STT_MODEL_KEYS = {
    "sarvam": ("SARVAM_STT_MODEL", "saaras:v3-realtime"),
    "soniox": ("SONIOX_STT_MODEL", "stt-rt-v5"),
    "elevenlabs": ("ELEVENLABS_STT_MODEL", "scribe_v2_realtime"),
}


def load_settings(path: str | None = None) -> None:
    """Load settings.env into the environment. Call BEFORE load_dotenv() for the keys."""
    load_dotenv(path or SETTINGS_FILE, override=False)


def get_stt_provider() -> str:
    p = (os.getenv("STT_PROVIDER") or "sarvam").strip().lower()
    if p not in STT_PROVIDERS:
        raise ValueError(f"STT_PROVIDER={p!r} is not one of {', '.join(STT_PROVIDERS)}")
    return p


def get_models() -> dict:
    """The models in use right now: {'llm', 'tts', 'stt', 'stt_provider'}. The prices are looked up by these names."""
    def one(key: str) -> str:
        return (os.getenv(key) or DEFAULT_MODELS[key]).strip()
    provider = get_stt_provider()
    key, default = STT_MODEL_KEYS[provider]
    stt = (os.getenv(key) or "").strip()
    if not stt and provider == "sarvam":
        stt = (os.getenv("STT_MODEL") or "").strip()          # the old name, still honoured for Sarvam
    return {"llm": one("GROQ_LLM_MODEL"), "tts": one("TTS_MODEL"), "stt": stt or default, "stt_provider": provider}

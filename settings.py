"""
Where the settings come from.

    .env          secrets only: GROQ_API_KEY, SARVAM_API_KEY, TAVILY_API_KEY, EXA_API_KEY
    settings.env  everything else: model names, voices, timings (no secrets, safe to share)
    prices.ini    prices, keyed by the model names found in settings.env

Order of priority: a variable set in the shell > settings.env > .env.
(So `LOCAL_VAD=1 python Structure-II.py console` still works for a one-off test.)
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

SETTINGS_FILE = os.getenv("SETTINGS_FILE", "settings.env")

# Used when settings.env does not say. This is the ONE place these defaults live: the agent and the
# price calculator both read the models through get_models().
DEFAULT_MODELS = {
    "GROQ_LLM_MODEL": "qwen/qwen3.8-27b",
    "TTS_MODEL": "bulbul:v4-flash",
    "STT_MODEL": "saaras:v3-realtime",
}


def load_settings(path: str | None = None) -> None:
    """Load settings.env into the environment. Call BEFORE load_dotenv() for the keys."""
    load_dotenv(path or SETTINGS_FILE, override=False)


def get_models() -> dict:
    """The models in use right now: {'llm': ..., 'tts': ..., 'stt': ...}. The prices are looked up by these names."""
    def one(key: str) -> str:
        return (os.getenv(key) or DEFAULT_MODELS[key]).strip()
    return {"llm": one("GROQ_LLM_MODEL"), "tts": one("TTS_MODEL"), "stt": one("STT_MODEL")}

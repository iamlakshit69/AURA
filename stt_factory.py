"""
Builds the speech-to-text engine from STT_PROVIDER (settings.env): sarvam | soniox | elevenlabs.
Everything provider-specific lives here, so the agent only calls build_stt().

Turn detection: the agent runs with turn_detection="stt", so every provider must emit END_OF_SPEECH itself.
    sarvam      server-side VAD (endpointing="vad")
    soniox      endpoint detection (max_endpoint_delay_ms, endpoint_sensitivity, endpoint_latency_adjustment_level)
    elevenlabs  server VAD (VADOptions)
"""
from __future__ import annotations

import inspect
import logging
import os

log = logging.getLogger("voice-agent.stt")

# LiveKit plugins must be imported on the MAIN thread, at process start (they register themselves then).
# Importing them inside the job crashes with "Plugins must be registered on the main thread".
# This module is imported at the top of Structure-II.py, so import every installed plugin here.
sarvam = soniox = elevenlabs = None
try:
    from livekit.plugins import sarvam
except ImportError:
    pass
try:
    from livekit.plugins import soniox
except ImportError:
    pass
try:
    from livekit.plugins import elevenlabs
except ImportError:
    pass


def _need(module, provider: str, package: str):
    if module is None:
        raise RuntimeError(f"STT_PROVIDER={provider} needs the plugin:  pip install {package}")
    return module


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _csv(key: str, default: str) -> list[str]:
    return [x.strip() for x in _env(key, default).split(",") if x.strip()]


def build_stt(provider: str, model: str):
    p = provider.lower()
    if p == "sarvam":
        return _sarvam(model)
    if p == "soniox":
        return _soniox(model)
    if p == "elevenlabs":
        return _elevenlabs(model)
    raise ValueError(f"unknown STT_PROVIDER {provider!r}")


def _sarvam(model: str):
    _need(sarvam, "sarvam", "livekit-plugins-sarvam")
    extra = {}
    try:        # pass the model only if this plugin version takes one (it is also what gets priced)
        if "model" in inspect.signature(sarvam.STTRealtime).parameters:
            extra["model"] = model
        else:
            log.warning("sarvam.STTRealtime has no model parameter: the plugin default is used; "
                        "%s is only used for the price lookup", model)
    except (TypeError, ValueError):
        pass
    return sarvam.STTRealtime(
        **extra,
        language=_env("SARVAM_RT_LANGUAGE", "auto"),
        stream_type=_env("SARVAM_RT_STREAM_TYPE", "fast"),
        mode=_env("SARVAM_RT_TASK", "transcribe"),
        endpointing="vad",
        sample_rate=16000,
        vad_min_silence_ms=int(_env("SARVAM_VAD_SILENCE_MS", "300")),
        vad_sot_threshold=float(_env("SARVAM_VAD_THRESHOLD", "0.5")),
        vad_min_speech_ms=int(_env("SARVAM_VAD_MIN_SPEECH_MS", "80")),
    )


def _soniox(model: str):
    _need(soniox, "soniox", "livekit-plugins-soniox")
    if not _env("SONIOX_API_KEY"):
        raise RuntimeError("STT_PROVIDER=soniox needs SONIOX_API_KEY in .env")
    kw: dict = dict(
        model=model,
        language_hints=_csv("SONIOX_LANGUAGE_HINTS", "hi,pa,en"),
        language_hints_strict=_env("SONIOX_HINTS_STRICT", "0") == "1",
        enable_language_identification=_env("SONIOX_LANG_ID", "1") == "1",
        max_endpoint_delay_ms=int(_env("SONIOX_ENDPOINT_MS", "700")),          # 500..3000
    )
    # v5 only: older models reject these, so they are sent only when you set them
    if _env("SONIOX_ENDPOINT_SENSITIVITY"):
        kw["endpoint_sensitivity"] = float(_env("SONIOX_ENDPOINT_SENSITIVITY"))            # -1..1
    if _env("SONIOX_ENDPOINT_LATENCY_LEVEL"):
        kw["endpoint_latency_adjustment_level"] = int(_env("SONIOX_ENDPOINT_LATENCY_LEVEL"))   # 0..3
    if _env("SONIOX_CONTEXT"):
        kw["context"] = _env("SONIOX_CONTEXT")          # free text: names, jargon, the wake name
    return soniox.STT(params=soniox.STTOptions(**kw))


def _elevenlabs(model: str):
    _need(elevenlabs, "elevenlabs", "livekit-plugins-elevenlabs")
    if not _env("ELEVEN_API_KEY"):
        raise RuntimeError("STT_PROVIDER=elevenlabs needs ELEVEN_API_KEY in .env")
    kw: dict = dict(
        model=model,
        server_vad={"vad_silence_threshold_secs": float(_env("ELEVEN_VAD_SILENCE_S", "0.6")),
                    "vad_threshold": float(_env("ELEVEN_VAD_THRESHOLD", "0.5"))},
    )
    lang = _env("ELEVEN_LANGUAGE", "")
    if lang:
        kw["language_code"] = lang           # e.g. hi, en. Empty = auto-detect
    return elevenlabs.STT(**kw)

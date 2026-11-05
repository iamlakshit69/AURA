"""
Builds the text-to-speech engine from TTS_PROVIDER (settings.env): sarvam | bakbak.
Everything provider-specific lives here, so Structure-II.py only calls make_tts() / set_voice() / voice_label().

    sarvam   bulbul:v4-flash, one persona per language (SPEAKER_HI / SPEAKER_PA / SPEAKER_EN)
    bakbak   model "standard" or "m1", voice IDs per language (BAKBAK_VOICE_HI / _PA / _EN). Voice IDs belong to a
             model. If you leave one empty, the first voice of that language for your model is taken from /v1/voices.
Languages the agent switches between: hi, pa, en. Bakbak also has mr te kn bn as gu ne ml ta (see bakbak_client.py).
"""
from __future__ import annotations

import logging
import os

# LiveKit plugins must be imported on the MAIN thread at process start (same rule as stt_factory.py).
try:
    from livekit.plugins import sarvam
except ImportError:
    sarvam = None

log = logging.getLogger("voice-agent.tts")

BAKBAK_LANG = {"hi": "hi", "pa": "pa", "en": "en-in"}       # the agent's language -> Bakbak's code
_voices_cache: list | None = None
_resolved: dict[tuple[str, str], tuple[str, str]] = {}       # (model, lang) -> (voice_id, name)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def bakbak_voice(lang: str, model: str) -> tuple[str, str]:
    """(voice_id, display name) for this language: BAKBAK_VOICE_<LANG> if set, else the first matching voice."""
    global _voices_cache
    set_id = _env(f"BAKBAK_VOICE_{lang.upper()}")
    if set_id:
        return set_id, set_id[:8]
    if (model, lang) in _resolved:
        return _resolved[(model, lang)]
    import bakbak_client as bb
    if _voices_cache is None:
        _voices_cache = bb.list_voices(_env("BAKBAK_API_KEY"), _env("BAKBAK_BASE_URL") or bb.BASE_URL)
    v = bb.pick_voice(_voices_cache, BAKBAK_LANG[lang], model)
    if v is None:
        raise RuntimeError(f"Bakbak: no '{model}' voice for {BAKBAK_LANG[lang]}. Run: python bakbak_voices.py "
                           f"and set BAKBAK_VOICE_{lang.upper()} in settings.env")
    _resolved[(model, lang)] = (v["id"], v.get("name") or v["id"][:8])
    log.info("bakbak voice for %s picked automatically: %s (%s)", lang, v.get("name"), v["id"])
    return _resolved[(model, lang)]


def make_tts(provider: str, model: str, lang: str, *, rate: int, codes: dict, speakers: dict, **sarvam_extra):
    """sarvam_extra (min_buffer_size, max_chunk_length) only apply to Sarvam."""
    if provider == "bakbak":
        from bakbak_tts import TTS
        return TTS(voice_id=bakbak_voice(lang, model)[0], language=BAKBAK_LANG[lang], model=model, sample_rate=rate,
                   speed=float(_env("BAKBAK_SPEED", "1.0")), stream=_env("BAKBAK_STREAM", "1") != "0")
    if sarvam is None:
        raise RuntimeError("TTS_PROVIDER=sarvam needs the plugin:  pip install livekit-plugins-sarvam")
    return sarvam.TTS(model=model, target_language_code=codes[lang], speaker=speakers[lang], speech_sample_rate=rate,
                      output_audio_codec="linear16", **sarvam_extra)


def set_voice(tts, provider: str, model: str, lang: str, *, codes: dict, speakers: dict) -> None:
    if provider == "bakbak":
        tts.update_options(voice_id=bakbak_voice(lang, model)[0], language=BAKBAK_LANG[lang])
    else:
        tts.update_options(target_language_code=codes[lang], speaker=speakers[lang])


def voice_label(provider: str, model: str, lang: str, *, codes: dict, speakers: dict) -> str:
    if provider == "bakbak":
        return f"{BAKBAK_LANG[lang]} · {bakbak_voice(lang, model)[1]}"
    return f"{codes[lang]} · {speakers[lang]}"

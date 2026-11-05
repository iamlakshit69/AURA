"""
Bakbak (Raya / LitWiz Labs) HTTP client. No LiveKit imports, so it can be tested on its own.

    GET  {base}/voices                    X-API-Key header; each voice: id, name, language, model
    POST {base}/text-to-speech            whole utterance back (we ask for wav)
    POST {base}/text-to-speech/stream     Server-Sent Events: event "chunk" = base64 PCM float32 little-endian,
                                          event "done" = finished
Voice IDs belong to ONE model ("standard" or "m1"): never mix them.
Language codes: hi mr te kn bn as gu ne ml ta pa en-in en-us
"""
from __future__ import annotations

import base64
import io
import json
import wave
from typing import AsyncIterator

import httpx
import numpy as np

BASE_URL = "https://hub.getraya.app/v1"


class BakbakError(Exception):
    def __init__(self, message: str, status: int = -1, body: str = "") -> None:
        super().__init__(message)
        self.status, self.body = status, body

    @property
    def retryable(self) -> bool:
        return self.status in (-1, 408, 429) or self.status >= 500


def headers(api_key: str) -> dict:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def payload(text: str, voice_id: str, model: str, language: str, sample_rate: int, speed: float, codec: str | None = None) -> dict:
    body = {"text": text, "voice_id": voice_id, "model": model, "language": language,
            "sample_rate": sample_rate, "speed": speed}
    if codec:
        body["codec"] = codec
    return body


def f32le_to_int16(buf: bytes) -> bytes:
    """Bakbak streams float32 samples in [-1, 1]; LiveKit wants 16-bit PCM."""
    x = np.frombuffer(buf, dtype="<f4")
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


async def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code != 200:
        body = (await r.aread()).decode("utf-8", "replace")[:300]
        raise BakbakError(f"Bakbak HTTP {r.status_code}: {body}", r.status_code, body)


async def stream_pcm(client: httpx.AsyncClient, api_key: str, base_url: str, **p) -> AsyncIterator[bytes]:
    """Yield 16-bit PCM bytes as they arrive from the SSE endpoint."""
    leftover = b""
    try:
        async with client.stream("POST", f"{base_url}/text-to-speech/stream", json=payload(**p),
                                 headers={**headers(api_key), "Accept": "text/event-stream"}) as r:
            await _raise_for_status(r)
            data_lines: list[str] = []
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line.strip() or not data_lines:        # a blank line ends one event
                    continue
                msg = json.loads("\n".join(data_lines))
                data_lines = []
                kind = msg.get("type")
                if kind == "chunk" and msg.get("data"):
                    raw = leftover + base64.b64decode(msg["data"])
                    cut = len(raw) - len(raw) % 4         # whole float32 samples only
                    leftover = raw[cut:]
                    if cut:
                        yield f32le_to_int16(raw[:cut])
                elif kind == "done":
                    return
                elif kind == "error" or (msg.get("status_code") or 200) >= 400:
                    raise BakbakError(f"Bakbak stream error: {msg}", int(msg.get("status_code") or 500))
    except httpx.HTTPError as e:
        raise BakbakError(f"Bakbak connection problem: {e!r}") from e


async def synth_wav(client: httpx.AsyncClient, api_key: str, base_url: str, **p) -> tuple[bytes, int]:
    """Non-streaming fallback. Returns (16-bit mono PCM, sample_rate)."""
    try:
        r = await client.post(f"{base_url}/text-to-speech", json=payload(codec="wav", **p), headers=headers(api_key))
    except httpx.HTTPError as e:
        raise BakbakError(f"Bakbak connection problem: {e!r}") from e
    if r.status_code != 200:
        raise BakbakError(f"Bakbak HTTP {r.status_code}: {r.text[:300]}", r.status_code, r.text[:300])
    with wave.open(io.BytesIO(r.content), "rb") as w:
        pcm, rate, ch = w.readframes(w.getnframes()), w.getframerate(), w.getnchannels()
    if ch > 1:
        pcm = np.frombuffer(pcm, "<i2").reshape(-1, ch).mean(axis=1).astype("<i2").tobytes()
    return pcm, rate


def list_voices(api_key: str, base_url: str = BASE_URL, timeout: float = 8.0) -> list[dict]:
    r = httpx.get(f"{base_url}/voices", headers=headers(api_key), timeout=timeout)
    if r.status_code != 200:
        raise BakbakError(f"Bakbak HTTP {r.status_code}: {r.text[:300]}", r.status_code)
    data = r.json()
    return data if isinstance(data, list) else data.get("voices", data.get("data", []))


def pick_voice(voices: list[dict], language: str, model: str) -> dict | None:
    """First voice of this model that speaks `language` (compared on the part before '-', so en-in matches en)."""
    want = language.lower().split("-")[0]
    for v in voices:
        if (v.get("model") or "standard") == model and str(v.get("language", "")).lower().split("-")[0] == want:
            return v
    return None

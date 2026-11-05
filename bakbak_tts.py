"""
LiveKit TTS plugin for Bakbak (Raya). Same job as livekit.plugins.sarvam.TTS: build it once, switch voice
and language with update_options(), and synthesize(text).collect() gives an AudioFrame (used for the fillers).

LiveKit treats this as a non-streaming TTS and feeds it one sentence at a time. Each sentence is streamed from
Bakbak's SSE endpoint, so audio starts coming back while the sentence is still being generated. If the stream
endpoint fails BEFORE any audio arrived, the same sentence is retried on the non-streaming endpoint.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

import httpx
from livekit.agents import (DEFAULT_API_CONNECT_OPTIONS, APIConnectionError, APIConnectOptions, APIStatusError,
                            tts, utils)

import bakbak_client as bb

log = logging.getLogger("voice-agent.bakbak")


@dataclass
class _Opts:
    model: str
    voice_id: str
    language: str
    speed: float
    sample_rate: int
    stream: bool


class TTS(tts.TTS):
    def __init__(self, *, voice_id: str, language: str = "hi", model: str = "standard", speed: float = 1.0,
                 sample_rate: int = 24000, stream: bool = True, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self._key = (api_key or os.getenv("BAKBAK_API_KEY") or "").strip()
        if not self._key:
            raise ValueError("Bakbak needs BAKBAK_API_KEY in .env (get one at https://bakbak.getraya.app)")
        self._base = (base_url or os.getenv("BAKBAK_BASE_URL") or bb.BASE_URL).rstrip("/")
        self._opts = _Opts(model, voice_id, language, speed, sample_rate, stream)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0), limits=httpx.Limits(max_keepalive_connections=4))

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "bakbak"

    def update_options(self, *, voice_id: str | None = None, language: str | None = None, model: str | None = None,
                       speed: float | None = None) -> None:
        """Takes effect from the next sentence. Voice IDs belong to one model: change both together."""
        o = self._opts
        self._opts = replace(o, voice_id=voice_id or o.voice_id, language=language or o.language,
                             model=model or o.model, speed=speed or o.speed)

    def prewarm(self) -> None:
        """Open the TLS connection now so the first reply does not pay for it."""
        import asyncio

        async def _go():
            try:
                await self._client.get(f"{self._base}/voices", headers=bb.headers(self._key))
            except Exception:
                pass
        try:
            asyncio.get_running_loop().create_task(_go())
        except RuntimeError:
            pass

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._client.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._t, self._o = tts, replace(tts._opts)           # freeze the options of THIS sentence

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        o, t = self._o, self._t
        p = dict(text=self._input_text, voice_id=o.voice_id, model=o.model, language=o.language,
                 sample_rate=o.sample_rate, speed=o.speed)
        rid = utils.shortuuid()
        output_emitter.initialize(request_id=rid, sample_rate=o.sample_rate, num_channels=1, mime_type="audio/pcm")
        try:
            got = False
            if o.stream:
                try:
                    async for pcm in bb.stream_pcm(t._client, t._key, t._base, **p):
                        got = True
                        output_emitter.push(pcm)
                except bb.BakbakError as e:
                    if got or e.status in (400, 401, 404):
                        raise                                   # audio already played, or a request error a retry cannot fix
                    log.warning("stream failed (%s), using the non-streaming endpoint", e)
                if got:
                    output_emitter.flush()
                    return
                log.warning("stream gave no audio, using the non-streaming endpoint")
            pcm, rate = await bb.synth_wav(t._client, t._key, t._base, **p)
            if rate != o.sample_rate:
                log.warning("Bakbak returned %d Hz but %d Hz was requested: the voice will sound off", rate, o.sample_rate)
            output_emitter.push(pcm)
            output_emitter.flush()
        except bb.BakbakError as e:
            if e.status == -1:
                raise APIConnectionError(str(e)) from e
            raise APIStatusError(str(e), status_code=e.status, request_id=rid, body=e.body, retryable=e.retryable) from e

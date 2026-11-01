"""
Usage meter for the voice agent. No LiveKit imports, so it can be tested on its own.

It collects raw usage (tokens, characters, seconds, search calls) while a reply is being made and
writes ONE ROW PER REPLY to usage_log.csv. Prices are NOT applied here: apply them later with
cost_report.py, so you can change prices without losing history.

What is counted (billing unit of each provider):
    LLM (Groq)      prompt tokens, cached prompt tokens, completion tokens   -> per million tokens
    TTS (Sarvam)    characters sent to the synthesiser                       -> per 10K characters
    STT (Sarvam)    seconds of audio                                         -> per hour of audio
    Search          one request (Tavily: credits, Exa: requests)             -> per credit / per request

A reply with a tool call has several LLM requests (ask -> tool -> answer); they are all summed into
the row of the reply they belong to. Usage that happens but is never followed by a finished reply
(user barge-in, failed turn) is NOT lost: it stays in the counters and lands in the next row.
Note: STT seconds are streamed continuously, so per-row STT is 'audio since the previous row'; trust the total.
tts_estimated=1 means LiveKit sent no TTS metrics for that reply and tts_chars is the reply length.
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime

COUNTERS = [
    "llm_requests", "llm_prompt_tokens", "llm_cached_tokens", "llm_completion_tokens",
    "tts_chars", "tts_audio_s",
    "stt_audio_s",
    "tavily_requests", "tavily_credits", "exa_requests", "weather_calls",
]
HEADER = ["time", "llm", "reply_chars"] + COUNTERS + ["tts_estimated"]


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


class UsageMeter:
    def __init__(self, path: str = "usage_log.csv", llm_model: str = "") -> None:
        self.path = path
        self.llm_model = llm_model
        self.turn = dict.fromkeys(COUNTERS, 0.0)       # since the last written row
        self.session = dict.fromkeys(COUNTERS, 0.0)    # since the agent started
        self.session["prerender_tts_chars"] = 0.0      # filler clips rendered once at startup
        self.started = time.monotonic()

    # ---- raw adders -------------------------------------------------------
    def add(self, **kw) -> None:
        for k, v in kw.items():
            self.turn[k] += _num(v)
            self.session[k] += _num(v)

    def search_call(self, provider: str, credits: float = 1.0) -> None:
        """Called by Search for every request actually sent (hedges, retries and prefetches too)."""
        if provider == "tavily":
            self.add(tavily_requests=1, tavily_credits=credits)
        elif provider == "exa":
            self.add(exa_requests=1)

    def prerender(self, chars: int) -> None:
        self.session["prerender_tts_chars"] += chars

    # ---- LiveKit metrics --------------------------------------------------
    def on_metrics(self, m) -> None:
        """Feed every object of the session's `metrics_collected` event here."""
        kind = type(m).__name__
        if kind == "LLMMetrics":
            self.add(llm_requests=1,
                     llm_prompt_tokens=getattr(m, "prompt_tokens", 0),
                     llm_cached_tokens=getattr(m, "prompt_cached_tokens", 0),
                     llm_completion_tokens=getattr(m, "completion_tokens", 0))
        elif kind == "TTSMetrics":
            self.add(tts_chars=getattr(m, "characters_count", 0),
                     tts_audio_s=getattr(m, "audio_duration", 0))
        elif kind == "STTMetrics":
            self.add(stt_audio_s=getattr(m, "audio_duration", 0))

    # ---- output -----------------------------------------------------------
    def flush(self, reply_text: str = "") -> dict:
        """Write one row for the reply that just finished and reset the per-reply counters."""
        row = {"time": datetime.now().isoformat(timespec="seconds"), "llm": self.llm_model,
               "reply_chars": len(reply_text or "")}
        row.update({k: round(self.turn[k], 3) if k.endswith("_s") else int(self.turn[k]) for k in COUNTERS})
        # LiveKit sometimes never reports TTS metrics for a reply (interrupted / preempted). The reply
        # was still spoken, so fall back to its length and flag the row.
        row["tts_estimated"] = 0
        if row["tts_chars"] == 0 and reply_text:
            row["tts_chars"], row["tts_estimated"] = len(reply_text), 1
            self.session["tts_chars"] += len(reply_text)
        self._check_header()
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            if new:
                w.writeheader()
            w.writerow(row)
        self.turn = dict.fromkeys(COUNTERS, 0.0)
        return row

    def _check_header(self) -> None:
        """If an older log with different columns exists, keep it aside instead of writing misaligned rows."""
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return
        with open(self.path, newline="") as f:
            first = next(csv.reader(f), [])
        if first != HEADER:
            old = self.path.rsplit(".", 1)[0] + f".old-{int(time.time())}.csv"
            os.replace(self.path, old)

    def line(self, row: dict) -> str:
        """One-line console summary."""
        return (f"llm {row['llm_prompt_tokens']} in ({row['llm_cached_tokens']} cached) / "
                f"{row['llm_completion_tokens']} out, tts {row['tts_chars']} chars, "
                f"stt {row['stt_audio_s']} s (since last reply), search tavily {row['tavily_requests']} "
                f"({row['tavily_credits']} cr) exa {row['exa_requests']}")

    def session_summary(self) -> str:
        s = self.session
        return (f"[usage] session total: llm {int(s['llm_prompt_tokens'])} in / {int(s['llm_completion_tokens'])} out "
                f"tokens, tts {int(s['tts_chars'])} chars (+{int(s['prerender_tts_chars'])} filler pre-render), "
                f"stt {s['stt_audio_s']:.1f} s, tavily {int(s['tavily_credits'])} credits, "
                f"exa {int(s['exa_requests'])} requests, session length {time.monotonic() - self.started:.0f} s")

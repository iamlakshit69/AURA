"""
Usage meter for the voice agent. No LiveKit imports, so it can be tested on its own.

It collects raw usage (tokens, characters, seconds, search calls) while a reply is being made and
writes ONE ROW PER REPLY to usage_log.csv (raw usage only, no prices). On top of that it prices every
reply and the whole session using prices.ini (see pricing.py):

    usage_log.csv   raw usage per reply (unchanged format)
    cost_log.csv    one row per reply AND one row per session, with the cost at the price valid then.
                    Append-only: new runs add to the end, nothing is overwritten.
    agent.log       human-readable log of everything (see agent_logging.py)

What is counted (billing unit of each provider):
    LLM (Groq)      prompt tokens, cached prompt tokens, completion tokens   -> per million tokens
    TTS (Sarvam)    characters sent to the synthesiser (+ filler clips at startup) -> per 10K characters
    STT (Sarvam)    seconds of audio                                         -> per hour of audio
    Search          one request (Tavily: credits, Exa: requests)             -> per credit / per request

A reply with a tool call has several LLM requests (ask -> tool -> answer); they are all summed into
the row of the reply they belong to. Usage that happens but is never followed by a finished reply
(user barge-in, failed turn) is NOT lost: it stays in the counters and lands in the next row, and it is
always inside the session total.
Note: STT seconds are streamed continuously, so per-row STT is 'audio since the previous row'; trust the total.
tts_estimated=1 means LiveKit sent no TTS metrics for that reply and tts_chars is the reply length.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime

from pricing import COST_KEYS, Prices

log = logging.getLogger("voice-agent.usage")

COUNTERS = [
    "llm_requests", "llm_prompt_tokens", "llm_cached_tokens", "llm_completion_tokens",
    "tts_chars", "tts_audio_s",
    "stt_audio_s",
    "tavily_requests", "tavily_credits", "exa_requests", "weather_calls",
]
HEADER = ["time", "llm", "reply_chars"] + COUNTERS + ["tts_estimated", "tts_model", "stt_model"]
COST_HEADER = ["time", "session_id", "kind", "llm", "replies", "reply_chars",
               "llm_prompt_tokens", "llm_cached_tokens", "llm_completion_tokens",
               "tts_chars", "stt_audio_s", "tavily_credits", "exa_requests", "weather_calls",
               "cost_llm", "cost_tts", "cost_stt", "cost_search", "cost_total", "currency", "unpriced",
               "tts_model", "stt_model"]


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


class UsageMeter:
    def __init__(self, path: str = "usage_log.csv", models=None,
                 prices_path: str | None = None, cost_path: str | None = None) -> None:
        """models: {'llm': ..., 'tts': ..., 'stt': ...} (settings.get_models()); a plain string = the LLM model."""
        if isinstance(models, str):
            models = {"llm": models}
        self.models = {"llm": "", "tts": "", "stt": "", **(models or {})}
        self.path = path
        self.llm_model = self.models["llm"]
        self.cost_path = cost_path or os.getenv("COST_CSV", "cost_log.csv")
        self.prices = Prices(prices_path or os.getenv("PRICES_FILE", "prices.ini"), self.models)
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.turn = dict.fromkeys(COUNTERS, 0.0)       # since the last written row
        self.session = dict.fromkeys(COUNTERS, 0.0)    # since the agent started
        self.session["prerender_tts_chars"] = 0.0      # filler clips rendered once at startup
        self.reply_cost = dict.fromkeys(COST_KEYS + ["total"], 0.0)   # sum of the per-reply costs
        self.replies = 0
        self.external_stt = False                      # True: the wake gate reports the STT seconds it sends (any provider)
        self._ended = False
        self.started = time.monotonic()
        log.info("session %s started | llm %s | tts %s | stt %s (%s) | report currency %s | prices file %s",
                 self.session_id, self.models["llm"] or "?", self.models["tts"] or "?", self.models["stt"] or "?",
                 self.models.get("stt_provider") or "?",
                 self.currency, self.prices.path)
        todo = self.prices.missing(tavily=bool(os.getenv("TAVILY_API_KEY")), exa=bool(os.getenv("EXA_API_KEY")))
        if todo:
            log.warning("NO PRICE for: %s. Costs of these show as UNPRICED (old prices are not reused). "
                        "Add them to %s or run: python price_check.py", "; ".join(i.label for i in todo), self.prices.path)

    @property
    def currency(self) -> str:
        return self.prices.currency

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
        log.info("search request sent: %s%s", provider, f" ({credits:g} credit)" if provider == "tavily" else "")

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
        elif kind == "STTMetrics" and not self.external_stt:     # else the gate counts exactly what it sent
            self.add(stt_audio_s=getattr(m, "audio_duration", 0))

    # ---- money ------------------------------------------------------------
    def session_cost(self) -> dict:
        """Cost of everything since the agent started, fillers included. Live: safe to call any time."""
        return self.prices.cost(self.session, self.models, extra_tts_chars=self.session["prerender_tts_chars"])

    def _append(self, path: str, header: list, row: dict) -> None:
        """Append one row. If an older file has different columns it is renamed, never overwritten."""
        self._check_header(path, header)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if new:
                w.writeheader()
            w.writerow(row)

    # ---- output -----------------------------------------------------------
    def flush(self, reply_text: str = "") -> dict:
        """Write one row for the reply that just finished and reset the per-reply counters.
        The returned row also has row['cost'] = {llm, tts, stt, search, total} for this reply."""
        row = {"time": datetime.now().isoformat(timespec="seconds"), "llm": self.llm_model,
               "reply_chars": len(reply_text or "")}
        row.update({k: round(self.turn[k], 3) if k.endswith("_s") else int(self.turn[k]) for k in COUNTERS})
        # LiveKit sometimes never reports TTS metrics for a reply (interrupted / preempted). The reply
        # was still spoken, so fall back to its length and flag the row.
        row["tts_estimated"] = 0
        row["tts_model"], row["stt_model"] = self.models["tts"], self.models["stt"]
        if row["tts_chars"] == 0 and reply_text:
            row["tts_chars"], row["tts_estimated"] = len(reply_text), 1
            self.session["tts_chars"] += len(reply_text)
        self._append(self.path, HEADER, row)
        self.turn = dict.fromkeys(COUNTERS, 0.0)

        # price the reply; a pricing problem must never stop the usage log
        try:
            c = self.prices.cost(row, self.models)
            self.replies += 1
            for k in self.reply_cost:
                self.reply_cost[k] += c[k]
            self._append(self.cost_path, COST_HEADER, self._cost_row("reply", row["time"], row, c, 1, row["reply_chars"]))
            log.info("reply #%d | %s", self.replies, self.line(row, c))
        except Exception as e:
            c = {**dict.fromkeys(COST_KEYS + ["total"], 0.0), "unpriced": ["error"]}
            log.warning("cost calculation failed: %r", e)
        row["cost"] = c
        return row

    def _cost_row(self, kind: str, when: str, u: dict, c: dict, replies: int, reply_chars) -> dict:
        return {"time": when, "session_id": self.session_id, "kind": kind, "llm": self.llm_model,
                "replies": replies, "reply_chars": reply_chars,
                "llm_prompt_tokens": int(_num(u.get("llm_prompt_tokens"))),
                "llm_cached_tokens": int(_num(u.get("llm_cached_tokens"))),
                "llm_completion_tokens": int(_num(u.get("llm_completion_tokens"))),
                "tts_chars": int(_num(u.get("tts_chars"))), "stt_audio_s": round(_num(u.get("stt_audio_s")), 2),
                "tavily_credits": _num(u.get("tavily_credits")), "exa_requests": int(_num(u.get("exa_requests"))),
                "weather_calls": int(_num(u.get("weather_calls"))),
                "cost_llm": round(c["llm"], 6), "cost_tts": round(c["tts"], 6), "cost_stt": round(c["stt"], 6),
                "cost_search": round(c["search"], 6), "cost_total": round(c["total"], 6), "currency": self.currency,
                "unpriced": ";".join(c.get("unpriced", [])), "tts_model": self.models["tts"], "stt_model": self.models["stt"]}

    def end_session(self) -> str:
        """Call once when the session ends: writes the session row + log lines, returns the summary text."""
        if self._ended:
            return ""
        self._ended = True
        text = self.session_summary()
        try:
            c = self.session_cost()
            u = dict(self.session)
            u["tts_chars"] += u["prerender_tts_chars"]
            self._append(self.cost_path, COST_HEADER,
                         self._cost_row("session", datetime.now().isoformat(timespec="seconds"), u, c, self.replies, ""))
            other = c["total"] - self.reply_cost["total"]
            log.info("session %s ended | %d replies | %s", self.session_id, self.replies, text)
            log.info("session %s cost | llm %s tts %s stt %s search %s | TOTAL %s | replies %s + fillers/unfinished turns %s",
                     self.session_id, self.money(c["llm"]), self.money(c["tts"]), self.money(c["stt"]),
                     self.money(c["search"]), self.cost_text(c), self.money(self.reply_cost["total"]),
                     self.money(max(other, 0.0)))
        except Exception as e:
            log.warning("session cost failed: %r", e)
        return text

    def _check_header(self, path: str, header: list) -> None:
        """Different columns in an existing log: if we only ADDED columns, upgrade the file in place (every old
        row is kept, new columns are empty). Anything else is renamed aside, never overwritten."""
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        with open(path, newline="", encoding="utf-8-sig") as f:
            first = next(csv.reader(f), [])
        if first == header:
            return
        if set(first) <= set(header):
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, restval="", extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, path)
            log.info("%s: added new columns, %d old rows kept", path, len(rows))
        else:
            old = path.rsplit(".", 1)[0] + f".old-{int(time.time())}.csv"
            os.replace(path, old)
            log.info("%s has other columns: kept as %s", path, old)

    def money(self, v: float) -> str:
        return f"{v:.5f} {self.currency}" if abs(v) < 1 else f"{v:.2f} {self.currency}"

    def cost_text(self, c: dict) -> str:
        """Total, plus a flag when something used had no price (it is NOT priced with an old price)."""
        unpriced = c.get("unpriced")
        return self.money(c["total"]) + (f" + UNPRICED {', '.join(unpriced)}" if unpriced else "")

    def line(self, row: dict, cost: dict | None = None) -> str:
        """One-line console summary (with the cost of the reply when known)."""
        cost = cost or row.get("cost")
        s = (f"llm {row['llm_prompt_tokens']} in ({row['llm_cached_tokens']} cached) / "
             f"{row['llm_completion_tokens']} out, tts {row['tts_chars']} chars, "
             f"stt {row['stt_audio_s']} s (since last reply), search tavily {row['tavily_requests']} "
             f"({row['tavily_credits']} cr) exa {row['exa_requests']}")
        if cost:
            s += f" | COST {self.cost_text(cost)} (llm {cost['llm']:.5f}, tts {cost['tts']:.5f}, stt {cost['stt']:.5f}, search {cost['search']:.5f})"
        return s

    def session_summary(self) -> str:
        s = self.session
        c = self.session_cost()
        return (f"[usage] session total: llm {int(s['llm_prompt_tokens'])} in / {int(s['llm_completion_tokens'])} out "
                f"tokens, tts {int(s['tts_chars'])} chars (+{int(s['prerender_tts_chars'])} filler pre-render), "
                f"stt {s['stt_audio_s']:.1f} s, tavily {int(s['tavily_credits'])} credits, "
                f"exa {int(s['exa_requests'])} requests, session length {time.monotonic() - self.started:.0f} s"
                f" | SESSION COST {self.cost_text(c)} (llm {c['llm']:.4f}, tts {c['tts']:.4f}, "
                f"stt {c['stt']:.4f}, search {c['search']:.4f})")

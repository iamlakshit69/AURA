"""
Prices and the cost formula. No LiveKit imports, so it can be tested on its own.

Prices live in prices.ini and are KEYED BY MODEL NAME:  [llm:<model>]  [tts:<model>]  [stt:<model>].
The model names come from settings.env (see settings.py). A model with no price there is never priced
with another model's numbers: its cost shows as UNPRICED (0) until you add the price (price_check.py
asks for it at startup). Switching back to a model you priced earlier uses that earlier price again.
The file is re-read whenever it changes, so you can edit prices while the agent runs.
"""
from __future__ import annotations

import configparser
import logging
import os
import re
from dataclasses import dataclass, field

log = logging.getLogger("voice-agent.pricing")

COST_KEYS = ["llm", "tts", "stt", "search"]       # "search" = web search + weather


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Item:
    """One thing that needs a price."""
    section: str                      # section name in prices.ini
    label: str                        # for people
    fields: list                      # [(key, question)]
    required: tuple                   # keys that must be filled in
    currency: str                     # suggested currency
    current: dict = field(default_factory=dict)


LLM_FIELDS = [("input_per_million", "price per 1M input tokens"),
              ("cached_input_per_million", "price per 1M CACHED input tokens (Enter = same as input)"),
              ("output_per_million", "price per 1M output tokens")]


class Prices:
    def __init__(self, path: str = "prices.ini", models: dict | None = None) -> None:
        self.path = path
        self.models = {"llm": "", "tts": "", "stt": ""}
        self.models.update({k: v for k, v in (models or {}).items() if v})
        self._mtime: float | None = None
        self._cp: configparser.ConfigParser | None = None
        self._warned: set[str] = set()
        self.currency = "?"
        self.reload()

    # ---- file handling ----------------------------------------------------
    def reload(self) -> None:
        """Re-read prices.ini if it changed. A broken edit keeps the last good prices."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            if self._cp is None:
                self._cp = configparser.ConfigParser()
                self._warn_once("missing", f"{self.path} not found: nothing is priced, only usage is tracked")
            return
        if mtime == self._mtime and self._cp is not None:
            return
        cp = configparser.ConfigParser(delimiters=("=",), inline_comment_prefixes=("#", ";"), interpolation=None)
        cp.optionxform = str.lower
        try:
            with open(self.path, encoding="utf-8-sig") as f:
                cp.read_file(f)
        except (configparser.Error, OSError) as e:
            self._warn_once(f"parse{mtime}", f"cannot read {self.path} ({e}); keeping the previous prices")
            self._mtime = mtime
            if self._cp is None:
                self._cp = configparser.ConfigParser()
            return
        first = self._cp is None
        self._cp, self._mtime = cp, mtime
        self.currency = cp.get("general", "currency", fallback="?").strip() or "?"
        if not first:
            log.info("prices reloaded from %s", self.path)

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg)

    # ---- lookups ----------------------------------------------------------
    def _raw(self, section: str, key: str) -> str:
        cp = self._cp
        return cp.get(section, key, fallback="").strip() if cp is not None else ""

    def _rate(self, currency: str) -> float:
        currency = (currency or self.currency).strip()
        if currency.upper() == self.currency.upper():
            return 1.0
        return num(self._raw("exchange_rates", currency.lower()))

    def _money(self, section: str, key: str) -> float:
        """Price in the report currency (0 if blank)."""
        raw = self._raw(section, key)
        value = num(raw) if raw != "" else 0.0
        return value * self._rate(self._raw(section, "currency")) if value else 0.0

    def is_priced(self, item: Item) -> bool:
        """True when every required price is filled in (0 = free counts) and its currency can be converted."""
        for key in item.required:
            raw = self._raw(item.section, key)
            try:
                float(raw)
            except ValueError:
                return False
        value_is_money = any(num(self._raw(item.section, k)) for k in item.required)
        if value_is_money and self._rate(self._raw(item.section, "currency")) == 0:
            self._warn_once(f"rate-{item.section}",
                            f"[{item.section}] is priced in {self._raw(item.section, 'currency')} but "
                            f"[exchange_rates] has no rate for it: shown as UNPRICED")
            return False
        return True

    # ---- what needs a price ----------------------------------------------
    def items(self, models: dict | None = None, tavily: bool = False, exa: bool = False) -> list[Item]:
        m = {**self.models, **{k: v for k, v in (models or {}).items() if v}}
        out = [Item(f"llm:{m['llm']}", f"LLM model '{m['llm']}' (Groq)", LLM_FIELDS,
                    ("input_per_million", "output_per_million"), "USD"),
               Item(f"tts:{m['tts']}", f"TTS model '{m['tts']}' (Sarvam)",
                    [("per_10k_chars", "price per 10,000 characters")], ("per_10k_chars",), "INR"),
               Item(f"stt:{m['stt']}", f"STT model '{m['stt']}' ({(m.get('stt_provider') or 'sarvam').capitalize()})",
                    [("per_hour", "price per hour of audio")], ("per_hour",),
                    "INR" if (m.get("stt_provider") or "sarvam") == "sarvam" else "USD")]
        if tavily:
            out.append(Item("search_tavily", "Tavily web search", [("per_credit", "price per credit")], ("per_credit",), "USD"))
        if exa:
            out.append(Item("search_exa", "Exa web search", [("per_request", "price per request")], ("per_request",), "USD"))
        for it in out:
            it.current = {k: self._raw(it.section, k) for k, _ in it.fields}
            it.current["currency"] = self._raw(it.section, "currency")
        return out

    def missing(self, models: dict | None = None, tavily: bool = False, exa: bool = False) -> list[Item]:
        self.reload()
        return [it for it in self.items(models, tavily, exa) if not self.is_priced(it)]

    def known_models(self, kind: str) -> list[str]:
        """Models of this kind that already have a price in the file (for the prompt)."""
        self.reload()
        if self._cp is None:
            return []
        return [s[len(kind) + 1:] for s in self._cp.sections()
                if s.startswith(kind + ":") and self.is_priced(Item(s, "", [], self._required(kind), ""))]

    @staticmethod
    def _required(kind: str) -> tuple:
        return {"llm": ("input_per_million", "output_per_million"), "tts": ("per_10k_chars",), "stt": ("per_hour",)}[kind]

    # ---- writing (keeps your comments) -----------------------------------
    def save(self, section: str, values: dict) -> None:
        """Write values into [section] of prices.ini, creating it if needed. Comments are preserved."""
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            lines = ["[general]", "currency = INR", "", "[exchange_rates]", "USD = 0"]
        header = f"[{section}]"
        if not any(l.strip() == header for l in lines):
            lines += ["", header]
        for key, value in values.items():
            start = next(i for i, l in enumerate(lines) if l.strip() == header)
            end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
            hit = next((i for i in range(start + 1, end) if re.match(rf"\s*{re.escape(key)}\s*=", lines[i], re.I)), None)
            if hit is not None:
                lines[hit] = f"{key} = {value}"
            else:
                lines.insert(start + 1, f"{key} = {value}")
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.path)
        self._mtime = None               # force a re-read
        self.reload()
        log.info("prices saved to %s: [%s]", self.path, section)

    # ---- the formula ------------------------------------------------------
    def cost(self, c: dict, models: dict | None = None, extra_tts_chars: float = 0.0) -> dict:
        """c: a usage record (CSV row or the session counters). Returns llm/tts/stt/search/total in the
        report currency, plus 'unpriced': what was used but has no price (it counts as 0, never as an old price).
        extra_tts_chars: characters billed but not in any reply (filler clips)."""
        self.reload()
        g = lambda k: num(c.get(k))
        m = {**self.models, **{k: v for k, v in (models or {}).items() if v}}
        by_section = {it.section: it for it in self.items(m, tavily=True, exa=True)}
        unpriced: list[str] = []

        def priced(section: str, used: float, what: str) -> bool:
            if not used:
                return False
            if self.is_priced(by_section[section]):
                return True
            unpriced.append(what)
            return False

        llm = tts = stt = search = 0.0
        if priced(f"llm:{m['llm']}", g("llm_prompt_tokens") + g("llm_completion_tokens"), f"llm:{m['llm']}"):
            s = f"llm:{m['llm']}"
            p_in, p_out = self._money(s, "input_per_million"), self._money(s, "output_per_million")
            p_cached = self._money(s, "cached_input_per_million") if self._raw(s, "cached_input_per_million") else p_in
            cached = g("llm_cached_tokens")
            fresh = max(g("llm_prompt_tokens") - cached, 0.0)      # prompt_tokens includes the cached ones
            llm = (fresh * p_in + cached * p_cached + g("llm_completion_tokens") * p_out) / 1e6
        chars = g("tts_chars") + extra_tts_chars
        if priced(f"tts:{m['tts']}", chars, f"tts:{m['tts']}"):
            tts = chars / 1e4 * self._money(f"tts:{m['tts']}", "per_10k_chars")
        if priced(f"stt:{m['stt']}", g("stt_audio_s"), f"stt:{m['stt']}"):
            stt = g("stt_audio_s") / 3600 * self._money(f"stt:{m['stt']}", "per_hour")
        if priced("search_tavily", g("tavily_credits"), "tavily"):
            search += g("tavily_credits") * self._money("search_tavily", "per_credit")
        if priced("search_exa", g("exa_requests"), "exa"):
            search += g("exa_requests") * self._money("search_exa", "per_request")
        search += g("weather_calls") * self._money("weather_open_meteo", "per_call")     # free unless you set a price
        out = {"llm": llm, "tts": tts, "stt": stt, "search": search}
        out["total"] = sum(out[k] for k in COST_KEYS)
        out["unpriced"] = unpriced
        return out

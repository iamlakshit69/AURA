#!/usr/bin/env python3
"""
LiveKit version of the voice latency demo: same providers, but LiveKit Agents runs the loop
(turn-taking, barge-in, preemptive generation, streaming) instead of hand-written threads.

Stack (all open source on the LiveKit side, Apache 2.0):
    STT  Sarvam realtime (saaras:v3-realtime). Sarvam's own server-side VAD decides end of turn.
    LLM  Groq, same model as before (GROQ_LLM_MODEL), reasoning off.
    TTS  Sarvam Bulbul v4 Flash. The voice follows the language Sarvam detects in your speech.
    Tool web_search (Tavily, optionally raced against Exa) and get_weather (Open-Meteo, no key).
         Pre-rendered "let me check" clips play while a search runs. The search layer lives in
         search_providers.py (keep it next to this file).

Install (Python 3.10+):
    pip install -r requirements-livekit.txt
    (Linux also needs: sudo apt install libportaudio2)
    Optional local VAD for faster barge-in:  pip install livekit-plugins-silero   and set LOCAL_VAD=1

Run, talking through your own mic and speakers. Use headphones: with open speakers the agent
hears itself and interrupts itself.
    python Structure-II.py console

Settings: models, voices and timings are in settings.env; .env holds only the keys. Prices are in
prices.ini, keyed by the model names from settings.env: a new model has no price, so the agent asks for it
at startup (or run: python price_check.py) instead of reusing the old model's price.
Keys (.env): GROQ_API_KEY, SARVAM_API_KEY, plus:
    TAVILY_API_KEY=tvly-...        # web search; without it the agent says search is unavailable
    EXA_API_KEY=...                # optional: second provider, raced against Tavily (first answer wins)
Optional tuning (defaults shown):
    SARVAM_VAD_SILENCE_MS=300      # Sarvam's end-of-turn silence window
    ENDPOINT_MIN_DELAY_S=0.25      # extra hold after that; the two add up
    FILLER_DELAY_S=0.3             # skip the filler if the search comes back faster than this
    START_LANG=hi                  # voice used until the first reply: hi, pa or en
    REPLY_LANG=auto                # auto = answer in the language you speak; hi / pa / en = always answer in that
                                   # language. Changeable while running: L key in the dashboard, or just ask
                                   # ("talk to me in Punjabi", "हिंदी में बोलो", ...): the model calls set_language.
    ECHO_GUARD=filter              # (default is now filter) stops the agent from hearing itself on open speakers:
                                   #   gate   = mic closed while it speaks (no voice barge-in; press S to stop it)
                                   #   filter = mic stays open (voice barge-in), but turns that repeat its own words are dropped
                                   #   off    = nothing (use headphones)
    ECHO_TAIL_S=0.6                # gate mode: keep the mic closed this long after it stops speaking (room reverb)
    ECHO_WINDOW_S=15               # how far back its own words count as "echo candidates"
    SARVAM_VAD_THRESHOLD=0.5       # speech-onset probability (Sarvam's default 0.3 also triggers on quiet echo)
    SARVAM_VAD_MIN_SPEECH_MS=80
    SPEAK_LANG_CONFIRM=1           # after a menu change, speak a short confirmation in the new language
    SPEAKER_HI / SPEAKER_PA / SPEAKER_EN   # Bulbul v4-flash persona per language (see VOICE CHOICES below)
Search tuning (defaults shown):
    SEARCH_TIMEOUT_S=3.0           # hard cap for the WHOLE search step (prefetch wait + retry included)
    SEARCH_MAX_RESULTS=3           # snippets handed to the model
    SEARCH_SNIPPET_CHARS=300       # characters per snippet
    TAVILY_DEPTH=ultra-fast        # ultra-fast | fast | basic (falls back to basic if Tavily rejects it)
    HEDGE_SAME_S=0                 # >0: with ONE provider, re-send the query if no answer by then
    PREFETCH_MODE=intent           # intent | all | off: start searching before the model asks
    LONG_WAIT_S=2.0                # second "still looking" filler after this many seconds
    KEEPALIVE_S=40                 # how often the search connections are touched to stay warm
    DEFAULT_CITY=Ludhiana          # used by get_weather when the user names no city
New: LLM_TIMEOUT_S=4 (per LLM attempt), LOCAL_VAD=0|1 (default OFF), PREEMPTIVE=1|0,
INTERRUPT_MIN_WORDS=1, INTERRUPT_MIN_DURATION_S=0.5, FALSE_INTERRUPT_S=1.3.
Existing variables still honoured: GROQ_LLM_MODEL, MAX_REPLY_TOKENS, TTS_MODEL, SPEAKER_HI,
SPEAKER_EN, SPEAKER_PA, SARVAM_RT_LANGUAGE, SARVAM_RT_STREAM_TYPE, SARVAM_RT_TASK.

Timings go to latency_log_livekit.csv (new file, different columns from latency_log.csv).

Money: put your prices in prices.ini (every provider used is listed there). After every reply the cost
is printed and appended to cost_log.csv; when the session ends its total cost is written too.
Everything is also appended to agent.log (never truncated). Report on history: python cost_report.py
"""
import asyncio
import collections
import csv
import difflib
import inspect
import logging
import os
import random
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
try:
    from livekit.agents import StopResponse       # raised in on_user_turn_completed to skip the reply
except ImportError:                               # very old versions: echo turns are still logged, just not skipped
    StopResponse = None
from livekit.plugins import groq, sarvam

from search_providers import Prefetcher, Search, Weather
from usage_meter import UsageMeter
from agent_logging import setup_logging
from settings import get_models, load_settings

load_settings()                 # settings.env: models, voices, timings (no secrets)
load_dotenv()                   # .env: API keys only
load_dotenv(".env.local")

logger = logging.getLogger("voice-agent")
setup_logging()                 # agent.log: append-only log of everything (LOG_FILE / LOG_LEVEL to change)


def env(key, default=""):
    return os.getenv(key, default).strip()


# ---------- config (same names as env.example where they exist) ----------
MODELS = get_models()           # the model names; the price calculator reads the same ones
LLM_MODEL = MODELS["llm"]
MAX_REPLY_TOKENS = int(env("MAX_REPLY_TOKENS", "500"))       # Hindi/Punjabi text costs many tokens per character
TTS_MODEL = MODELS["tts"]
STT_MODEL = MODELS["stt"]
REASONING_EFFORT = env("GROQ_REASONING_EFFORT", "low")
STT_LANGUAGE = env("SARVAM_RT_LANGUAGE", "auto")          # auto, hi-IN, pa-IN, en-IN
STT_STREAM_TYPE = env("SARVAM_RT_STREAM_TYPE", "fast")    # fast | balanced | simulated
STT_TASK = env("SARVAM_RT_TASK", "transcribe")            # transcribe | codemix | translate ...
VAD_SILENCE_MS = int(env("SARVAM_VAD_SILENCE_MS", "300"))
VAD_THRESHOLD = float(env("SARVAM_VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(env("SARVAM_VAD_MIN_SPEECH_MS", "80"))
ECHO_GUARD = env("ECHO_GUARD", "filter").lower()
if ECHO_GUARD not in ("gate", "filter", "off"):
    ECHO_GUARD = "gate"
ECHO_TAIL_S = float(env("ECHO_TAIL_S", "0.6"))
ECHO_WINDOW_S = float(env("ECHO_WINDOW_S", "15"))
MIN_DELAY_S = float(env("ENDPOINT_MIN_DELAY_S", "0.25"))
FILLER_DELAY_S = float(env("FILLER_DELAY_S", "0.3"))
MEMORY_DB = env("MEMORY_DB", "memory.db")                 # everything stays on this machine
MAX_HISTORY_ITEMS = int(env("MAX_HISTORY_ITEMS", "20"))   # chat items kept between turns (keeps Groq TPM low)
TAVILY_KEY = env("TAVILY_API_KEY")
EXA_KEY = env("EXA_API_KEY")
SEARCH_TIMEOUT_S = float(env("SEARCH_TIMEOUT_S", "3.0"))
SEARCH_MAX_RESULTS = int(env("SEARCH_MAX_RESULTS", "3"))
SEARCH_SNIPPET_CHARS = int(env("SEARCH_SNIPPET_CHARS", "300"))
TAVILY_DEPTH = env("TAVILY_DEPTH", "ultra-fast")
HEDGE_SAME_S = float(env("HEDGE_SAME_S", "0"))
PREFETCH_MODE = env("PREFETCH_MODE", "intent")
LONG_WAIT_S = float(env("LONG_WAIT_S", "2.0"))
KEEPALIVE_S = float(env("KEEPALIVE_S", "40"))
DEFAULT_CITY = env("DEFAULT_CITY", "Ludhiana")
LLM_TIMEOUT_S = float(env("LLM_TIMEOUT_S", "4.0"))
LOCAL_VAD = env("LOCAL_VAD", "0") == "1"                     # OFF = your old behaviour (interrupt on a transcribed word). 1 = bundled Silero VAD: faster, but can react to echo
INTERRUPT_MIN_WORDS = int(env("INTERRUPT_MIN_WORDS", "1"))   # real words needed to interrupt (as in your old file); 0 only with LOCAL_VAD=1
INTERRUPT_MIN_DURATION_S = float(env("INTERRUPT_MIN_DURATION_S", "0.5"))   # speech length that counts as a barge-in
FALSE_INTERRUPT_S = float(env("FALSE_INTERRUPT_S", "1.3"))   # noise only: resume talking after this much silence
PREEMPTIVE = env("PREEMPTIVE", "1") != "0"
SEARCH_FAILED_MSG = "The search failed or timed out. Tell the user in a few words that you could not check right now."
TTS_RATE = 24000

CODES = {"hi": "hi-IN", "pa": "pa-IN", "en": "en-IN"}

# VOICE CHOICES. bulbul:v4-flash voices are personas named <name>_<language>_<style> and each one is recorded
# in ONE language, so the voice must match the language of the text. Roster at
# https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/voices (v3 short names like "shubh" are rejected).
#   hi: aparna_hi_customer (Sarvam's top pick for Hindi), simran_hi_conversation, sarika_hi_conversation,
#       roopa_hi_conversational, rahul_hi_conversational (male), shubh_hi_customer (male)
#   pa: anand_pa_conversation, anand_pa_customer, harpreet_pa_narration, jaspal_pa_banking  (all male: the only
#       Punjabi personas that exist today)
#   en: simran_en_customer, simran_en_conversation, aparna_en_companion, dev_en_conversational (male)
SPEAKERS = {
    "hi": env("SPEAKER_HI", "aparna_hi_customer"),
    "pa": env("SPEAKER_PA", "anand_pa_conversation"),
    "en": env("SPEAKER_EN", "simran_en_customer"),
}
_start = env("START_LANG") or (STT_LANGUAGE[:2] if STT_LANGUAGE[:2] in CODES else "hi")
START_LANG = _start if _start in CODES else "hi"

# Reply language: "auto" follows the user, "hi" / "pa" / "en" pins it.
REPLY_LANG = env("REPLY_LANG", "auto").lower()
if REPLY_LANG not in ("auto", *CODES):
    REPLY_LANG = "auto"
INITIAL_LANG = REPLY_LANG if REPLY_LANG in CODES else START_LANG      # voice loaded into the TTS at startup
SPEAK_LANG_CONFIRM = env("SPEAK_LANG_CONFIRM", "1") != "0"

LANG_NAMES = {"hi": ("Hindi", "Devanagari"), "pa": ("Punjabi", "Gurmukhi"), "en": ("English", "")}
UI_LANGUAGES = [("auto", "Auto", "follow my language"), ("hi", "Hindi", "हिन्दी"),
                ("pa", "Punjabi", "ਪੰਜਾਬੀ"), ("en", "English", "English")]

# What the model may pass to set_language, in any of these spellings (the model also translates the user's wording).
_LANG_ALIASES = {
    "auto": "auto", "automatic": "auto", "same": "auto", "follow": "auto", "default": "auto", "mirror": "auto",
    "hi": "hi", "hindi": "hi", "hin": "hi", "hi-in": "hi", "हिंदी": "hi", "हिन्दी": "hi", "ਹਿੰਦੀ": "hi",
    "pa": "pa", "punjabi": "pa", "panjabi": "pa", "pan": "pa", "pa-in": "pa", "पंजाबी": "pa", "ਪੰਜਾਬੀ": "pa",
    "en": "en", "english": "en", "eng": "en", "en-in": "en", "en-us": "en", "अंग्रेज़ी": "en", "अंग्रेजी": "en",
    "ਅੰਗਰੇਜ਼ੀ": "en", "ਅੰਗਰੇਜ਼ੀ": "en",
}


def normalize_language(value: str) -> Optional[str]:
    """'Hindi' / 'hi-IN' / 'ਪੰਜਾਬੀ' / 'auto' -> 'hi' / 'pa' / 'auto'; None if it is not one we can speak."""
    v = (value or "").strip().lower().replace("_", "-")
    return _LANG_ALIASES.get(v) or _LANG_ALIASES.get(v.split("-")[0])


# Script detection. The voice is chosen from the script of the reply text itself, so the voice always matches
# what the model actually wrote, whatever language the speech recogniser thought it heard.
# U+0964 / U+0965 (the danda) are shared by Devanagari and Gurmukhi text, so they do not count for either.
_GURMUKHI = re.compile(r"[\u0A00-\u0A7F]")
_DEVANAGARI = re.compile(r"[\u0900-\u0963\u0966-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")
ROUTE_MIN_LATIN = 6        # Latin letters to see before an English reply is trusted (an Indic letter decides at once)


_WORD_SPLIT = re.compile(r"[\s\.,;:!?\"'()\[\]{}|\u0964\u0965\-\u2013\u2014\u2026]+")


def _words(text: str) -> list[str]:
    """Lower-case words. Splits on spaces and punctuation only, so Devanagari / Gurmukhi words stay whole."""
    return [w for w in _WORD_SPLIT.split((text or "").lower()) if w]


def script_lang(text: str) -> Optional[str]:
    """'pa' for Gurmukhi, 'hi' for Devanagari, 'en' for Latin-only text, None if there are no letters yet."""
    g, d, l = len(_GURMUKHI.findall(text)), len(_DEVANAGARI.findall(text)), len(_LATIN.findall(text))
    if g or d:
        return "pa" if g >= d else "hi"       # an Indic letter beats English loan words inside the sentence
    return "en" if l else None


# Spoken after a change from the dashboard menu (gender-neutral wording, edit freely).
CONFIRM_TEXT = {
    "hi": "ठीक है, अब हिंदी में बात होगी।",
    "pa": "ਠੀਕ ਹੈ, ਹੁਣ ਪੰਜਾਬੀ ਵਿੱਚ ਗੱਲ ਕਰਦੇ ਹਾਂ।",
    "en": "Okay, switching to English.",
}
CONFIRM_AUTO = {
    "hi": "ठीक है, आप जिस भाषा में बोलेंगे, उसी में जवाब मिलेगा।",
    "pa": "ਠੀਕ ਹੈ, ਤੁਸੀਂ ਜਿਸ ਭਾਸ਼ਾ ਵਿੱਚ ਬੋਲੋਗੇ, ਉਸੇ ਵਿੱਚ ਜਵਾਬ ਮਿਲੇਗਾ।",
    "en": "Okay, I will reply in whichever language you speak.",
}

# Gender-neutral wording, so it fits any speaker voice.
FILLER_TEXT = {
    "en": ["Okay, let me check.", "One sec, checking that."],
    "hi": ["ठीक है, एक सेकंड, चेक हो रहा है।", "एक सेकंड, अभी देखते हैं।"],
    "pa": ["ਠੀਕ ਹੈ, ਇੱਕ ਸਕਿੰਟ, ਚੈੱਕ ਹੋ ਰਿਹਾ ਹੈ।", "ਇੱਕ ਸਕਿੰਟ, ਹੁਣੇ ਵੇਖਦੇ ਹਾਂ।"],
}

# Played if a search is still running after LONG_WAIT_S, so the line never goes quiet for long.
LONG_FILLER_TEXT = {
    "en": ["Still looking, one more moment.", "Just a bit more, still checking."],
    "hi": ["अभी भी चेक हो रहा है, बस थोड़ा और।", "थोड़ा और समय लगेगा।"],
    "pa": ["ਅਜੇ ਵੀ ਚੈੱਕ ਹੋ ਰਿਹਾ ਹੈ, ਬਸ ਥੋੜ੍ਹਾ ਹੋਰ।", "ਥੋੜ੍ਹਾ ਹੋਰ ਸਮਾਂ ਲੱਗੇਗਾ।"],
}


_tasks: set = set()


def spawn(coro, name: str = "bg"):
    """Fire-and-forget that keeps a reference (so it is not garbage collected) and logs failures."""
    t = asyncio.ensure_future(coro)
    _tasks.add(t)

    def _done(task):
        _tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("background task %s failed: %r", name, task.exception())
    t.add_done_callback(_done)
    return t


class Memory:
    """Local memory. Hot path rule: nothing here touches a normal turn.
    - facts: small, loaded once at startup, shown to the model in its instructions
    - turns: full log of the conversation, searched only when the recall tool is called
    Writes happen after the reply, in a background thread."""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self.session_start = time.time()      # recall must not return the current conversation
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS facts(key TEXT PRIMARY KEY, value TEXT, ts REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, ts REAL, role TEXT, text TEXT)")
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(text, content='turns', "
                            "content_rowid='id', tokenize='unicode61')")
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        self.db.commit()
        self.facts = {k: v for k, v in self.db.execute("SELECT key, value FROM facts")}

    def facts_block(self) -> str:
        if not self.facts:
            return ""
        return " Known about the user: " + "; ".join(f"{k}: {v}" for k, v in self.facts.items()) + "."

    def set_fact(self, key: str, value: str) -> None:
        key = re.sub(r"[^\w]+", "_", key.strip().lower()).strip("_")[:40] or "note"   # name / Name / user-name -> name
        value = value.strip()[:120]
        with self._lock:
            self.db.execute("INSERT INTO facts(key, value, ts) VALUES(?,?,?) ON CONFLICT(key) "
                            "DO UPDATE SET value=excluded.value, ts=excluded.ts", (key, value, time.time()))
            self.db.commit()
        self.facts[key] = value

    def add_turn(self, role: str, text: str) -> None:
        with self._lock:
            cur = self.db.execute("INSERT INTO turns(ts, role, text) VALUES(?,?,?)", (time.time(), role, text))
            if self.fts:
                self.db.execute("INSERT INTO turns_fts(rowid, text) VALUES(?,?)", (cur.lastrowid, text))
            self.db.commit()

    _STOP = frozenset("a an the i me my we you your is am are was were be do did does what when where who why how "
                      "to of in on at for and or about tell said say asked ask yesterday earlier before last time "
                      "that this it with".split())

    def _rows(self, words: list[str], joiner: str, limit: int):
        if self.fts:
            match = joiner.join(f'"{w}"' for w in words)
            return self.db.execute(
                "SELECT t.ts, t.role, t.text FROM turns_fts f JOIN turns t ON t.id = f.rowid "
                "WHERE turns_fts MATCH ? AND t.ts < ? ORDER BY bm25(turns_fts), t.ts DESC LIMIT ?",
                (match, self.session_start, limit)).fetchall()
        where = joiner.join("text LIKE ?" for _ in words)
        return self.db.execute(f"SELECT ts, role, text FROM turns WHERE ({where}) AND ts < ? ORDER BY ts DESC LIMIT ?",
                               (*[f"%{w}%" for w in words], self.session_start, limit)).fetchall()

    def search(self, query: str, limit: int = 3) -> str:
        words = re.findall(r"\w+", query, flags=re.UNICODE)
        keep = [w for w in words if w.lower() not in self._STOP] or words
        if not keep:
            return "Nothing found."
        with self._lock:
            rows = self._rows(keep, " AND ", limit) if len(keep) > 1 else []
            if not rows:
                rows = self._rows(keep, " OR ", limit)
        if not rows:
            return "Nothing found about that."
        return "\n".join(f"{datetime.fromtimestamp(ts):%d %b %H:%M} {role}: {text[:160]}"
                         for ts, role, text in rows)


INSTRUCTIONS = (
    "You are a voice assistant. Your reply is spoken aloud, so use plain text only: no markdown, "
    "lists, emojis or links. "
    "{language}"
    "If the user asks you to change the language you speak, in ANY wording or language (for example "
    "'talk to me in Punjabi', 'speak Hindi', 'switch to English', 'ਹਿੰਦੀ ਵਿੱਚ ਗੱਲ ਕਰੋ', 'अंग्रेज़ी में बोलो', "
    "'go back to my own language'), call the set_language tool at once without saying anything first, "
    "then confirm the change in one short sentence in the new language. "
    "Do not call it when the user only wants a phrase translated or one word explained. "
    "Maximum 3-4 short sentences, under 240 characters. "
    "Only go into detail if the user asks you to continue or asks a detailed question. "
    "For weather or temperature questions call the get_weather tool. "
    "For anything else current (news, prices, scores, recent events) call the web_search tool "
    "instead of guessing. Call the tool right away without saying anything first. "
    "After the results, answer in one or two short sentences and never read out links. "
    "Use remember only for lasting personal facts the user states, and recall only when the user "
    "asks about an earlier conversation. "
    "Today is {today}."
)



def language_clause(mode: str) -> str:
    """The language rule that goes into the model's instructions."""
    if mode in LANG_NAMES:
        name, script = LANG_NAMES[mode]
        where = f" written in {script} script" if script else ""
        return (f"Always reply in {name}{where}, whatever language the user speaks, keeping English words in "
                f"Latin script. Keep doing that until the user asks for another language. ")
    return ("Reply in the language the user spoke: Punjabi in Gurmukhi script, Hindi in Devanagari script, "
            "English in English. If the user speaks Punjabi or Hindi, you speak in that language too, "
            "keeping English words in Latin script. ")


UI_ENABLED = env("UI", "1") != "0"                    # colorful terminal dashboard + mute key (needs: pip install rich)
USAGE_CSV = env("USAGE_CSV", "usage_log.csv")      # one row per reply: tokens, characters, seconds, search calls
USAGE_FLUSH_DELAY_S = float(env("USAGE_FLUSH_DELAY_S", "1.2"))
CSV_PATH = "latency_log_livekit.csv"
CSV_HEADER = ["stt", "llm", "transcription_ms", "end_of_turn_ms", "llm_ttft_ms", "tts_ttfb_ms",
              "e2e_ms", "total_ms_comparable", "search_ms"]


async def _one_frame(frame: rtc.AudioFrame):
    yield frame


def _ms(value):
    return round(value * 1000) if isinstance(value, (int, float)) else ""


class VoiceAgent(Agent):
    def __init__(self) -> None:
        self.mem = Memory(MEMORY_DB)
        self.facts_dirty = False
        self._date = datetime.now().date()
        self.lang_mode = REPLY_LANG                  # auto | hi | pa | en: what the model is told to answer in
        super().__init__(instructions=self._build_instructions())
        self.lang = INITIAL_LANG                     # language of the fillers; follows the user in auto mode
        self.voice_lang = INITIAL_LANG               # language whose voice is loaded in the TTS right now
        self.tts_engine = None                       # the sarvam.TTS object, set in entrypoint
        self.said: collections.deque = collections.deque(maxlen=200)    # (time, text) of what it recently said aloud
        self.speaking = False
        self.last_said = 0.0
        self.fillers: dict[str, list[tuple[str, rtc.AudioFrame]]] = {}   # filled in the background
        self.fillers_long: dict[str, list[tuple[str, rtc.AudioFrame]]] = {}
        self.last_search_ms = None
        self.ui = None                      # set in entrypoint when the dashboard is on
        self.meter = UsageMeter(USAGE_CSV, MODELS)
        self.search = Search(on_call=self.meter.search_call, tavily_key=TAVILY_KEY, exa_key=EXA_KEY, timeout=SEARCH_TIMEOUT_S,
                             max_results=SEARCH_MAX_RESULTS, snippet_chars=SEARCH_SNIPPET_CHARS,
                             tavily_depth=TAVILY_DEPTH, hedge_same_s=HEDGE_SAME_S)
        self.weather = Weather(self.search.http, timeout=SEARCH_TIMEOUT_S)
        self.prefetch = Prefetcher(self.search, mode=PREFETCH_MODE)
        self._bg: list[asyncio.Task] = []

    def _build_instructions(self) -> str:
        return (INSTRUCTIONS.format(today=datetime.now().strftime("%A, %d %B %Y"), language=language_clause(self.lang_mode))
                + self.mem.facts_block())

    # --- language + voice ---
    def voice_label(self, lang: str) -> str:
        return f"{CODES[lang]} · {SPEAKERS[lang]}"

    def apply_voice(self, lang: str) -> None:
        """Load the voice of `lang` into the TTS. Takes effect from the next synthesis."""
        if lang not in CODES or lang == self.voice_lang or self.tts_engine is None:
            return
        try:
            self.tts_engine.update_options(target_language_code=CODES[lang], speaker=SPEAKERS[lang])
        except Exception as e:
            logger.warning("voice switch to %s failed: %s", lang, e)
            return
        self.voice_lang = lang
        if self.ui:
            self.ui.voice = self.voice_label(lang)
        print(f"  [voice] {CODES[lang]} ({SPEAKERS[lang]})")

    async def apply_language(self, mode: str, source: str) -> bool:
        """Switch the reply language. mode: auto | hi | pa | en. source: voice | menu. Returns False if nothing changed."""
        if mode == self.lang_mode:
            return False
        self.lang_mode = mode
        if mode in CODES:
            self.lang = mode                         # fillers and the next voice follow at once
        try:
            await self.update_instructions(self._build_instructions())
        except Exception as e:
            logger.warning("language: instructions update failed: %s", e)
        print(f"  [language] reply language is now {mode} (from {source})")
        if self.ui:
            self.ui.set_language(mode, source)
        return True

    async def menu_language(self, mode: str) -> None:
        """Called when the language is picked in the dashboard: switch, then say so in the new language."""
        changed = await self.apply_language(mode, "menu")
        if not (changed and SPEAK_LANG_CONFIRM):
            return
        try:
            lang = mode if mode in CODES else self.lang
            self.apply_voice(lang)
            self.session.say(CONFIRM_TEXT[mode] if mode in CODES else CONFIRM_AUTO[lang])
        except Exception as e:
            logger.warning("language confirmation failed: %s", e)

    # --- echo guard: recognise our own voice coming back through the microphone ---
    def note_said(self, text: str) -> None:
        if text:
            self.said.append((time.monotonic(), text))
            self.last_said = time.monotonic()

    def is_echo(self, text: str) -> bool:
        """True if `text` (a transcript) looks like what the agent itself just said."""
        if ECHO_GUARD == "off":
            return False
        toks = _words(text)
        if not toks:
            return False
        now = time.monotonic()
        recent = "".join(c for t, c in self.said if now - t <= ECHO_WINDOW_S)
        agent_words = _words(recent)
        if not agent_words:
            return False
        mine, theirs = " ".join(toks), " ".join(agent_words)
        if len(toks) <= 2:        # "stop", "yes please": only an echo if we were talking a moment ago and said exactly that
            return (self.speaking or now - self.last_said <= 3.0) and mine in theirs
        known = set(agent_words)
        if len(toks) >= 4 and sum(1 for w in toks if w in known) / len(toks) >= 0.85:
            return True
        m = difflib.SequenceMatcher(None, mine, theirs, autojunk=False)
        matched = sum(b.size for b in m.get_matching_blocks() if b.size >= 4)
        return matched / len(mine) >= 0.8

    async def tts_node(self, text, model_settings):
        # Pick the voice from the reply itself, right before it is spoken.
        return Agent.default.tts_node(self, self._route_voice(text), model_settings)

    async def _route_voice(self, text):
        held, decided = [], False
        async for chunk in text:
            if decided:
                self.note_said(chunk)
                yield chunk
                continue
            held.append(chunk)
            joined = "".join(held)
            lang = script_lang(joined)
            # Gurmukhi / Devanagari decide at the first letter; Latin needs a few letters so that a leading
            # English word in a Hindi sentence does not pick the English voice.
            if lang in ("hi", "pa") or (lang == "en" and len(_LATIN.findall(joined)) >= ROUTE_MIN_LATIN):
                self.apply_voice(lang)
                decided = True
                held = []
                self.note_said(joined)
                yield joined
        if not decided and held:
            joined = "".join(held)
            self.apply_voice(script_lang(joined) or self.lang)
            self.note_said(joined)
            yield joined

    async def aclose(self) -> None:
        summary = self.meter.end_session()      # writes the session row to cost_log.csv + agent.log
        if summary:
            print(summary)
        for t in self._bg:
            t.cancel()
        await self.search.aclose()

    async def warm(self) -> None:
        """Open the search connections (and look up the default city) before the first question,
        then keep them alive in the background."""
        t0 = time.perf_counter()
        await self.search.warm(extra_urls=Weather.HOSTS)
        if DEFAULT_CITY:
            try:
                await self.weather.geocode(DEFAULT_CITY)
            except Exception as e:
                logger.warning("default city lookup failed: %r", e)
        print(f"  [search] providers={self.search.provider_names or 'none'} depth={self.search.tavily_depth} "
              f"timeout={SEARCH_TIMEOUT_S}s prefetch={PREFETCH_MODE}; "
              f"connections warmed in {round((time.perf_counter() - t0) * 1000)} ms")
        self._bg.append(asyncio.create_task(self.search.keepalive_loop(KEEPALIVE_S, Weather.HOSTS)))

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        text = new_message.text_content or ""
        if self.is_echo(text):                       # our own voice came back through the mic: do not answer it
            print(f"  [echo] ignored: {text[:70]!r}")
            if StopResponse is not None:
                raise StopResponse()
            return
        # Backup trigger: the transcript event normally starts the prefetch first, and this call is then a no-op.
        try:
            self.prefetch.on_final(text)
        except Exception as e:
            logger.warning("prefetch (turn end) failed: %r", e)

    # --- filler: pre-rendered audio if we have it, otherwise normal TTS ---
    def _pick_filler(self, step: int):
        clips = self.fillers.get(self.lang)
        if clips:
            text, frame = random.choice(clips)
            self.note_said(" " + text + " ")
            return self.session.say(text, audio=_one_frame(frame), add_to_chat_ctx=False)
        return random.choice(FILLER_TEXT[self.lang])

    async def _long_wait_filler(self) -> None:
        """Second filler ("still looking") if the search is still running after LONG_WAIT_S.
        The tool cancels this task the moment it has an answer."""
        await asyncio.sleep(LONG_WAIT_S)
        try:
            clips = self.fillers_long.get(self.lang)
            if clips:
                text, frame = random.choice(clips)
                self.note_said(" " + text + " ")
                self.session.say(text, audio=_one_frame(frame), add_to_chat_ctx=False)
            else:
                self.session.say(random.choice(LONG_FILLER_TEXT[self.lang]), add_to_chat_ctx=False)
        except Exception as e:
            logger.warning("long-wait filler failed: %r", e)

    # --- language tool: the model calls it when the user asks to change the language, in any wording ---
    @function_tool
    async def set_language(self, context: RunContext, language: str) -> str:
        """Change the language you speak to the user. Call this whenever the user asks you to talk in
        another language or to go back to following their language, in any wording or language.
        Examples: "talk to me in Punjabi", "change the language to Hindi", "speak English please",
        "हिंदी में बात करो", "ਪੰਜਾਬੀ ਵਿੱਚ ਬੋਲੋ", "use my language again".

        Args:
            language: One of: hindi, punjabi, english, auto. Use auto when the user wants you to answer in
                whatever language they speak.
        """
        mode = normalize_language(language)
        if mode is None:
            return "That language is not available. Tell the user you can speak Hindi, Punjabi or English."
        await self.apply_language(mode, "voice")
        if mode == "auto":
            return ("Done: you now answer in the language the user speaks. Confirm in one short sentence "
                    "in the language the user is speaking.")
        name, script = LANG_NAMES[mode]
        where = f" in {script} script" if script else ""
        return f"Done: language changed to {name}. Confirm it in one short sentence written in {name}{where}."

    # --- memory tools (only run when the model asks for them) ---
    @function_tool
    async def remember(self, context: RunContext, key: str, value: str) -> str:
        """Save a lasting fact about the user, such as their name, city or a preference.
        Use only when the user states such a fact or says "remember". Not for small talk.

        Args:
            key: Short label in English, e.g. name, city, favourite_food.
            value: The fact itself.
        """
        await asyncio.to_thread(self.mem.set_fact, key, value)
        self.facts_dirty = True            # shown in the instructions from the next turn on
        return "Saved. Confirm in a few words."

    @function_tool
    async def recall(self, context: RunContext, query: str) -> str:
        """Look up what was said in EARLIER conversations. Use only when the user asks about
        something from the past, like "what did I ask yesterday". Not for the current chat.

        Args:
            query: A few key words to search for.
        """
        t0 = time.perf_counter()
        out = await asyncio.to_thread(self.mem.search, query)
        print(f"  [memory] recall took {round((time.perf_counter() - t0) * 1000)} ms")
        return out

    # --- the tools ---
    @function_tool
    async def web_search(self, context: RunContext, query: str) -> str:
        """Search the internet for current information: news, prices, scores, recent
        events, or anything that changes. Do not use for weather, chit-chat or stable facts.

        Args:
            query: Short search query, a few words.
        """
        if not self.search.configured:
            return "Web search is not set up. Tell the user you cannot check the internet right now."
        t0 = time.perf_counter()
        if self.ui:
            self.ui.searching = "search"
        long_task = asyncio.create_task(self._long_wait_filler())
        text, source = None, ""
        try:
            # First filler starts once the session is idle for FILLER_DELAY_S, so a fast search
            # skips it. The second one (above) only fires if the search drags on.
            async with context.with_filler(self._pick_filler, delay=FILLER_DELAY_S):
                try:
                    # ONE deadline around the whole step (prefetch wait + possible retry).
                    text, source = await asyncio.wait_for(self._search_step(query, t0), timeout=self.search.timeout + 0.3)
                except asyncio.TimeoutError:
                    text, source = None, "timeout"
        finally:
            long_task.cancel()
            if self.ui:
                self.ui.searching = ""
        self.last_search_ms = round((time.perf_counter() - t0) * 1000)
        print(f"  [search] {source or 'none'}: waited {self.last_search_ms} ms ({'ok' if text else 'FAILED'})")
        return text or SEARCH_FAILED_MSG

    async def _search_step(self, query: str, t0: float):
        found, text, source = await self.prefetch.take()          # a search already running?
        # It failed fast and there is still time: one fresh try, but only inside the same overall deadline.
        if found and text is None and time.perf_counter() - t0 < self.search.timeout * 0.5:
            found = False
        if not found:
            text, source = await self.search.query(query)
        return text, source

    @function_tool
    async def get_weather(self, context: RunContext, city: str = "") -> str:
        """Get the current weather and today's forecast for a city. Use for any weather,
        temperature or rain question.

        Args:
            city: City name in English. Leave empty if the user did not name a city.
        """
        t0 = time.perf_counter()
        if self.ui:
            self.ui.searching = "weather"
        city = city.strip() or DEFAULT_CITY
        text, source = None, "weather"
        try:
            async with context.with_filler(self._pick_filler, delay=max(FILLER_DELAY_S, 0.8)):
                self.meter.add(weather_calls=1)                       # free API, counted for reference
                try:
                    text = await asyncio.wait_for(self._weather_step(city), timeout=self.search.timeout * 2 + 0.3)
                except asyncio.TimeoutError:
                    text = None
        finally:
            if self.ui:
                self.ui.searching = ""
        self.last_search_ms = round((time.perf_counter() - t0) * 1000)
        print(f"  [search] {source or 'none'}: waited {self.last_search_ms} ms ({'ok' if text else 'FAILED'})")
        return text or SEARCH_FAILED_MSG

    async def _weather_step(self, city: str):
        text = await self.weather.get(city)
        if text is None and self.search.configured:                   # direct API failed: general search
            text, _ = await self.search.query(f"weather in {city} today")
        return text


def make_components():
    """Build STT, LLM and TTS. Bad parameters raise here, so test this without a mic."""
    stt_extra = {}
    try:        # pass the STT model only if this version of the plugin takes one (it is also what gets priced)
        if "model" in inspect.signature(sarvam.STTRealtime).parameters:
            stt_extra["model"] = STT_MODEL
        else:
            logger.warning("sarvam.STTRealtime has no model parameter: the plugin default is used; "
                           "STT_MODEL=%s is only used for the price lookup", STT_MODEL)
    except (TypeError, ValueError):
        pass
    stt = sarvam.STTRealtime(
        **stt_extra,
        language=STT_LANGUAGE,
        stream_type=STT_STREAM_TYPE,
        mode=STT_TASK,
        endpointing="vad",              # Sarvam's server-side VAD sends speech start/end
        sample_rate=16000,
        vad_min_silence_ms=VAD_SILENCE_MS,
        vad_sot_threshold=VAD_THRESHOLD,         # higher = ignores quiet echo and room noise
        vad_min_speech_ms=VAD_MIN_SPEECH_MS,
    )
    llm_kwargs = {}
    if "gpt-oss" in LLM_MODEL:
        llm_kwargs["reasoning_effort"] = REASONING_EFFORT
    elif "qwen" in LLM_MODEL:
        llm_kwargs["reasoning_effort"] = "none"       # instruct mode: no thinking tokens
    llm = groq.LLM(model=LLM_MODEL, temperature=0.7, max_completion_tokens=MAX_REPLY_TOKENS,
                   **llm_kwargs)
    tts = sarvam.TTS(
        model=TTS_MODEL,
        target_language_code=CODES[INITIAL_LANG],
        speaker=SPEAKERS[INITIAL_LANG],
        speech_sample_rate=TTS_RATE,
        output_audio_codec="linear16",   # raw PCM: skips a decode step
        min_buffer_size=30,              # smaller = first audio sooner
        max_chunk_length=150,
    )
    return stt, llm, tts


def make_session(stt, llm, tts, vad=None) -> AgentSession:
    return AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,                          # None: Sarvam's server-side VAD drives turns. LOCAL_VAD=1 adds silero for barge-in.
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",         # trust Sarvam's end-of-speech event
            endpointing={"mode": "fixed", "min_delay": MIN_DELAY_S, "max_delay": 2.0},
            interruption={
                "enabled": True,
                "mode": "vad",            # with LOCAL_VAD=0 the transcript triggers barge-in (old behaviour)
                "min_duration": INTERRUPT_MIN_DURATION_S,
                "min_words": INTERRUPT_MIN_WORDS,   # 1 = a real transcribed word must arrive, so quiet echo / clicks do not interrupt
                "false_interruption_timeout": FALSE_INTERRUPT_S,
                "resume_false_interruption": True,
            },
            preemptive_generation={"enabled": PREEMPTIVE},
        ),
        max_tool_steps=3,                 # e.g. set_language + get_weather + the answer
        user_away_timeout=None,
        conn_options=SessionConnectOptions(
            stt_conn_options=APIConnectOptions(timeout=5.0),
            tts_conn_options=APIConnectOptions(max_retry=1, timeout=5.0),
            llm_conn_options=APIConnectOptions(max_retry=1, timeout=LLM_TIMEOUT_S),   # was 8 s: one stall = 10 s of silence
        ),
    )


async def load_fillers(agent: VoiceAgent) -> None:
    """Render the filler phrases once, in the background, so playing one costs no TTS time."""
    for lang in FILLER_TEXT:
        t = sarvam.TTS(model=TTS_MODEL, target_language_code=CODES[lang], speaker=SPEAKERS[lang],
                       speech_sample_rate=TTS_RATE, output_audio_codec="linear16", min_buffer_size=30)
        for table, target in ((FILLER_TEXT, agent.fillers), (LONG_FILLER_TEXT, agent.fillers_long)):
            clips = []
            for text in table[lang]:
                try:
                    clips.append((text, await t.synthesize(text).collect()))
                    agent.meter.prerender(len(text))
                except Exception as e:
                    logger.warning("filler %r (%s) not cached: %s", text, lang, e)
            target[lang] = clips
        try:
            await t.aclose()
        except Exception:
            pass
    logger.info("fillers cached: %s, long: %s", {k: len(v) for k, v in agent.fillers.items()},
                {k: len(v) for k, v in agent.fillers_long.items()})


async def trim_history(agent: VoiceAgent) -> None:
    """Keep only the last MAX_HISTORY_ITEMS chat items (tool results included).
    Runs after a reply finishes, so it never touches a request that is already in flight."""
    try:
        before = len(agent.chat_ctx.items)
        if before <= MAX_HISTORY_ITEMS + 2:             # +2 = slack for a tool call/result pair
            return
        ctx = agent.chat_ctx.copy()
        ctx.truncate(max_items=MAX_HISTORY_ITEMS)
        await agent.update_chat_ctx(ctx)
        print(f"  [history] trimmed {before} -> {len(ctx.items)} items")
    except Exception as e:
        logger.warning("history trim failed: %s", e)


async def refresh_facts(agent: VoiceAgent) -> None:
    """After a reply: if the user taught us something, put it in the instructions for next time."""
    today = datetime.now().date()
    if not agent.facts_dirty and today == agent._date:
        return
    agent.facts_dirty = False
    agent._date = today                # also rebuilds the instructions after midnight so "Today is ..." stays right
    try:
        await agent.update_instructions(agent._build_instructions())
        print(f"  [memory] facts now: {agent.mem.facts}")
    except Exception as e:
        logger.warning("facts refresh failed: %s", e)


async def flush_usage(agent: VoiceAgent, ui, reply_text: str) -> None:
    """Write the usage row a moment AFTER the reply is committed: LiveKit emits the TTS/LLM metrics of
    a reply around (sometimes just after) conversation_item_added, so flushing at once lost characters."""
    await asyncio.sleep(USAGE_FLUSH_DELAY_S)
    try:
        u = agent.meter.flush(reply_text)
        if ui:
            ui.last_usage = u
        else:
            print(f"  usage    : {agent.meter.line(u)}")
    except Exception as e:
        logger.warning("usage log failed: %r", e)


def attach_logging(session: AgentSession, agent: VoiceAgent, tts, ui=None) -> None:     # tts kept for callers
    warned_no_transcript: list = []
    gate = {"closed": False, "timer": None}

    def _mic(on: bool) -> None:
        if on and ui and ui.muted:                      # the user's own mute always wins
            return
        try:
            session.input.set_audio_enabled(on)
        except Exception as e:
            logger.warning("echo gate: could not switch the microphone: %r", e)

    def _gate_close() -> None:
        if gate["timer"]:
            gate["timer"].cancel()
            gate["timer"] = None
        if not gate["closed"]:
            gate["closed"] = True
            _mic(False)

    def _gate_open() -> None:
        gate["timer"] = None
        gate["closed"] = False
        _mic(True)

    def _gate_reopen_later() -> None:
        if gate["timer"] is None:
            gate["timer"] = asyncio.get_running_loop().call_later(ECHO_TAIL_S, _gate_open)

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        state = str(getattr(ev, "new_state", ""))
        agent.speaking = state == "speaking"
        if ui:
            ui.set_agent_state(state)
        if ECHO_GUARD == "gate":
            if state == "speaking":
                _gate_close()
            elif gate["closed"]:
                _gate_reopen_later()

    @session.on("user_state_changed")
    def _on_user_state(ev):
        if ui:
            ui.set_user_state(str(getattr(ev, "new_state", "")))

    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        # Prefetch: partials may start a speculative search, the final transcript confirms or replaces it.
        try:
            text = getattr(ev, "transcript", None)
            if text is None:
                if not warned_no_transcript:
                    warned_no_transcript.append(1)
                    logger.warning("user_input_transcribed event has no .transcript field; prefetch is off")
            elif agent.is_echo(text):
                pass                                    # our own voice: no speculative search, nothing on screen
            elif ev.is_final:
                agent.prefetch.on_final(text)
            else:
                agent.prefetch.on_partial(text)
                if ui:
                    ui.set_partial(text)
        except Exception as e:
            logger.warning("prefetch failed: %r", e)
        # With language=auto, the final transcript carries the language Sarvam heard. It only steers the
        # "let me check" fillers now. The reply voice is chosen from the reply text (VoiceAgent.tts_node),
        # and a pinned reply language is never overridden by what you happen to speak.
        if not ev.is_final or not ev.language or agent.lang_mode != "auto":
            return
        lang = str(ev.language).lower()[:2]
        if lang in CODES and lang != agent.lang:
            agent.lang = lang
            print(f"  [language] heard {CODES[lang]}")

    @session.on("metrics_collected")
    def _on_metrics(ev):
        try:
            agent.meter.on_metrics(ev.metrics)
        except Exception as e:
            logger.warning("usage metrics failed: %r", e)

    last_user: dict = {}

    @session.on("conversation_item_added")
    def _on_item(ev):
        item = ev.item
        role = getattr(item, "role", None)
        metrics = getattr(item, "metrics", None) or {}
        if role == "user":
            if agent.is_echo(item.text_content or ""):
                return                                  # echo of the agent's own voice: not a real turn
            last_user.clear()
            last_user.update(metrics)
            if ui:
                ui.user(item.text_content or "")
            else:
                print(f"\n  you said : {item.text_content}")
            if item.text_content:
                spawn(asyncio.to_thread(agent.mem.add_turn, "user", item.text_content), "memory")
            return
        if role != "assistant":
            return
        e2e = metrics.get("e2e_latency")
        row = [
            "sarvam_realtime", LLM_MODEL,
            _ms(last_user.get("transcription_delay")), _ms(last_user.get("end_of_turn_delay")),
            _ms(metrics.get("llm_node_ttft")), _ms(metrics.get("tts_node_ttfb")),
            _ms(e2e),
            # e2e starts after Sarvam's silence window; the old CSV's total added END_SILENCE_MS
            _ms(e2e + VAD_SILENCE_MS / 1000) if isinstance(e2e, (int, float)) else "",
            agent.last_search_ms if agent.last_search_ms is not None else "",
        ]
        if ui:
            ui.set_latency(stt=row[2], eot=row[3], llm=row[4], tts=row[5], e2e=row[6] if row[6] != "" else None,
                           search=row[8] or None)
            ui.agent(item.text_content or "", row[6] if row[6] != "" else None)
        else:
            print(f"  reply    : {item.text_content}")
            print(f"  e2e {row[6]} ms  (stt {row[2]}, end-of-turn {row[3]}, llm first token {row[4]}, "
                  f"tts first byte {row[5]}, search {row[8] or '-'})")
        logger.info("latency | e2e %s ms | stt %s | end-of-turn %s | llm first token %s | tts first byte %s | search %s",
                    row[6], row[2], row[3], row[4], row[5], row[8] or "-")
        agent.last_search_ms = None
        spawn(flush_usage(agent, ui, item.text_content or ""), "usage")
        new = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_HEADER)
            w.writerow(row)
        if item.text_content:
            spawn(asyncio.to_thread(agent.mem.add_turn, "assistant", item.text_content), "memory")
        spawn(trim_history(agent), "trim")
        spawn(refresh_facts(agent), "facts")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    setup_logging()                       # again: LiveKit may have reset logging when the job started
    stt, llm, tts = make_components()
    try:
        tts.prewarm()                     # open the TTS socket now, not on the first reply
    except Exception as e:
        logger.warning("tts prewarm skipped: %s", e)
    vad = None
    if LOCAL_VAD:
        try:
            vad = inference.VAD()         # same VAD as the starter template; runs locally
        except Exception as e:
            logger.warning("local VAD unavailable (%r): voice interruption will be slow", e)
    session = make_session(stt, llm, tts, vad)
    agent = VoiceAgent()
    agent.tts_engine = tts
    loop = asyncio.get_running_loop()
    ui = None
    if UI_ENABLED and sys.stdout.isatty():
        try:
            from cli_ui import VoiceUI          # imported here so a missing 'rich' never stops the agent
            ui = VoiceUI(stt="sarvam realtime", llm=LLM_MODEL, tts=TTS_MODEL,
                         voice=agent.voice_label(INITIAL_LANG), meter=agent.meter,
                         on_mute=lambda muted: session.input.set_audio_enabled(not muted),
                         languages=UI_LANGUAGES, language=agent.lang_mode, guard=ECHO_GUARD,
                         on_interrupt=lambda: loop.call_soon_threadsafe(lambda: session.interrupt()),
                         # called from the keyboard thread, so hand the work to the event loop
                         on_language=lambda code: loop.call_soon_threadsafe(
                             lambda: spawn(agent.menu_language(code), "language")))
            agent.ui = ui
        except ImportError as e:
            print(f"  [ui] dashboard off ({e}). pip install rich, or set UI=0")
    attach_logging(session, agent, tts, ui)
    ctx.add_shutdown_callback(agent.aclose)
    agent._bg.append(asyncio.create_task(agent.warm()))   # open search connections while the session starts
    agent._bg.append(asyncio.create_task(load_fillers(agent)))   # start rendering fillers now, not after the session is up

    await session.start(agent=agent, room=ctx.room)
    if ui:
        ui.start()

        async def _stop_ui() -> None:
            ui.stop()
        ctx.add_shutdown_callback(_stop_ui)


if __name__ == "__main__":
    if sys.argv[1:2] in (["console"], ["dev"], ["start"], ["connect"]):
        from price_check import ensure_prices
        ensure_prices()                  # asks for the price of any model in settings.env that has none yet
    cli.run_app(server)

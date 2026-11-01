#!/usr/bin/env python3
"""
LiveKit version of the voice latency demo: same providers, but LiveKit Agents runs the loop
(turn-taking, barge-in, preemptive generation, streaming) instead of hand-written threads.

Stack (all open source on the LiveKit side, Apache 2.0):
    STT  Sarvam realtime (saaras:v3-realtime). Sarvam's own server-side VAD decides end of turn.
    LLM  Groq, same model as before (GROQ_LLM_MODEL), reasoning off.
    TTS  Sarvam Bulbul v4 Flash. The voice follows the language Sarvam detects in your speech.
    Tool web_search (Tavily). A pre-rendered "let me check" clip plays while the search runs.

Install (Python 3.10+):
    pip install -r requirements-livekit.txt
    (Linux also needs: sudo apt install libportaudio2)

Run, talking through your own mic and speakers. Use headphones: with open speakers the agent
hears itself and interrupts itself.
    python livekit_agent.py console

Keys come from the same .env as the old demo (GROQ_API_KEY, SARVAM_API_KEY) plus:
    TAVILY_API_KEY=tvly-...        # web search; without it the agent says search is unavailable
Optional tuning (defaults shown):
    SARVAM_VAD_SILENCE_MS=300      # Sarvam's end-of-turn silence window
    ENDPOINT_MIN_DELAY_S=0.25      # extra hold after that; the two add up
    FILLER_DELAY_S=0.3             # skip the filler if the search comes back faster than this
    START_LANG=hi                  # voice used until the first turn is detected: hi, pa or en
Existing variables still honoured: GROQ_LLM_MODEL, MAX_REPLY_TOKENS, TTS_MODEL, SPEAKER_HI,
SPEAKER_EN, SPEAKER_PA, SARVAM_RT_LANGUAGE, SARVAM_RT_STREAM_TYPE, SARVAM_RT_TASK.

Timings go to latency_log_livekit.csv (new file, different columns from latency_log.csv).
"""
import asyncio
import csv
import logging
import os
import random
import re
import sqlite3
import threading
import time
from datetime import datetime

import httpx
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
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import groq, sarvam

load_dotenv()
load_dotenv(".env.local")

logger = logging.getLogger("voice-agent")


def env(key, default=""):
    return os.getenv(key, default).strip()


# ---------- config (same names as env.example where they exist) ----------
LLM_MODEL = env("GROQ_LLM_MODEL", "qwen/qwen3.8-27b")
MAX_REPLY_TOKENS = int(env("MAX_REPLY_TOKENS", "300"))
TTS_MODEL = env("TTS_MODEL", "bulbul:v4-flash")
STT_LANGUAGE = env("SARVAM_RT_LANGUAGE", "auto")          # auto, hi-IN, pa-IN, en-IN
STT_STREAM_TYPE = env("SARVAM_RT_STREAM_TYPE", "fast")    # fast | balanced | simulated
STT_TASK = env("SARVAM_RT_TASK", "transcribe")            # transcribe | codemix | translate ...
VAD_SILENCE_MS = int(env("SARVAM_VAD_SILENCE_MS", "300"))
MIN_DELAY_S = float(env("ENDPOINT_MIN_DELAY_S", "0.25"))
FILLER_DELAY_S = float(env("FILLER_DELAY_S", "0.3"))
MEMORY_DB = env("MEMORY_DB", "memory.db")                 # everything stays on this machine
MAX_HISTORY_ITEMS = int(env("MAX_HISTORY_ITEMS", "8"))   # chat items kept between turns (keeps Groq TPM low)
TAVILY_KEY = env("TAVILY_API_KEY")
TTS_RATE = 24000

CODES = {"hi": "hi-IN", "pa": "pa-IN", "en": "en-IN"}
SPEAKERS = {
    "hi": env("SPEAKER_HI", "aparna_hi_customer"),
    "pa": env("SPEAKER_PA", "anand_pa_customer"),
    "en": env("SPEAKER_EN", "simran_en_customer"),
}
_start = env("START_LANG") or (STT_LANGUAGE[:2] if STT_LANGUAGE[:2] in CODES else "hi")
START_LANG = _start if _start in CODES else "hi"

FILLER_TEXT = {
    "en": ["Okay, let me check.", "One sec, checking that."],
    "hi": ["ठीक है, मैं देखता हूँ।", "एक सेकंड, चेक करता हूँ।"],
    "pa": ["ਠੀਕ ਹੈ, ਮੈਂ ਵੇਖਦਾ ਹਾਂ।", "ਇੱਕ ਸਕਿੰਟ, ਚੈੱਕ ਕਰਦਾ ਹਾਂ।"],
}

class Memory:
    """Local memory. Hot path rule: nothing here touches a normal turn.
    - facts: small, loaded once at startup, shown to the model in its instructions
    - turns: full log of the conversation, searched only when the recall tool is called
    Writes happen after the reply, in a background thread."""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
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
        key, value = key.strip()[:40], value.strip()[:120]
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

    def search(self, query: str, limit: int = 3) -> str:
        words = re.findall(r"\w+", query, flags=re.UNICODE)
        if not words:
            return "Nothing found."
        with self._lock:
            if self.fts:
                match = " OR ".join(f'"{w}"' for w in words)
                rows = self.db.execute(
                    "SELECT t.ts, t.role, t.text FROM turns_fts f JOIN turns t ON t.id = f.rowid "
                    "WHERE turns_fts MATCH ? ORDER BY bm25(turns_fts) LIMIT ?", (match, limit)).fetchall()
            else:
                like = f"%{words[0]}%"
                rows = self.db.execute("SELECT ts, role, text FROM turns WHERE text LIKE ? "
                                       "ORDER BY ts DESC LIMIT ?", (like, limit)).fetchall()
        if not rows:
            return "Nothing found about that."
        return "\n".join(f"{datetime.fromtimestamp(ts):%d %b %H:%M} {role}: {text[:160]}"
                         for ts, role, text in rows)


INSTRUCTIONS = (
    "You are a voice assistant. Your reply is spoken aloud, so use plain text only: no markdown, "
    "lists, emojis or links. "
    "Reply in the language the user spoke: Punjabi in Gurmukhi script, Hindi in Devanagari script, "
    "English in English. If the user speaks Punjabi or Hindi, you speak in that language too, "
    "keeping English words in Latin script. "
    "Maximum 3-4 short sentences, under 240 characters. "
    "Only go into detail if the user asks you to continue or asks a detailed question. "
    "For anything current (news, weather, prices, scores, recent events) call the web_search tool "
    "instead of guessing. Call the tool right away without saying anything first. "
    "After the results, answer in one or two short sentences and never read out links. "
    "Use remember only for lasting personal facts the user states, and recall only when the user "
    "asks about an earlier conversation. "
    "Today is {today}."
)

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
        super().__init__(instructions=self._build_instructions())
        self.lang = START_LANG
        self.fillers: dict[str, list[tuple[str, rtc.AudioFrame]]] = {}   # filled in the background
        self.last_search_ms = None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0))

    def _build_instructions(self) -> str:
        return INSTRUCTIONS.format(today=datetime.now().strftime("%A, %d %B %Y")) + self.mem.facts_block()

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- filler: pre-rendered audio if we have it, otherwise normal TTS ---
    def _pick_filler(self, step: int):
        clips = self.fillers.get(self.lang)
        if clips:
            text, frame = random.choice(clips)
            return self.session.say(text, audio=_one_frame(frame), add_to_chat_ctx=False)
        return random.choice(FILLER_TEXT[self.lang])

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

    # --- the tool ---
    @function_tool
    async def web_search(self, context: RunContext, query: str) -> str:
        """Search the internet for current information: news, weather, prices, scores, recent
        events, or anything that changes. Do not use for chit-chat or stable facts.

        Args:
            query: Short search query, a few words.
        """
        t0 = time.perf_counter()
        # The filler starts once the session is idle for FILLER_DELAY_S, so a fast search
        # skips it, and a slow one is covered. The search runs underneath either way.
        async with context.with_filler(self._pick_filler, delay=FILLER_DELAY_S):
            result = await self._search(query)
        self.last_search_ms = round((time.perf_counter() - t0) * 1000)
        return result

    async def _search(self, query: str) -> str:
        if not TAVILY_KEY:
            return "Web search is not set up. Tell the user you cannot check the internet right now."
        try:
            r = await self._http.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {TAVILY_KEY}"},
                json={"query": query, "max_results": 3, "search_depth": "basic",
                      "include_answer": True},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("search failed: %s", e)
            return "The search failed. Tell the user you could not check right now."
        parts = [data["answer"]] if data.get("answer") else []
        parts += [x.get("content", "")[:300] for x in data.get("results", [])[:3]]
        return "\n".join(parts) or "No results found."


def make_components():
    """Build STT, LLM and TTS. Bad parameters raise here, so test this without a mic."""
    stt = sarvam.STTRealtime(
        language=STT_LANGUAGE,
        stream_type=STT_STREAM_TYPE,
        mode=STT_TASK,
        endpointing="vad",              # Sarvam's server-side VAD sends speech start/end
        sample_rate=16000,
        vad_min_silence_ms=VAD_SILENCE_MS,
    )
    llm_kwargs = {}
    if "gpt-oss" in LLM_MODEL:
        llm_kwargs["reasoning_effort"] = "low"
    elif "qwen" in LLM_MODEL:
        llm_kwargs["reasoning_effort"] = "none"       # instruct mode: no thinking tokens
    llm = groq.LLM(model=LLM_MODEL, temperature=0.7, max_completion_tokens=MAX_REPLY_TOKENS,
                   **llm_kwargs)
    tts = sarvam.TTS(
        model=TTS_MODEL,
        target_language_code=CODES[START_LANG],
        speaker=SPEAKERS[START_LANG],
        speech_sample_rate=TTS_RATE,
        output_audio_codec="linear16",   # raw PCM: skips a decode step
        min_buffer_size=30,              # smaller = first audio sooner
        max_chunk_length=150,
    )
    return stt, llm, tts


def make_session(stt, llm, tts) -> AgentSession:
    return AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=None,                         # Sarvam's server-side VAD drives turns instead
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",         # trust Sarvam's end-of-speech event
            endpointing={"mode": "fixed", "min_delay": MIN_DELAY_S, "max_delay": 2.0},
            interruption={
                "enabled": True,
                "mode": "vad",            # the adaptive detector needs a local VAD
                "min_words": 1,           # real words only, so coughs and clicks don't interrupt
                "false_interruption_timeout": 1.3,
                "resume_false_interruption": True,
            },
            preemptive_generation={"enabled": True},
        ),
        max_tool_steps=2,
        user_away_timeout=None,
        conn_options=SessionConnectOptions(
            stt_conn_options=APIConnectOptions(timeout=5.0),
            tts_conn_options=APIConnectOptions(max_retry=1, timeout=5.0),
            llm_conn_options=APIConnectOptions(max_retry=1, timeout=8.0),
        ),
    )


async def load_fillers(agent: VoiceAgent) -> None:
    """Render the filler phrases once, in the background, so playing one costs no TTS time."""
    for lang, texts in FILLER_TEXT.items():
        t = sarvam.TTS(model=TTS_MODEL, target_language_code=CODES[lang], speaker=SPEAKERS[lang],
                       speech_sample_rate=TTS_RATE, output_audio_codec="linear16", min_buffer_size=30)
        clips = []
        for text in texts:
            try:
                clips.append((text, await t.synthesize(text).collect()))
            except Exception as e:
                logger.warning("filler %r (%s) not cached: %s", text, lang, e)
        agent.fillers[lang] = clips
        try:
            await t.aclose()
        except Exception:
            pass
    logger.info("fillers cached: %s", {k: len(v) for k, v in agent.fillers.items()})


async def trim_history(agent: VoiceAgent) -> None:
    """Keep only the last MAX_HISTORY_ITEMS chat items (tool results included).
    Runs after a reply finishes, so it never touches a request that is already in flight."""
    try:
        before = len(agent.chat_ctx.items)
        if before <= MAX_HISTORY_ITEMS + 2:             # +2 = the instruction items
            return
        ctx = agent.chat_ctx.copy()
        ctx.truncate(max_items=MAX_HISTORY_ITEMS)
        await agent.update_chat_ctx(ctx)
        print(f"  [history] trimmed {before} -> {len(ctx.items)} items")
    except Exception as e:
        logger.warning("history trim failed: %s", e)


async def refresh_facts(agent: VoiceAgent) -> None:
    """After a reply: if the user taught us something, put it in the instructions for next time."""
    if not agent.facts_dirty:
        return
    agent.facts_dirty = False
    try:
        await agent.update_instructions(agent._build_instructions())
        print(f"  [memory] facts now: {agent.mem.facts}")
    except Exception as e:
        logger.warning("facts refresh failed: %s", e)


def attach_logging(session: AgentSession, agent: VoiceAgent, tts) -> None:
    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        # With language=auto, the final transcript carries the detected language. Switch the
        # voice before the reply is spoken, so the answer comes out in the language you used.
        if not ev.is_final or not ev.language:
            return
        lang = str(ev.language).lower()[:2]
        if lang in CODES and lang != agent.lang:
            try:
                tts.update_options(target_language_code=CODES[lang], speaker=SPEAKERS[lang])
                agent.lang = lang
                print(f"  [voice] switched to {CODES[lang]} ({SPEAKERS[lang]})")
            except Exception as e:
                logger.warning("voice switch to %s failed: %s", lang, e)

    last_user: dict = {}

    @session.on("conversation_item_added")
    def _on_item(ev):
        item = ev.item
        role = getattr(item, "role", None)
        metrics = getattr(item, "metrics", None) or {}
        if role == "user":
            last_user.clear()
            last_user.update(metrics)
            print(f"\n  you said : {item.text_content}")
            if item.text_content:
                asyncio.ensure_future(asyncio.to_thread(agent.mem.add_turn, "user", item.text_content))
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
        print(f"  reply    : {item.text_content}")
        print(f"  e2e {row[6]} ms  (stt {row[2]}, end-of-turn {row[3]}, llm first token {row[4]}, "
              f"tts first byte {row[5]}, search {row[8] or '-'})")
        agent.last_search_ms = None
        new = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_HEADER)
            w.writerow(row)
        if item.text_content:
            asyncio.ensure_future(asyncio.to_thread(agent.mem.add_turn, "assistant", item.text_content))
        asyncio.ensure_future(trim_history(agent))
        asyncio.ensure_future(refresh_facts(agent))


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    stt, llm, tts = make_components()
    try:
        tts.prewarm()                     # open the TTS socket now, not on the first reply
    except Exception as e:
        logger.warning("tts prewarm skipped: %s", e)
    session = make_session(stt, llm, tts)
    agent = VoiceAgent()
    attach_logging(session, agent, tts)
    ctx.add_shutdown_callback(agent.aclose)

    await session.start(agent=agent, room=ctx.room)
    asyncio.create_task(load_fillers(agent))


if __name__ == "__main__":
    cli.run_app(server)

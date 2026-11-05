"""
Full-screen terminal dashboard for the voice agent. No LiveKit imports. Needs:  pip install rich

It takes over the terminal (alternate screen), so nothing is left in your scrollback when you quit.

    +-- header: status pill, mic, reply language, voice, models -----------------+
    | conversation (newest at the bottom)      | latency, per-stage bars          |
    |                                          | COST: split by LLM/TTS/STT/search|
    +-- log: everything the agent print()s or logs, captured so it cannot break the screen -+
    |  M mute   L language   1-4 quick language   J/K scroll   Q quit            |

Keys (all act on the FIRST press):
    M / Space      mute / unmute the microphone
    L / Tab        open the language dropdown (Up/Down or J/K to move, Enter to pick, Esc to close)
    1 2 3 4        pick a reply language directly (Auto, Hindi, Punjabi, English)
    (header shows the running session cost at all times)
    K / Up         scroll the conversation up      J / Down   scroll down
    PgUp / PgDn    scroll faster                   B / End    jump back to the latest message
    Q              quit (Ctrl+C also works)

Env:    UI=0         turn the dashboard off (plain printing)
        UI_KEYS=0    do not read the keyboard
        UI_FPS=8     max redraws per second (it only redraws when something changed)

Why the old keys needed several presses
---------------------------------------
LiveKit's own `console` mode runs a keyboard reader (livekit.agents.cli.readchar) on the same stdin.
Each of its reads flushes pending input and puts the terminal back in line mode, so it ate or undid
this dashboard's keypresses. start() now takes sole ownership of the keyboard: file descriptor 0 is
pointed at a dummy pseudo-terminal that nobody types into (LiveKit's reader blocks there harmlessly)
and this module reads the real terminal through a private copy. Its [Ctrl+B] text/audio toggle and
its own [Q] no longer work while the dashboard is on; this dashboard's keys replace them.

How output capture works: stdout and stderr (file descriptors 1 and 2) are pointed at a pipe, and a
thread turns each line into an entry of the log panel. The screen itself is drawn on a private copy
of the real terminal. On exit everything is restored and the last log lines are printed.
"""
from __future__ import annotations

import atexit
import collections
import math
import os
import signal
import sys
import threading
import time
from typing import Callable, Optional

from rich import box
from rich.console import Console, ConsoleDimensions, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

# ------------------------------------------------------------------ palette
C_TEXT = "#c0caf5"
C_SOFT = "#a9b1d6"
C_DIM = "#565f89"
C_LINE = "#3b4261"
C_INK = "#16161e"
C_BLUE = "#7aa2f7"
C_CYAN = "#7dcfff"
C_TEAL = "#73daca"
C_GREEN = "#9ece6a"
C_YELLOW = "#e0af68"
C_ORANGE = "#ff9e64"
C_RED = "#f7768e"
C_PURPLE = "#bb9af7"

BARS = "▁▂▃▄▅▆▇█"
ACTIVE = ("speaking", "hearing", "thinking", "searching")

# status -> (label, pill style, accent colour used for the header border)
BADGES = {
    "muted":     ("MUTED",       f"bold {C_TEXT} on {C_RED}",   C_RED),
    "searching": ("SEARCHING",   f"bold {C_INK} on {C_PURPLE}", C_PURPLE),
    "speaking":  ("SPEAKING",    f"bold {C_INK} on {C_CYAN}",   C_CYAN),
    "thinking":  ("THINKING",    f"bold {C_INK} on {C_YELLOW}", C_YELLOW),
    "hearing":   ("HEARING YOU", f"bold {C_INK} on {C_GREEN}",  C_GREEN),
    "listening": ("LISTENING",   f"bold {C_INK} on {C_TEAL}",   C_TEAL),
    "asleep":    ("ASLEEP",      f"bold {C_TEXT} on {C_DIM}",   C_DIM),
    "starting":  ("STARTING",    f"bold {C_TEXT} on {C_LINE}",  C_LINE),
    "stopping":  ("STOPPING",    f"bold {C_INK} on {C_ORANGE}", C_ORANGE),
}

# (code, name, native name / hint) for the language dropdown; the agent can pass its own list
DEFAULT_LANGS = [
    ("auto", "Auto", "follow my language"),
    ("hi", "Hindi", "हिन्दी"),
    ("pa", "Punjabi", "ਪੰਜਾਬੀ"),
    ("en", "English", "English"),
]

TOAST_COLORS = {"lang": C_PURPLE, "info": C_BLUE, "error": C_RED, "ok": C_GREEN}
GUTTER = 15            # width of "  14:02  YOU    " in front of every chat line


# ------------------------------------------------------------------ small helpers
def _spark(values: list[float], top: float = 4000.0) -> str:
    return "".join(BARS[min(len(BARS) - 1, int(max(v, 0) / top * len(BARS)))] for v in values)


def _lat_style(ms: Optional[float]) -> str:
    if ms is None:
        return C_DIM
    return C_GREEN if ms < 1500 else C_YELLOW if ms < 2500 else C_RED


def _stage_style(ms: float) -> str:
    return C_GREEN if ms < 500 else C_YELLOW if ms < 1200 else C_RED


_STOPS = ((0x9E, 0xCE, 0x6A), (0xE0, 0xAF, 0x68), (0xF7, 0x76, 0x8E))     # green -> yellow -> red


def _grad(t: float) -> str:
    """Colour at position t (0..1) along a green -> yellow -> red ramp (the btop look)."""
    t = min(max(t, 0.0), 1.0) * 2
    i = min(int(t), 1)
    f = t - i
    a, b = _STOPS[i], _STOPS[i + 1]
    return "#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def _bar(ms: float, top: float = 3000.0, width: int = 16) -> Text:
    """Latency bar: every filled cell has its own colour, so a long bar visibly runs into the red."""
    n = min(width, max(0, int(ms / top * width + 0.5)))
    t = Text()
    for i in range(n):
        t.append("━", style=_grad(i / max(width - 1, 1)))
    t.append("─" * (width - n), style=C_LINE)
    return t


_BL = (0x40, 0x04, 0x02, 0x01)       # braille dots of the left column, bottom -> top
_BR = (0x80, 0x20, 0x10, 0x08)       # right column


def _braille(values: list, top: float = 0.0, width: int = 24) -> str:
    """Tiny graph with 2 samples per character and 4 height levels each (twice the detail of ▁▂▃)."""
    vals = [max(float(v), 0.0) for v in values][-width * 2:]
    if not vals:
        return ""
    if len(vals) % 2:
        vals.insert(0, 0.0)
    top = top or max(vals) or 1.0
    out = []
    for i in range(0, len(vals), 2):
        mask = 0
        for col, v in ((_BL, vals[i]), (_BR, vals[i + 1])):
            h = min(4, round(v / top * 4))
            if v > 0:
                h = max(h, 1)
            for bit in col[:h]:
                mask |= bit
        out.append(chr(0x2800 + mask))
    return "".join(out)


_EIGHTHS = " ▏▎▍▌▋▊▉█"


def _cbar(share: float, width: int = 10) -> str:
    """Horizontal bar with 1/8-cell resolution."""
    eighths = round(min(max(share, 0.0), 1.0) * width * 8)
    full, rem = divmod(eighths, 8)
    bar = "█" * full + (_EIGHTHS[rem] if rem else "")
    return bar + "░" * (width - len(bar))


COST_PARTS = ("tts", "stt", "llm", "search")
COST_COLORS = {"tts": C_CYAN, "stt": C_GREEN, "llm": C_PURPLE, "search": C_ORANGE}


def _parts(c) -> dict:
    """Whatever the meter gives for a cost (dict or bare number) -> {tts, stt, llm, search, total}."""
    if isinstance(c, dict):
        d = {k: float(c.get(k) or 0.0) for k in COST_PARTS}
        d["total"] = float(c.get("total") or sum(d.values()))
        return d
    try:
        total = float(c)
    except (TypeError, ValueError):
        total = 0.0
    return {**{k: 0.0 for k in COST_PARTS}, "total": total}


def _fmt(n: float) -> str:
    return f"{int(n):,}"


def _chip(key: str, label: str) -> Text:
    return Text.assemble((f" {key} ", f"bold {C_INK} on {C_BLUE}"), (f" {label}   ", C_SOFT))


def _cut(line: list, x: int) -> tuple[list, list]:
    """Split one rendered line (a list of Segments) at terminal column x."""
    left, right, pos = [], [], 0
    for seg in line:
        if seg.control:
            (left if pos < x else right).append(seg)
            continue
        n = seg.cell_length
        if pos + n <= x:
            left.append(seg)
        elif pos >= x:
            right.append(seg)
        else:
            a, b = seg.split_cells(x - pos)
            left.append(a)
            right.append(b)
        pos += n
    return left, right


def decode_keys(data: bytes) -> list[str]:
    """Turn raw terminal bytes into key names. Several keys can arrive in one read."""
    table = {b"[A": "up", b"[B": "down", b"OA": "up", b"OB": "down", b"[5~": "pgup", b"[6~": "pgdn",
             b"[F": "end", b"[4~": "end", b"OF": "end", b"[8~": "end"}
    out: list[str] = []
    i, n = 0, len(data)
    while i < n:
        b = data[i]
        if b == 0x1B:
            rest = data[i + 1:]
            if rest[:1] in (b"[", b"O"):
                j = 1
                while j < len(rest) and not (0x40 <= rest[j] <= 0x7E):
                    j += 1
                seq = rest[:j + 1]
                i += 1 + len(seq)
                name = table.get(seq)
                if name:
                    out.append(name)
            else:
                out.append("esc")
                i += 1
        elif b in (0x0D, 0x0A):
            out.append("enter")
            i += 1
        elif b == 0x09:
            out.append("tab")
            i += 1
        elif b < 0x20 or b == 0x7F:
            i += 1                                                  # other control keys: ignore
        else:
            ln = 1 if b < 0x80 else 2 if b >> 5 == 0b110 else 3 if b >> 4 == 0b1110 else 4
            out.append(data[i:i + ln].decode("utf-8", "ignore").lower())
            i += ln
    return [k for k in out if k]


class _TermConsole(Console):
    """Console whose size comes from the REAL terminal fd, because fds 1 and 2 point at our log pipe."""
    term_fd: Optional[int] = None

    @property
    def size(self) -> ConsoleDimensions:            # type: ignore[override]
        if self.term_fd is not None:
            try:
                w, h = os.get_terminal_size(self.term_fd)
                return ConsoleDimensions(w, h)
            except OSError:
                pass
        return super().size


class _Screen:
    """What Live draws: the dashboard, with the language dropdown floated on top when it is open."""

    def __init__(self, ui: "VoiceUI") -> None:
        self.ui = ui

    def __rich_console__(self, console, options):
        ui = self.ui
        base = ui.render()
        if not ui.menu_open:
            yield base
            return
        lines = None
        try:
            w = options.max_width
            h = options.height or console.size.height
            lines = console.render_lines(base, options.update_dimensions(w, h), pad=True)
            mw, mh = ui.menu_size()
            mlines = console.render_lines(ui.menu_panel(mw), options.update_dimensions(mw, mh), pad=True)
            x, y = max(w - mw - 2, 0), 3                     # under the language pill, right side
            for i, ml in enumerate(mlines):
                r = y + i
                if r >= len(lines):
                    break
                left, rest = _cut(lines[r], x)
                _, right = _cut(rest, mw)
                lines[r] = left + ml + right
        except Exception:
            lines = None
        if lines is None:
            yield base
            return
        for i, line in enumerate(lines):
            yield from line
            if i < len(lines) - 1:
                yield Segment.line()


class VoiceUI:
    def __init__(self, *, stt: str, llm: str, tts: str, voice: str, meter=None,
                 on_mute: Optional[Callable[[bool], None]] = None,
                 on_language: Optional[Callable[[str], None]] = None,
                 on_interrupt: Optional[Callable[[], None]] = None, guard: str = "",
                 on_wake: Optional[Callable[[], None]] = None, wake_name: str = "",
                 languages: Optional[list] = None, language: str = "auto") -> None:
        self.console: Console = Console()
        self.stt, self.llm, self.tts, self.voice = stt, llm, tts, voice
        self.meter = meter
        self.on_mute = on_mute
        self.on_language = on_language
        self.on_interrupt = on_interrupt
        self.guard = guard
        self.langs = list(languages or DEFAULT_LANGS)
        self.lang_mode = language
        self.menu_open = False
        self.menu_idx = 0
        self.muted = False
        self.on_wake = on_wake             # key W: wake / sleep by hand (only when a wake word is configured)
        self.wake_name = wake_name         # "" = no wake word, the agent always listens
        self.asleep = False                # True while the wake gate is closed (nothing goes to STT)
        self.quitting = False
        self.agent_state = "starting"      # LiveKit: initializing / idle / listening / thinking / speaking
        self.user_speaking = False
        self.searching = ""                # "" or a label such as "search" / "weather"
        self.lat: dict = {}                # stt, eot, llm, tts, e2e, search (ms or None)
        self.e2e_history: list[float] = []
        self.cost_history: collections.deque = collections.deque(maxlen=48)   # cost of each finished reply
        self.replies_cost = 0.0            # sum of all reply costs: session total minus this = overhead
        self._last_usage: Optional[dict] = None
        self.partial = ""                  # what you are saying right now (live transcript)
        self.scroll = 0                    # lines scrolled up in the conversation (0 = following the latest)
        self.turns = 0
        self.started = time.monotonic()
        self.toast: Optional[tuple[str, str, float]] = None      # (text, kind, show until)
        self.toast_id = 0
        self.chat: collections.deque = collections.deque(maxlen=300)   # (role, text, e2e_ms, "HH:MM")
        self.log: collections.deque = collections.deque(maxlen=500)    # Text lines
        self._screen = _Screen(self)
        self._live: Optional[Live] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._stopped = False
        self._keys: Optional[threading.Thread] = None
        self._tick: Optional[threading.Thread] = None
        self._reader: Optional[threading.Thread] = None
        self._sig = None
        self._saved: Optional[tuple[int, int]] = None
        self._pipe_r: Optional[int] = None
        self._term_fd: Optional[int] = None
        self._tty_fd: Optional[int] = None         # private handle to the real keyboard terminal
        self._tty_attrs = None
        self._pty_master: Optional[int] = None

    # ---------------- state -> badge ----------------
    @property
    def status(self) -> str:
        if self.quitting:
            return "stopping"
        if self.muted:
            return "muted"
        if self.searching:
            return "searching"
        if self.agent_state in ("speaking", "thinking"):
            return self.agent_state
        if self.asleep:
            return "asleep"
        if self.user_speaking:
            return "hearing"
        return "listening" if self.agent_state != "starting" else "starting"

    def set_agent_state(self, state: str) -> None:
        self.agent_state = state
        if state in ("idle", "listening"):
            self.searching = ""            # safety net if a search was cancelled by a barge-in
        self.poke()

    def set_awake(self, awake: bool, reason: str = "") -> None:
        """Wake gate opened / closed (called from the agent's event loop)."""
        self.asleep = not awake
        if reason != "start":
            self.notify(f"Awake: listening ({reason})" if awake else f"Asleep: say {self.wake_name} ({reason})",
                        "ok" if awake else "info", chat=False)
        self.poke()

    def toggle_wake(self) -> None:
        if not self.on_wake:
            return
        try:
            self.on_wake()
        except Exception as e:
            self.notify(f"Wake failed: {e!r}", "error", chat=False)

    def set_user_state(self, state: str) -> None:
        self.user_speaking = state == "speaking"
        self.poke()

    def set_latency(self, *, stt=None, eot=None, llm=None, tts=None, e2e=None, search=None) -> None:
        self.lat = dict(stt=stt, eot=eot, llm=llm, tts=tts, e2e=e2e, search=search)
        if isinstance(e2e, (int, float)):
            self.e2e_history = (self.e2e_history + [float(e2e)])[-48:]
        self.poke()

    def poke(self) -> None:
        """Ask the draw thread to redraw right now (key presses use this, so they feel instant)."""
        self._wake.set()

    @property
    def last_usage(self) -> Optional[dict]:
        return self._last_usage

    @last_usage.setter
    def last_usage(self, u: Optional[dict]) -> None:
        """The agent sets this after every reply; we also keep the reply's cost for the cost panel."""
        self._last_usage = u
        if u and u.get("cost") is not None:
            total = _parts(u["cost"])["total"]
            self.cost_history.append(total)
            self.replies_cost += total
        self.poke()

    # ---------------- conversation (drawn inside the screen) ----------------
    def set_partial(self, text: str) -> None:
        self.partial = (text or "").strip()
        self.poke()

    def user(self, text: str) -> None:
        self.partial = ""
        self.turns += 1
        self.scroll = 0
        self.chat.append(("user", text, None, time.strftime("%H:%M")))
        self.poke()

    def agent(self, text: str, e2e_ms=None) -> None:
        self.scroll = 0
        self.chat.append(("agent", text, e2e_ms if isinstance(e2e_ms, (int, float)) else None,
                          time.strftime("%H:%M")))
        self.poke()

    def notify(self, text: str, kind: str = "info", chat: bool = True) -> None:
        """A short banner in the header for 5 s, plus (optionally) a line in the conversation."""
        self.toast_id += 1
        self.toast = (text, kind, time.monotonic() + 5.0)
        if chat:
            self.scroll = 0
            self.chat.append(("system", text, kind, time.strftime("%H:%M")))
        self.log.append(Text(f"[ui] {text}", style=TOAST_COLORS.get(kind, C_BLUE)))
        self.poke()

    # ---------------- language ----------------
    def lang_label(self, code: str) -> str:
        for c, name, _ in self.langs:
            if c == code:
                return name
        return code

    def set_language(self, code: str, source: str = "") -> None:
        """The agent calls this after it really switched (voice request or menu)."""
        self.lang_mode = code
        by = {"voice": "you asked for it", "menu": "from the menu"}.get(source, "")
        self.notify(f"Language changed to {self.lang_label(code)}" + (f"  ·  {by}" if by else ""), "lang")

    def open_menu(self) -> None:
        self.menu_idx = next((i for i, l in enumerate(self.langs) if l[0] == self.lang_mode), 0)
        self.menu_open = True

    def choose_language(self, code: str) -> None:
        self.menu_open = False
        if code == self.lang_mode:
            self.notify(f"Already replying in {self.lang_label(code)}", "info", chat=False)
            return
        if self.on_language:
            try:
                self.on_language(code)          # the agent applies it, then calls set_language()
            except Exception as e:
                self.notify(f"Language change failed: {e!r}", "error")
        else:
            self.set_language(code, "menu")

    def menu_size(self) -> tuple[int, int]:
        return 40, len(self.langs) + 3          # one row per language + hint row + 2 border rows

    def menu_panel(self, width: int) -> Panel:
        rows = []
        for i, (code, name, hint) in enumerate(self.langs):
            cur, active = i == self.menu_idx, code == self.lang_mode
            row = Text(no_wrap=True, overflow="ellipsis")
            row.append(" ▸ " if cur else "   ", style=f"bold {C_PURPLE}")
            row.append("●" if active else "○", style=C_GREEN if active else C_DIM)
            row.append(f" {i + 1}  ", style=C_DIM)
            row.append(f"{name:<9}", style=f"bold {C_TEXT}")
            row.append(hint, style=C_SOFT if cur else C_DIM)
            if cur:
                row.stylize(f"on {C_LINE}")
            rows.append(row)
        rows.append(Text(" ↑↓ move  ⏎ pick  esc close", style=C_DIM, no_wrap=True))
        title = Text.assemble((" ⇄ ", C_PURPLE), ("Reply language ", f"bold {C_TEXT}"))
        return Panel(Group(*rows), title=title, title_align="left", box=box.ROUNDED,
                     border_style=C_PURPLE, padding=(0, 0), width=width)

    def _menu_key(self, k: str) -> None:
        n = len(self.langs)
        if k in ("up", "k"):
            self.menu_idx = (self.menu_idx - 1) % n
        elif k in ("down", "j", "tab"):
            self.menu_idx = (self.menu_idx + 1) % n
        elif k == "enter" or k == " ":
            self.choose_language(self.langs[self.menu_idx][0])
        elif k in ("esc", "q", "l"):
            self.menu_open = False
        elif k.isdigit() and 1 <= int(k) <= n:
            self.choose_language(self.langs[int(k) - 1][0])

    # ---------------- mute / quit ----------------
    def toggle_mute(self) -> None:
        self.muted = not self.muted
        self.user_speaking = False
        try:
            if self.on_mute:
                self.on_mute(self.muted)
            self.notify("Microphone muted" if self.muted else "Microphone on", "info", chat=False)
        except Exception as e:
            self.muted = not self.muted    # did not take effect: show the truth
            self.notify(f"Mute failed: {e!r}", "error", chat=False)

    def interrupt(self) -> None:
        """Stop the agent talking right now (the way to cut in when the echo guard has closed the mic)."""
        if not self.on_interrupt:
            return
        try:
            self.on_interrupt()
            self.notify("Stopped speaking", "info", chat=False)
        except Exception as e:
            self.notify(f"Interrupt failed: {e!r}", "error", chat=False)

    def quit(self) -> None:
        """Ask the agent to shut down now, and force it if it is still alive after a few seconds."""
        if self.quitting:                  # second press: leave immediately
            try:
                self.stop()
            finally:
                os._exit(0)
        self.quitting = True
        self.poke()

        def _force() -> None:
            try:
                self.stop()
            finally:
                os._exit(0)
        t = threading.Timer(4.0, _force)
        t.daemon = True
        t.start()
        if os.name == "nt":
            signal.raise_signal(signal.SIGINT)
        else:
            os.kill(os.getpid(), signal.SIGINT)

    def _on_key(self, k: str) -> None:
        if self.menu_open:
            self._menu_key(k)
        elif k in ("m", " "):
            self.toggle_mute()
        elif k == "q":
            self.quit()
        elif k in ("l", "tab"):
            self.open_menu()
        elif k in ("s", "esc"):
            self.interrupt()
        elif k == "w":
            self.toggle_wake()
        elif k.isdigit() and 1 <= int(k) <= len(self.langs):
            self.choose_language(self.langs[int(k) - 1][0])
        elif k in ("k", "up"):
            self.scroll += 3
        elif k in ("j", "down"):
            self.scroll = max(0, self.scroll - 3)
        elif k == "pgup":
            self.scroll += 10
        elif k == "pgdn":
            self.scroll = max(0, self.scroll - 10)
        elif k in ("b", "end"):
            self.scroll = 0

    # ---------------- rendering ----------------
    def _eq(self, n: int = 7) -> str:
        t = time.monotonic() * 7.0
        return "".join(BARS[int((math.sin(t + i * 1.25) + 1) / 2 * (len(BARS) - 1))] for i in range(n))

    def _header(self) -> Panel:
        key = self.status
        label, style, accent = BADGES[key]
        if key == "searching":
            label = {"weather": "WEATHER"}.get(self.searching, "SEARCHING")
        elif key == "asleep" and self.wake_name:
            label = f"ASLEEP · SAY {self.wake_name.upper()}"
        left = Text.assemble((f" {label} ", style))
        if key in ACTIVE:
            left.append("  " + self._eq(), style=accent)
        mic = (Text(" ✖ MIC MUTED ", style=f"bold {C_TEXT} on {C_RED}") if self.muted
               else Text(" ● MIC ON ", style=f"bold {C_INK} on {C_GREEN}"))
        lang = Text(f" ⇄ {self.lang_label(self.lang_mode).upper()} ", style=f"bold {C_INK} on {C_PURPLE}")
        mins = int((time.monotonic() - self.started) // 60)
        right = Text.assemble((self.voice, C_BLUE), (f"   {mins}m · {self.turns} turns", C_DIM))

        top = Table.grid(expand=True)
        top.add_column(ratio=1)
        top.add_column(justify="center")
        top.add_column(justify="right", ratio=1)
        cost_pill = Text("")
        if self.meter:
            try:
                cost_pill = Text(f" ◆ {self.meter.cost_text(self.meter.session_cost())} ",
                                 style=f"bold {C_INK} on {C_ORANGE}", no_wrap=True)
            except Exception:
                pass
        top.add_row(left, Text.assemble(mic, "  ", lang, "  " if cost_pill.plain else "", cost_pill), right)

        if self.toast and time.monotonic() < self.toast[2]:
            text, kind, _ = self.toast
            sub = Text.assemble((" ✖ " if kind == "error" else " ✔ ", f"bold {TOAST_COLORS.get(kind, C_BLUE)}"),
                                (text, f"bold {TOAST_COLORS.get(kind, C_BLUE)}"))
        else:
            sub = Text.assemble(("STT ", C_DIM), (self.stt, C_SOFT), ("   LLM ", C_DIM), (self.llm, C_SOFT),
                                ("   TTS ", C_DIM), (self.tts, C_SOFT),
                                *([("   ECHO ", C_DIM), (self.guard, C_SOFT)] if self.guard else []))
        title = Text.assemble((" ◖ ", accent), ("VOICE AGENT", f"bold {C_TEXT}"), (" ◗ ", accent))
        return Panel(Group(top, sub), box=box.ROUNDED, border_style=accent, title=title, title_align="left",
                     padding=(0, 1))

    def _chat_lines(self, width: int, count: int) -> list:
        if count <= 0:
            return []
        width = max(width, 24)
        body_w = max(width - GUTTER, 10)
        pad = " " * GUTTER

        def block(ts: str, who: str, chip_style: str, text: str, body_style: str, tail: Optional[Text] = None):
            body = Text(text, style=body_style)
            wrapped = list(body.wrap(self.console, body_w, justify="left", overflow="fold")) or [Text("")]
            out = []
            for i, ln in enumerate(wrapped):
                t = Text()
                if i == 0:
                    t.append(f" {ts} ", style=C_DIM)
                    t.append(f" {who:<5} ", style=chip_style)
                    t.append(" ")
                else:
                    t.append(pad)
                t.append_text(ln)
                if tail is not None and i == len(wrapped) - 1:
                    t.append_text(tail)
                out.append(t)
            return out

        extra: list = []                                   # always pinned at the bottom
        if self.muted:
            extra.append(Text(" ▌ MIC MUTED · nobody can hear you · press M to unmute ",
                              style=f"bold {C_TEXT} on {C_RED}"))
        elif self.asleep and self.wake_name and not self.partial:
            extra.append(Text(f" ▌ asleep · say \"{self.wake_name}\" (or press W) to wake me ",
                              style=f"bold {C_TEXT} on {C_LINE}"))
        elif self.partial:
            extra.extend(block("", "YOU", f"bold {C_DIM} on {C_LINE}", self.partial + " …",
                               f"italic {C_SOFT}")[-2:])
        elif self.status in ("thinking", "searching"):
            dots = "●" * (1 + int(time.monotonic() * 3) % 3)
            extra.append(Text(pad + dots, style=C_YELLOW if self.status == "thinking" else C_PURPLE))

        lines: list = []
        for role, text, extra_v, ts in list(self.chat)[-120:]:
            if role == "user":
                lines.extend(block(ts, "YOU", f"bold {C_INK} on {C_GREEN}", text, C_TEXT))
            elif role == "agent":
                tail = Text(f"   {extra_v / 1000:.1f}s", style=_lat_style(extra_v)) if extra_v is not None else None
                lines.extend(block(ts, "AGENT", f"bold {C_INK} on {C_CYAN}", text, "white", tail))
                lines.append(Text(""))
            else:                                          # system line, e.g. language changed
                color = TOAST_COLORS.get(extra_v, C_BLUE)
                for ln in Text(f"⇄  {text}", style=f"bold {color}").wrap(self.console, width - 6):
                    lines.append(Text.assemble("   ", ln))
                lines.append(Text(""))
        room = max(count - len(extra), 0)
        self.scroll = min(self.scroll, max(len(lines) - room, 0))
        end = len(lines) - self.scroll
        window = lines[max(end - room, 0):end]
        return window + extra

    def _stats(self):
        rows = []
        e2e = self.lat.get("e2e")
        head = Text()
        head.append("LATENCY  ", style=f"bold {C_DIM}")
        if e2e is None:
            head.append("waiting for the first reply", style=C_DIM)
        else:
            head.append(f"{e2e / 1000:.2f}s", style=f"bold {_lat_style(e2e)}")
            head.append("  " + _braille(self.e2e_history, top=max(2000.0, *self.e2e_history), width=24),
                        style=_lat_style(e2e))
        rows.append(head)

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style=C_DIM, no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        for name, key in (("stt", "stt"), ("end of turn", "eot"), ("llm", "llm"), ("tts", "tts"), ("search", "search")):
            v = self.lat.get(key)
            if isinstance(v, (int, float)):
                grid.add_row(name, _bar(v), Text(f"{v:.0f} ms", style=_stage_style(v)))
            else:
                grid.add_row(name, _bar(0), Text("-", style=C_DIM))
        rows += [Text(""), grid, Text("")]

        h = self.e2e_history
        if len(h) >= 2:
            rows.append(Text(f"avg {sum(h) / len(h) / 1000:.2f}s   best {min(h) / 1000:.2f}s   "
                             f"worst {max(h) / 1000:.2f}s", style=C_DIM))
        return Group(*rows)

    # ---------------- cost ----------------
    def _currency(self) -> str:
        m = self.meter
        for obj in (m, getattr(m, "prices", None)):
            c = getattr(obj, "currency", None)
            if isinstance(c, str) and c and c != "?":
                return c
        return ""

    @staticmethod
    def _usage_text(part: str, s: dict) -> str:
        if part == "llm":
            n = int(s.get("llm_prompt_tokens", 0) + s.get("llm_completion_tokens", 0))
            return f"{n / 1000:.1f}k tok" if n >= 1000 else f"{n} tok"
        if part == "tts":
            return f"{_fmt(s.get('tts_chars', 0))} ch"
        if part == "stt":
            return f"{s.get('stt_audio_s', 0):.0f} s"
        return f"{_fmt(s.get('tavily_credits', 0))} tav / {_fmt(s.get('exa_requests', 0))} exa"

    def _cost_panel(self) -> Panel:
        cur = self._currency()
        title = Text.assemble((" ", ""), ("COST", f"bold {C_TEXT}"), (f" · {cur} " if cur else " ", C_DIM))
        box_kw = dict(title=title, title_align="left", box=box.ROUNDED, border_style=C_LINE, padding=(0, 1))
        if not self.meter:
            return Panel(Text("no usage meter attached", style=C_DIM), **box_kw)
        m = self.meter
        sess = m.session
        raw = m.session_cost()
        c = _parts(raw)
        shown = sum(c[k] for k in COST_PARTS)
        mins = (time.monotonic() - self.started) / 60
        rows: list = []

        head = Text()
        head.append(m.cost_text(raw), style=f"bold {C_PURPLE}")
        head.append("  this session", style=C_DIM)
        if mins >= 0.75 and c["total"] > 0:
            head.append(f"   ≈ {c['total'] / mins:.2f}/min", style=C_DIM)
        rows.append(head)

        W = 30                                             # one stacked bar: who got the money
        stack, done, acc = Text(), 0, 0.0
        if shown > 0:
            for k in COST_PARTS:
                acc += c[k]
                edge = round(acc / shown * W)
                stack.append("█" * (edge - done), style=COST_COLORS[k])
                done = edge
        stack.append("░" * (W - done), style=C_LINE)
        rows += [stack, Text("")]

        grid = Table.grid(padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_column(no_wrap=True, overflow="crop", style=C_DIM)
        for k in COST_PARTS:
            share = c[k] / shown if shown else 0.0
            grid.add_row(Text(k.upper(), style=f"bold {COST_COLORS[k]}"),
                         Text(_cbar(share), style=COST_COLORS[k] if c[k] else C_LINE),
                         Text(f"{c[k]:.2f}", style=C_TEXT if c[k] else C_DIM),
                         Text(f"{share * 100:.0f}%", style=C_SOFT if c[k] else C_DIM),
                         self._usage_text(k, sess))
        rows.append(grid)

        over = max(c["total"] - self.replies_cost, 0.0)
        if c["total"] > 0:
            rows.append(Text(f" ↳ overhead {over:.2f} ({over / c['total'] * 100:.0f}%) fillers+unfinished",
                             style=C_DIM, no_wrap=True, overflow="ellipsis"))
        unpriced = raw.get("unpriced") if isinstance(raw, dict) else None
        if unpriced:
            names = ", ".join(unpriced) if isinstance(unpriced, (list, tuple, set)) else str(unpriced)
            rows.append(Text(f" ✖ UNPRICED: {names} (counted as 0, run price_check.py)", style=f"bold {C_RED}",
                             no_wrap=True, overflow="ellipsis"))
        rows.append(Text(""))

        u = self.last_usage
        last = Text()
        last.append("LAST REPLY  ", style=f"bold {C_DIM}")
        if u and u.get("cost") is not None:
            lp = _parts(u["cost"])
            last.append(m.cost_text(u["cost"]), style=f"bold {C_PURPLE}")
            big = max(COST_PARTS, key=lambda k: lp[k])
            if lp["total"] > 0 and lp[big] > 0:
                last.append(f"  mostly {big.upper()} {lp[big] / max(sum(lp[k] for k in COST_PARTS), 1e-9) * 100:.0f}%",
                            style=COST_COLORS[big])
        else:
            last.append("nothing yet", style=C_DIM)
        rows.append(last)
        if u:
            line = (f" {_fmt(u.get('llm_prompt_tokens', 0))} in / {_fmt(u.get('llm_completion_tokens', 0))} out"
                    f" · {_fmt(u.get('tts_chars', 0))} ch" + (" (est.)" if u.get("tts_estimated") else ""))
            calls = u.get("tavily_requests", 0) + u.get("exa_requests", 0)
            if calls:
                line += f" · {_fmt(calls)} search"
            rows.append(Text(line, style=C_SOFT, no_wrap=True, overflow="ellipsis"))
        hist = list(self.cost_history)
        if len(hist) >= 2:
            rows.append(Text.assemble((" per reply ", C_DIM), (_braille(hist, width=20), C_PURPLE),
                                      (f"  avg {sum(hist) / len(hist):.3f}  max {max(hist):.3f}", C_DIM)))
        return Panel(Group(*rows), **box_kw)

    def _footer(self) -> Text:
        if os.getenv("UI_KEYS", "1") == "0":
            return Text(" Ctrl+C to quit", style=C_DIM)
        if self.menu_open:
            parts = [_chip("↑↓", "move"), _chip("⏎", "pick"), _chip("1-%d" % len(self.langs), "quick"),
                     _chip("esc", "close")]
        else:
            parts = [_chip("M", "unmute" if self.muted else "mute"), _chip("S", "stop talking"),
                     *([_chip("W", "wake/sleep")] if self.on_wake else []), _chip("L", "language"),
                     _chip("1-%d" % len(self.langs), "quick lang"), _chip("J/K", "scroll"),
                     _chip("B", "latest"), _chip("Q", "quit")]
        t = Text(" ")
        for p in parts:
            t.append_text(p)
        return t

    def render(self):
        w, h = self.console.size
        muted = self.muted
        header = self._header()

        log_h = 8 if h >= 32 else 6 if h >= 22 else 0
        body_h = max(h - 4 - log_h - 1, 4)
        wide = w >= 100
        stats_h = 0 if wide else min(17, max(body_h // 2, 0))
        chat_h = body_h if wide else body_h - stats_h
        chat_w = (w * 3 // 5 if wide else w) - 4

        chat_body = self._chat_lines(chat_w, chat_h - 2)
        chat_title = Text.assemble((" ", ""), ("CONVERSATION", f"bold {C_TEXT}"), (" ", ""))
        chat = Panel(Group(*chat_body), title=chat_title, title_align="left",
                     subtitle=(f"[{C_YELLOW}] scrolled up {self.scroll} lines · B = latest [/]" if self.scroll else None),
                     box=box.ROUNDED, border_style=C_RED if muted else C_LINE, padding=(0, 1))
        stats = Panel(self._stats(), title=Text.assemble((" ", ""), ("STATS", f"bold {C_TEXT}"), (" ", "")),
                      title_align="left", box=box.ROUNDED, border_style=C_LINE, padding=(0, 1))

        root = Layout()
        parts = [Layout(header, name="head", size=4)]
        body = Layout(name="body", ratio=1)
        cost = self._cost_panel()
        if wide:
            side = Layout(name="side", ratio=2)
            side.split_column(Layout(stats, name="stats", ratio=1),
                              Layout(cost, name="cost", size=min(16, max(body_h - 8, 7))))
            body.split_row(Layout(chat, name="chat", ratio=3), side)
        else:
            low = Layout(name="low", size=max(stats_h, 3))
            low.split_row(Layout(stats, name="stats", ratio=1), Layout(cost, name="cost", ratio=1))
            body.split_column(Layout(chat, name="chat", ratio=1), low)
        parts.append(body)
        if log_h:
            lines = []
            for t in list(self.log)[-(log_h - 2):]:
                t = t.copy()
                t.no_wrap = True
                t.overflow = "ellipsis"
                lines.append(t)
            parts.append(Layout(Panel(Group(*lines), title=Text.assemble((" ", ""), ("LOG", f"bold {C_DIM}"), (" ", "")),
                                      title_align="left", box=box.ROUNDED, border_style=C_LINE, padding=(0, 1)),
                                name="log", size=log_h))
        parts.append(Layout(self._footer(), name="foot", size=1))
        root.split_column(*parts)
        return root

    # ---------------- output capture ----------------
    def _add_log(self, text: Text) -> None:
        plain = text.plain
        if "ERROR" in plain or "CRITICAL" in plain or "Traceback" in plain:
            text.stylize(C_RED)
        elif "WARNING" in plain or "WARN" in plain:
            text.stylize(C_YELLOW)
        self.log.append(text)

    def _capture_output(self) -> None:
        """Point fds 1 and 2 at a pipe (so prints and log lines cannot garble the screen) and draw on a
        private duplicate of the real terminal."""
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            saved1, saved2 = os.dup(1), os.dup(2)
            self._term_fd = os.dup(saved1)
            out = os.fdopen(os.dup(saved1), "w", encoding="utf-8", errors="replace")
            r, w = os.pipe()
            os.dup2(w, 1)
            os.dup2(w, 2)
            os.close(w)
        except OSError:
            self._saved = None
            return
        self._saved, self._pipe_r = (saved1, saved2), r
        con = _TermConsole(file=out, force_terminal=True)
        con.term_fd = self._term_fd
        self.console = con
        self._reader = threading.Thread(target=self._read_pipe, name="ui-log", daemon=True)
        self._reader.start()

    def _read_pipe(self) -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(self._pipe_r, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                line = raw.decode("utf-8", "replace").replace("\r", "").rstrip()
                if not line:
                    continue
                try:
                    self._add_log(Text.from_ansi(line))
                except Exception:
                    self._add_log(Text(line))

    def _restore_output(self) -> None:
        if not self._saved:
            return
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(self._saved[0], 1)
            os.dup2(self._saved[1], 2)
            os.close(self._saved[0])
            os.close(self._saved[1])
        except OSError:
            pass
        self._saved = None
        if self._reader:
            self._reader.join(timeout=1.0)
        if self._pipe_r is not None:
            try:
                os.close(self._pipe_r)
            except OSError:
                pass
            self._pipe_r = None

    # ---------------- keyboard ownership ----------------
    def _claim_stdin(self) -> None:
        """Take the keyboard away from every other reader of file descriptor 0 (LiveKit's console keeps
        one running). fd 0 is pointed at a dummy pseudo-terminal that nobody types into; we keep a private
        copy of the real terminal in self._tty_fd and read keys from that."""
        if os.name == "nt" or not sys.stdin.isatty():
            return
        try:
            import pty
            import termios
            real = os.dup(0)
            try:
                self._tty_attrs = termios.tcgetattr(real)
            except termios.error:
                self._tty_attrs = None
            master, slave = pty.openpty()
            os.dup2(slave, 0)
            os.close(slave)
            self._pty_master, self._tty_fd = master, real
            try:        # wake a reader that is already blocked on the real terminal, so it does not eat our first key
                import fcntl
                fcntl.ioctl(real, termios.TIOCSTI, b"\x00")
            except Exception:
                pass
        except Exception as e:
            self._add_log(Text(f"WARNING keyboard takeover failed ({e!r}); keys may need several presses", style=C_YELLOW))

    def _release_stdin(self) -> None:
        if self._tty_fd is None:
            return
        try:
            import termios
            os.dup2(self._tty_fd, 0)
            attrs = self._tty_attrs
            if attrs:
                attrs[3] |= termios.ICANON | termios.ECHO | termios.ISIG    # back to normal line mode
                termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, attrs)
        except Exception:
            pass
        for fd in (self._tty_fd, self._pty_master):
            try:
                if fd is not None:
                    os.close(fd)
            except OSError:
                pass
        self._tty_fd = self._pty_master = None

    # ---------------- lifecycle ----------------
    def _signature(self):
        """Everything the screen shows. If it did not change, we do not redraw (redraws are what flicker)."""
        m = self.meter.session if self.meter else {}
        u = self.last_usage or {}
        now = time.monotonic()
        anim = int(now * 8) if self.status in ACTIVE else 0
        toast_on = bool(self.toast and now < self.toast[2])
        return (self.console.size, self.status, self.voice, self.muted, self.asleep, tuple(sorted(self.lat.items())),
                len(self.chat), len(self.log), self.partial, self.scroll, self.turns, int((now - self.started) // 60),
                tuple(self.e2e_history[-1:]), u.get("time"), self.lang_mode, self.menu_open, self.menu_idx,
                self.toast_id, toast_on, anim, len(self.cost_history),
                m.get("llm_prompt_tokens"), m.get("llm_completion_tokens"), m.get("tts_chars"),
                round(m.get("stt_audio_s", 0)), m.get("tavily_credits"), m.get("exa_requests"),
                self.meter.cost_text(self.meter.session_cost()) if self.meter else "")

    def _tick_loop(self) -> None:
        fps = max(float(os.getenv("UI_FPS", "8")), 0.5)
        while not self._stop.is_set():
            self._wake.wait(1.0 / fps)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                sig = self._signature()
                if sig != self._sig and self._live:
                    self._sig = sig
                    self._live.refresh()
            except Exception:
                pass

    def start(self) -> None:
        if self._live:
            return
        keys_on = sys.stdin.isatty() and os.getenv("UI_KEYS", "1") != "0"
        self._capture_output()
        self._live = Live(get_renderable=lambda: self._screen, console=self.console, screen=True, auto_refresh=False,
                          redirect_stdout=False, redirect_stderr=False, vertical_overflow="crop")
        self._live.start(refresh=True)
        atexit.register(self.stop)
        self._tick = threading.Thread(target=self._tick_loop, name="ui-tick", daemon=True)
        self._tick.start()
        if keys_on:
            def _keys_main() -> None:        # the keyboard takeover can block for seconds: never on the event loop
                self._claim_stdin()
                self._key_loop()
            self._keys = threading.Thread(target=_keys_main, name="ui-keys", daemon=True)
            self._keys.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        self._wake.set()
        for t in (self._keys, self._tick):
            if t:
                t.join(timeout=1.0)
        if self._live:
            try:
                self._live.stop()                 # leaves the full-screen mode
            except Exception:
                pass
            self._live = None
        self._release_stdin()
        self._restore_output()
        # the screen is gone: print what the log still holds (the session summary, errors) on the real terminal
        tail = list(self.log)[-15:]
        if tail:
            print("\n".join(t.plain for t in tail))

    def _key_loop(self) -> None:
        if os.name == "nt":
            import msvcrt
            special = {"H": "up", "P": "down", "I": "pgup", "Q": "pgdn", "O": "end"}
            while not self._stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):                    # special key: next char is the code
                        ch = special.get(msvcrt.getwch(), "")
                    elif ch == "\r":
                        ch = "enter"
                    elif ch == "\x1b":
                        ch = "esc"
                    elif ch == "\t":
                        ch = "tab"
                    if ch:
                        self._on_key(ch.lower() if len(ch) == 1 else ch)
                        self.poke()
                else:
                    time.sleep(0.02)
            return
        import select
        import termios
        import tty
        fd = self._tty_fd if self._tty_fd is not None else sys.stdin.fileno()
        try:
            termios.tcgetattr(fd)
        except termios.error:
            return
        last_check = 0.0
        try:
            tty.setcbreak(fd, termios.TCSANOW)    # single keys, no Enter, and never discard typed-ahead input
            while not self._stop.is_set():
                now = time.monotonic()
                if now - last_check > 0.25:       # something else put the terminal back in line mode? undo it
                    last_check = now
                    try:
                        if termios.tcgetattr(fd)[3] & termios.ICANON:
                            tty.setcbreak(fd, termios.TCSANOW)
                    except termios.error:
                        pass
                if not select.select([fd], [], [], 0.05)[0]:
                    continue
                try:
                    data = os.read(fd, 64)
                except OSError:
                    break
                if not data:
                    time.sleep(0.05)
                    continue
                for key in decode_keys(data):
                    try:
                        self._on_key(key)
                    except Exception as e:
                        self._add_log(Text(f"ERROR key handler: {e!r}", style=C_RED))
                self.poke()
        finally:
            pass

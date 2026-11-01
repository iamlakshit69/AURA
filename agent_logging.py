"""
One log file for everything: agent.log (LOG_FILE to change). It is opened in APPEND mode and never
truncated or rotated, so every run adds to the end and nothing from before is lost.

Written there: session start/end with totals and cost, every reply (latency, tokens, characters,
cost), every search request, price-file problems, and all warnings/errors (also from LiveKit and
uncaught exceptions). LOG_LEVEL=DEBUG for more.
"""
from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_MARK = "_voice_agent_log"
_hook_set = False


def _has_handler(logger: logging.Logger, path: str) -> bool:
    return any(getattr(h, _MARK, None) == path for h in logger.handlers)


def _file_handler(path: str, level: int) -> logging.Handler:
    h = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
    h.setLevel(level)
    h.setFormatter(logging.Formatter(_FORMAT))
    setattr(h, _MARK, path)
    return h


def setup_logging(path: str | None = None, level: str | None = None) -> None:
    """Safe to call again (the entrypoint does, because LiveKit may reset logging when it starts)."""
    global _hook_set
    path = path or os.getenv("LOG_FILE", "agent.log")
    lvl = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)

    app = logging.getLogger("voice-agent")
    app.setLevel(lvl)
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if name.startswith("voice-agent") and isinstance(lg, logging.Logger):
            lg.disabled = False                     # a dictConfig elsewhere may have switched it off
    fresh = not _has_handler(app, path)
    if fresh:
        app.addHandler(_file_handler(path, lvl))
    lk = logging.getLogger("livekit")               # library problems, warnings and up
    if not _has_handler(lk, path):
        lk.addHandler(_file_handler(path, logging.WARNING))
    if fresh:
        app.info("=" * 20 + " process %d started " % os.getpid() + "=" * 20)

    if not _hook_set:
        _hook_set = True
        previous = sys.excepthook

        def _hook(exc_type, exc, tb):
            if not issubclass(exc_type, KeyboardInterrupt):
                app.critical("uncaught exception", exc_info=(exc_type, exc, tb))
            previous(exc_type, exc, tb)
        sys.excepthook = _hook

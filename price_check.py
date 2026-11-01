#!/usr/bin/env python3
"""
Makes sure every model in settings.env has a price in prices.ini. Asks you for the missing ones.

    python price_check.py            # ask only for what is missing
    python price_check.py --edit     # go through ALL current models again (Enter keeps the shown value)

The agent runs this at startup too (Structure-II.py console / dev / start). Without a terminal to ask in,
or with ASK_PRICES=0, it only warns, and the unpriced parts show as UNPRICED instead of using an old price.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from pricing import Prices
from settings import get_models, load_settings

PRICES_FILE = os.getenv("PRICES_FILE", "prices.ini")


def _ask_item(item, prices: Prices, ask) -> bool:
    """Ask for the prices of one item. Returns True when every required price was entered."""
    print(f"\n  {item.label}")
    cur = item.current.get("currency") or ""
    currency = ask(f"    currency [{cur or item.currency}]: ").strip().upper() or cur or item.currency
    values = {"currency": currency}
    ok = True
    for key, question in item.fields:
        shown = item.current.get(key, "")
        answer = ask(f"    {question}{f' [{shown}]' if shown else ''} {'(0 = free, Enter = skip)' if key in item.required else '(0 = free)'}: ").strip()
        if answer == "":
            if shown:
                values[key] = shown
            elif key in item.required:
                ok = False
            continue
        try:
            float(answer)
        except ValueError:
            print(f"    '{answer}' is not a number, skipped")
            ok = ok and key not in item.required
            continue
        values[key] = answer
    if len(values) > 1:
        prices.save(item.section, values)
    return ok


def ensure_prices(force: bool = False, ask=input, interactive: bool | None = None) -> list:
    """Returns the items that are still without a price afterwards."""
    prices = Prices(PRICES_FILE, get_models())
    tavily, exa = bool(os.getenv("TAVILY_API_KEY")), bool(os.getenv("EXA_API_KEY"))
    todo = prices.items(tavily=tavily, exa=exa) if force else prices.missing(tavily=tavily, exa=exa)
    if not todo:
        return []
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty() and os.getenv("ASK_PRICES", "1") != "0"
    names = "\n".join(f"  - {it.label}" for it in todo)
    if not interactive:
        print(f"[prices] no price yet in {PRICES_FILE} for:\n{names}\n"
              f"         these show as UNPRICED (an old price is never reused). Run: python price_check.py")
        return todo
    m = prices.models
    print(f"\n[prices] {'Review prices' if force else 'These need a price in ' + PRICES_FILE} "
          f"(models come from settings.env; prices of other models are NOT reused):\n{names}")
    for kind in ("llm", "tts", "stt"):
        known = [x for x in prices.known_models(kind) if x != m[kind]]
        if known and any(it.section == f"{kind}:{m[kind]}" for it in todo):
            print(f"  (already priced {kind.upper()} models: {', '.join(known)})")
    try:
        left = [it for it in todo if not _ask_item(it, prices, ask)]
    except (EOFError, KeyboardInterrupt):
        print("\n[prices] cancelled, nothing more saved")
        return todo
    if left:
        print(f"\n[prices] still unpriced: {', '.join(it.label for it in left)}. They show as UNPRICED until you fill them in.")
    else:
        print("\n[prices] all set.\n")
    return left


if __name__ == "__main__":
    load_settings()
    load_dotenv()
    ensure_prices(force="--edit" in sys.argv)

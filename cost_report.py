#!/usr/bin/env python3
"""
Price your usage history.   python cost_report.py [usage_log.csv] [--since 2026-11-01]

Prices come from prices.ini, found by the model names (settings.env; older rows: the model recorded in the log). This RE-PRICES the raw usage in usage_log.csv
with the prices in prices.ini today, so you can ask "what would all of this cost at the new prices?".
Writes usage_costs.csv with a cost column per reply.

The prices that were valid at the time are in cost_log.csv (written live by the agent): its
kind=session rows are the exact total of every session, filler clips included. They are listed below
when the file exists.
"""
import argparse
import csv
import os
from collections import defaultdict

from pricing import COST_KEYS, Prices, num
from settings import get_models, load_settings

load_settings()
CURRENT = get_models()

ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?", default="usage_log.csv")
ap.add_argument("--since", default="", help="only rows from this date on, e.g. 2026-11-01")
ap.add_argument("--prices", default=os.getenv("PRICES_FILE", "prices.ini"))
ap.add_argument("--cost-log", default=os.getenv("COST_CSV", "cost_log.csv"))
args = ap.parse_args()

prices = Prices(args.prices, CURRENT)
cur = prices.currency
with open(args.log, newline="", encoding="utf-8-sig") as fh:
    rows = [r for r in csv.DictReader(fh) if r.get("time", "") >= args.since]
if not rows:
    raise SystemExit("no rows yet")
no_model = sum(1 for r in rows if not r.get("tts_model"))
if no_model:
    print(f"NOTE: {no_model} older rows do not record the TTS/STT model: they are priced with the models in settings.env "
          f"({CURRENT['tts']}, {CURRENT['stt']}).\n")

out, tot, by_model = [], dict.fromkeys(COST_KEYS + ["total"], 0.0), defaultdict(lambda: dict.fromkeys(COST_KEYS + ["total"], 0.0))
usage = defaultdict(float)
unpriced = defaultdict(int)
for r in rows:
    c = prices.cost(r, {"llm": r.get("llm", ""), "tts": r.get("tts_model", ""), "stt": r.get("stt_model", "")})
    for u in c["unpriced"]:
        unpriced[u] += 1
    for k in tot:
        tot[k] += c[k]
        by_model[r.get("llm", "?")][k] += c[k]
    for k, v in r.items():
        if k not in ("time", "llm", "tts_model", "stt_model"):
            usage[k] += num(v)
    out.append({**r, **{f"cost_{k}": round(c[k], 6) for k in COST_KEYS + ["total"]}, "currency": cur,
                "unpriced": ";".join(c["unpriced"])})
with open("usage_costs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

n = len(rows)
print(f"{n} replies from {rows[0]['time']} to {rows[-1]['time']}")
print(f"  LLM    {usage['llm_prompt_tokens']:.0f} in ({usage['llm_cached_tokens']:.0f} cached) / {usage['llm_completion_tokens']:.0f} out tokens"
      f"  -> {usage['llm_prompt_tokens']/n:.0f} in, {usage['llm_completion_tokens']/n:.0f} out per reply")
print(f"  TTS    {usage['tts_chars']:.0f} chars  -> {usage['tts_chars']/n:.0f} per reply (filler pre-render not in this log)")
print(f"  STT    {usage['stt_audio_s']:.1f} s audio  -> {usage['stt_audio_s']/n:.1f} s per reply (per-row STT is lumpy; the total is right)")
print(f"  Search tavily {usage['tavily_credits']:.0f} credits / exa {usage['exa_requests']:.0f} requests / weather {usage['weather_calls']:.0f} calls")
if usage.get("tts_estimated"):
    print(f"  note: {usage['tts_estimated']:.0f} reply(ies) had no TTS metrics, their tts_chars is the reply length")
print(f"\nCost at today's prices ({cur})   total {tot['total']:.4f}   per reply {tot['total']/n:.5f}")
print("      " + "   ".join(f"{k} {tot[k]:.4f}" for k in COST_KEYS))
if len(by_model) > 1:
    for m, t in by_model.items():
        print(f"      {m}: {t['total']:.4f}")
if unpriced:
    print("  UNPRICED (no price in prices.ini, counted as 0): "
          + ", ".join(f"{k} in {v} replies" for k, v in unpriced.items()) + "   -> python price_check.py --edit")
print("per-reply costs written to usage_costs.csv")

if os.path.exists(args.cost_log):
    with open(args.cost_log, newline="", encoding="utf-8-sig") as fh:
        sessions = [r for r in csv.DictReader(fh) if r.get("kind") == "session" and r.get("time", "") >= args.since]
    if sessions:
        print(f"\nSessions in {args.cost_log} (prices valid at that time):")
        print(f"  {'session':<17} {'replies':>7} {'total':>12}  {'llm':>10} {'tts':>10} {'stt':>10} {'search':>10}")
        for s in sessions:
            print(f"  {s['session_id']:<17} {s['replies']:>7} {num(s['cost_total']):>8.4f} {s['currency']:<3}"
                  f"  {num(s['cost_llm']):>10.4f} {num(s['cost_tts']):>10.4f} {num(s['cost_stt']):>10.4f} {num(s['cost_search']):>10.4f}")
        print(f"  {'ALL':<17} {sum(num(s['replies']) for s in sessions):>7.0f} {sum(num(s['cost_total']) for s in sessions):>8.4f}")

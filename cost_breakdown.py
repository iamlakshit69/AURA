#!/usr/bin/env python3
"""
Where does the money go?   python cost_breakdown.py [cost_log.csv] [--since 2026-11-01] [--last N]

Reads cost_log.csv (written live by the agent, prices valid at the time) and prints:
  1. the total split into LLM / TTS / STT / search, with percentages
  2. every session: replies, total, each component, and the "overhead" (filler clips rendered at startup
     + turns that never finished), which is session total minus the sum of its replies
  3. the average reply, and the 5 most expensive replies
Also written to cost_breakdown.txt, so you do not need to copy from the terminal.
"""
import argparse
import csv
import io
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?", default=os.getenv("COST_CSV", "cost_log.csv"))
ap.add_argument("--since", default="", help="only rows from this date on, e.g. 2026-11-01")
ap.add_argument("--last", type=int, default=0, help="only the last N sessions")
args = ap.parse_args()

PARTS = ["llm", "tts", "stt", "search"]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


with open(args.log, newline="", encoding="utf-8-sig") as fh:
    rows = [r for r in csv.DictReader(fh) if r.get("time", "") >= args.since]
sessions = [r for r in rows if r.get("kind") == "session"]
replies = [r for r in rows if r.get("kind") == "reply"]
if args.last:
    keep = {s["session_id"] for s in sessions[-args.last:]}
    sessions = [s for s in sessions if s["session_id"] in keep]
    replies = [r for r in replies if r["session_id"] in keep]
if not sessions:
    raise SystemExit("no finished sessions in the log yet (a session row is written when the agent shuts down)")

out = io.StringIO()


def p(*a):
    print(*a, file=out)


def bar(share, width=24):
    n = round(share * width)
    return "#" * n + "." * (width - n)


by_cur = defaultdict(list)
for s in sessions:
    by_cur[s.get("currency") or "?"].append(s)

for cur, ss in by_cur.items():
    ids = {s["session_id"] for s in ss}
    tot = {k: sum(f(s["cost_" + k]) for s in ss) for k in PARTS}
    total = sum(f(s["cost_total"]) for s in ss)
    p(f"=== {len(ss)} session(s), {cur} ===")
    p(f"TOTAL {total:.2f} {cur}\n")
    p(f"{'part':<8}{'cost':>10}  {'share':>6}")
    for k in sorted(PARTS, key=lambda k: -tot[k]):
        share = tot[k] / total if total else 0.0
        p(f"{k.upper():<8}{tot[k]:>10.3f}  {share*100:>5.1f}%  {bar(share)}")
    mine = [r for r in replies if r["session_id"] in ids]
    n = len(mine)
    in_replies = sum(f(r["cost_total"]) for r in mine)
    over = total - in_replies
    p(f"\n{n} replies cost {in_replies:.2f}; overhead {over:.2f} ({(over/total*100 if total else 0):.0f}% of total): "
      f"filler clips rendered at every start + turns that never finished")
    if n:
        p(f"average reply {in_replies/n:.3f} {cur}: " +
          ", ".join(f"{k} {sum(f(r['cost_'+k]) for r in mine)/n:.3f}" for k in PARTS))

    p(f"\n{'session':<17}{'stt model':<22}{'replies':>7}{'total':>8}" + "".join(f"{k:>8}" for k in PARTS) + f"{'overhead':>9}")
    for s in ss:
        rs = sum(f(r["cost_total"]) for r in replies if r["session_id"] == s["session_id"])
        p(f"{s['session_id']:<17}{(s.get('stt_model') or '?')[:21]:<22}{s['replies']:>7}{f(s['cost_total']):>8.2f}"
          + "".join(f"{f(s['cost_'+k]):>8.2f}" for k in PARTS) + f"{f(s['cost_total'])-rs:>9.2f}")

    top = sorted(mine, key=lambda r: -f(r["cost_total"]))[:5]
    if top:
        p("\nmost expensive replies:")
        for r in top:
            parts = {k: f(r["cost_" + k]) for k in PARTS}
            big = max(parts, key=parts.get)
            p(f"  {r['time']}  {f(r['cost_total']):.3f}  mostly {big.upper()} ({parts[big]:.3f})  "
              f"[{r['reply_chars']} chars, {r['llm_prompt_tokens']} prompt tok, {f(r['stt_audio_s']):.0f} s STT, "
              f"tavily {f(r['tavily_credits']):.0f} exa {r['exa_requests']}]")
    unp = {s["unpriced"] for s in ss if s.get("unpriced")}
    if unp:
        p(f"\nUNPRICED (counted as 0): {', '.join(sorted(unp))}   -> python price_check.py")
    p()

text = out.getvalue()
print(text)
with open("cost_breakdown.txt", "w", encoding="utf-8") as fh:
    fh.write(text)
print("also written to cost_breakdown.txt")

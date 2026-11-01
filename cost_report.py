#!/usr/bin/env python3
"""
Turn usage_log.csv into money.   python cost_report.py [usage_log.csv]

Fill in PRICES from each provider's pricing page (they change, so they are not hard-coded in the
agent). All prices in the same currency. Writes usage_costs.csv with a cost column per reply.
"""
import csv
import sys

PRICES = {
    "llm_input_per_m":      0.0,   # per 1M prompt tokens (non-cached)
    "llm_cached_input_per_m": 0.0, # per 1M cached prompt tokens (often a discount; same as input if none)
    "llm_output_per_m":     0.0,   # per 1M completion tokens
    "tts_per_10k_chars":    0.0,   # Sarvam Bulbul, per 10,000 characters
    "stt_per_hour":         0.0,   # Sarvam STT, per hour of audio
    "tavily_per_credit":    0.0,
    "exa_per_request":      0.0,
}

path = sys.argv[1] if len(sys.argv) > 1 else "usage_log.csv"
rows = list(csv.DictReader(open(path, newline="")))
if not rows:
    raise SystemExit("no rows yet")
if not any(PRICES.values()):
    print("NOTE: PRICES are all 0, so only usage is shown. Fill them in at the top of this file.\n")

def f(r, k): return float(r[k] or 0)

def cost(r):
    cached = f(r, "llm_cached_tokens")
    fresh = max(f(r, "llm_prompt_tokens") - cached, 0)
    parts = {
        "llm":    (fresh * PRICES["llm_input_per_m"] + cached * PRICES["llm_cached_input_per_m"]
                   + f(r, "llm_completion_tokens") * PRICES["llm_output_per_m"]) / 1e6,
        "tts":    f(r, "tts_chars") / 1e4 * PRICES["tts_per_10k_chars"],
        "stt":    f(r, "stt_audio_s") / 3600 * PRICES["stt_per_hour"],
        "search": f(r, "tavily_credits") * PRICES["tavily_per_credit"] + f(r, "exa_requests") * PRICES["exa_per_request"],
    }
    parts["total"] = sum(parts.values())
    return parts

out, tot = [], {"llm": 0.0, "tts": 0.0, "stt": 0.0, "search": 0.0, "total": 0.0}
for r in rows:
    c = cost(r)
    for k in tot: tot[k] += c[k]
    out.append({**r, **{f"cost_{k}": round(v, 6) for k, v in c.items()}})
with open("usage_costs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

n = len(rows)
sums = {k: sum(f(r, k) for r in rows) for k in rows[0] if k not in ("time", "llm")}
print(f"{n} replies")
print(f"  LLM   {sums['llm_prompt_tokens']:.0f} in ({sums['llm_cached_tokens']:.0f} cached) / {sums['llm_completion_tokens']:.0f} out tokens"
      f"  -> {sums['llm_prompt_tokens']/n:.0f} in, {sums['llm_completion_tokens']/n:.0f} out per reply")
print(f"  TTS   {sums['tts_chars']:.0f} chars  -> {sums['tts_chars']/n:.0f} per reply")
print(f"  STT   {sums['stt_audio_s']:.1f} s audio  -> {sums['stt_audio_s']/n:.1f} s per reply (average only: per-row STT is lumpy)")
if sums.get("tts_estimated"):
    print(f"  note: {sums['tts_estimated']:.0f} reply(ies) had no TTS metrics, their tts_chars is the reply length")
print(f"  Search tavily {sums['tavily_credits']:.0f} credits / exa {sums['exa_requests']:.0f} requests")
print(f"\nCost  total {tot['total']:.4f}   per reply {tot['total']/n:.5f}")
print("      " + "  ".join(f"{k} {v:.4f}" for k, v in tot.items() if k != "total"))
print("\nper-reply costs written to usage_costs.csv")

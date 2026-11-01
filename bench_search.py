#!/usr/bin/env python3
"""
Measure search latency from YOUR machine, which is the only number that matters for your agent.

    python bench_search.py                      # every configured provider, 3 rounds
    python bench_search.py --rounds 5
    python bench_search.py --depths ultra-fast,fast,basic     # compare Tavily depths

Uses TAVILY_API_KEY / EXA_API_KEY from .env. Caching is off, so every call hits the network.
Each query is sent one at a time. The first round also pays connection setup, so it is reported
separately from the warm rounds.
"""
import argparse
import asyncio
import statistics
import time

from dotenv import load_dotenv
import os

from search_providers import Search

load_dotenv()
load_dotenv(".env.local")

QUERIES = [
    "latest news India today",
    "bitcoin price now",
    "who won the last cricket match",
    "ESP32-S3 specific libraries",
    "petrol price Punjab",
    "iPhone latest model release",
    "Sensex today",
    "Python 3.14 release notes",
]


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


async def run_config(label, search, rounds):
    cold, warm, errors = [], [], 0
    for r in range(rounds):
        for q in QUERIES:
            t0 = time.perf_counter()
            text, _ = await search._hedged(q)          # bypass the cache on purpose
            ms = (time.perf_counter() - t0) * 1000
            if not text:
                errors += 1
                continue
            (cold if r == 0 else warm).append(ms)
    both = cold + warm
    print(f"{label:<22} n={len(both):<3} err={errors:<2} "
          f"p50={pct(both, 50):6.0f}  p95={pct(both, 95):6.0f}  mean={statistics.mean(both) if both else float('nan'):6.0f} ms"
          f"   (round 1 mean {statistics.mean(cold) if cold else float('nan'):.0f}, "
          f"later rounds mean {statistics.mean(warm) if warm else float('nan'):.0f})")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=6.0, help="per-search cap while benchmarking")
    ap.add_argument("--depths", default="ultra-fast,fast,basic", help="Tavily depths to compare")
    args = ap.parse_args()

    tavily = os.getenv("TAVILY_API_KEY", "").strip()
    exa = os.getenv("EXA_API_KEY", "").strip()
    if not (tavily or exa):
        raise SystemExit("Set TAVILY_API_KEY and/or EXA_API_KEY in .env first.")

    configs = []
    if tavily:
        for depth in [d.strip() for d in args.depths.split(",") if d.strip()]:
            configs.append((f"tavily/{depth}", dict(tavily_key=tavily, providers=["tavily"], tavily_depth=depth)))
    if exa:
        configs.append(("exa/instant", dict(exa_key=exa, providers=["exa"])))
    if tavily and exa:
        configs.append(("tavily+exa (hedged)", dict(tavily_key=tavily, exa_key=exa, tavily_depth="ultra-fast")))

    print(f"{len(QUERIES)} queries x {args.rounds} rounds per config, one at a time, cache off\n")
    for label, kw in configs:
        s = Search(timeout=args.timeout, use_cache=False, **kw)
        await s.warm()
        await run_config(label, s, args.rounds)
        await s.aclose()


if __name__ == "__main__":
    asyncio.run(main())

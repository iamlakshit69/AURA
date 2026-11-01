"""
Search layer for the LiveKit voice agent. No LiveKit imports, so it can be tested and
benchmarked on its own (see bench_search.py).

What lives here:
    Search      Tavily and/or Exa behind one call. Hard timeout, hedging across providers,
                short-TTL cache, de-duplication of identical in-flight queries, connection
                warm-up and keep-alive, readable error logging.
    Prefetcher  Starts a search while the user is still talking (partial transcripts) or the
                moment the final transcript lands, so the tool call only has to wait for a
                search that is already running.
    Weather     Open-Meteo (no API key). Direct API instead of a general web search.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

logger = logging.getLogger("voice-agent.search")

TAVILY_URL = "https://api.tavily.com/search"
EXA_URL = "https://api.exa.ai/search"
WARM_URLS = {"tavily": "https://api.tavily.com/", "exa": "https://api.exa.ai/"}


# ---------------------------------------------------------------- helpers
_SPLIT = re.compile(r"[\s\.,;:!?\"'()\[\]{}|\u0964\-]+")


def tokens(text: str) -> list[str]:
    """Lower-case words. Splits on whitespace and punctuation only, so Devanagari and
    Gurmukhi words (which contain combining marks) stay in one piece."""
    return [t for t in _SPLIT.split(text.lower()) if t]


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def describe(e: BaseException) -> str:
    """str(ReadTimeout()) is empty, which is why the old log line said just 'search failed:'."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code} {e.response.text[:200]!r}"
    return f"{type(e).__name__}({e})"


_PRICE = re.compile(r"\b(price|prices|stock|stocks|rate|rates|bitcoin|nifty|sensex|gold|silver|"
                    r"petrol|diesel|dollar|rupee)\b", re.I)
_FAST = re.compile(r"\b(news|score|scores|live|breaking|headlines?|match|won|winner|result|results)\b", re.I)


def ttl_for(query: str) -> float:
    """How long a cached answer stays usable. Prices go stale fastest, then news and scores."""
    if _PRICE.search(query):
        return 60.0
    if _FAST.search(query):
        return 90.0
    return 300.0


class TTLCache:
    def __init__(self, max_items: int = 128) -> None:
        self._d: dict[str, tuple[float, str]] = {}
        self._max = max_items

    def get(self, key: str) -> Optional[str]:
        item = self._d.get(key)
        if not item:
            return None
        if item[0] < time.monotonic():
            self._d.pop(key, None)
            return None
        return item[1]

    def set(self, key: str, value: str, ttl: float) -> None:
        if len(self._d) >= self._max:
            self._d.pop(next(iter(self._d)))          # oldest insertion
        self._d[key] = (time.monotonic() + ttl, value)


# ---------------------------------------------------------------- search
class Search:
    """query(text) -> (answer_text or None, source). Never raises, never waits longer than
    `timeout` (plus a hair), so the voice loop always gets control back."""

    def __init__(self, *, tavily_key: str = "", exa_key: str = "", providers: Optional[list[str]] = None,
                 timeout: float = 3.0, max_results: int = 2, snippet_chars: int = 200,
                 tavily_depth: str = "ultra-fast", hedge_same_s: float = 0.0,
                 use_cache: bool = True, http: Optional[httpx.AsyncClient] = None,
                 on_call: Optional[Callable[[str, float], None]] = None) -> None:
        self.on_call = on_call            # on_call(provider, credits) for every billable request sent
        self.tavily_key = tavily_key
        self.exa_key = exa_key
        names = [n for n, k in (("tavily", tavily_key), ("exa", exa_key)) if k]
        self.provider_names = [n for n in names if providers is None or n in providers]
        self.timeout = timeout
        self.max_results = max_results
        self.snippet_chars = snippet_chars
        self.tavily_depth = tavily_depth
        self.hedge_same_s = hedge_same_s
        self.use_cache = use_cache
        self.cache = TTLCache()
        self._inflight: dict[str, asyncio.Task] = {}
        self._waiters: dict[str, int] = {}
        self._losers: set[asyncio.Task] = set()   # hedge losers left to finish (keeps their connection pooled)
        self._own_http = http is None
        # keepalive_expiry: httpx closes idle connections after 5 s by default, which would
        # throw away the warm-up almost immediately.
        self.http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(1.5, timeout)),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=120.0),
        )

    @property
    def configured(self) -> bool:
        return bool(self.provider_names)

    async def aclose(self) -> None:
        for t in list(self._inflight.values()) + list(self._losers):
            t.cancel()
        if self._own_http:
            await self.http.aclose()

    # --- warm-up: open TLS connections before the first real search ---
    async def warm(self, extra_urls: tuple[str, ...] | list[str] = ()) -> None:
        urls = [WARM_URLS[n] for n in self.provider_names] + list(extra_urls)

        async def one(url: str) -> None:
            t0 = time.perf_counter()
            try:
                await self.http.get(url, timeout=2.0)     # status does not matter, the connection does
                logger.debug("warm %s %d ms", url, round((time.perf_counter() - t0) * 1000))
            except Exception as e:
                logger.debug("warm %s failed: %s", url, describe(e))

        await asyncio.gather(*(one(u) for u in urls))

    async def keepalive_loop(self, interval: float = 40.0, extra_urls: tuple[str, ...] | list[str] = ()) -> None:
        """Re-touch each host now and then so servers do not close the idle connections."""
        while True:
            await asyncio.sleep(interval)
            await self.warm(extra_urls)

    def _bill(self, provider: str, credits: float) -> None:
        if self.on_call:
            try:
                self.on_call(provider, credits)
            except Exception:
                pass

    # --- providers: each returns text or None, never raises ---
    async def _tavily(self, q: str) -> Optional[str]:
        t0 = time.perf_counter()
        data = None
        for attempt in (1, 2):
            try:
                self._bill("tavily", 2.0 if self.tavily_depth == "advanced" else 1.0)
                r = await self.http.post(
                    TAVILY_URL,
                    headers={"Authorization": f"Bearer {self.tavily_key}"},
                    json={"query": q, "max_results": self.max_results, "search_depth": self.tavily_depth,
                          "include_answer": False},     # the answer step is an extra LLM call on their side
                )
                r.raise_for_status()
                data = r.json()
                break
            except httpx.HTTPStatusError as e:
                body = e.response.text.lower()
                if (attempt == 1 and e.response.status_code in (400, 422)
                        and "depth" in body and self.tavily_depth != "basic"):
                    logger.warning("tavily rejected search_depth=%r (%s); falling back to 'basic'",
                                   self.tavily_depth, describe(e))
                    self.tavily_depth = "basic"
                    continue
                logger.warning("search[tavily] failed after %d ms: %s",
                               round((time.perf_counter() - t0) * 1000), describe(e))
                return None
            except Exception as e:
                logger.warning("search[tavily] failed after %d ms: %s",
                               round((time.perf_counter() - t0) * 1000), describe(e))
                return None
        if data is None:
            return None
        parts = [(x.get("content") or "").strip()[: self.snippet_chars]
                 for x in data.get("results", [])[: self.max_results]]
        return "\n".join(p for p in parts if p) or None

    async def _exa(self, q: str) -> Optional[str]:
        t0 = time.perf_counter()
        try:
            self._bill("exa", 1.0)
            r = await self.http.post(
                EXA_URL,
                headers={"x-api-key": self.exa_key},
                json={"query": q, "type": "instant", "numResults": self.max_results,
                      "contents": {"text": {"maxCharacters": self.snippet_chars}}},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("search[exa] failed after %d ms: %s", round((time.perf_counter() - t0) * 1000), describe(e))
            return None
        parts = []
        for x in data.get("results", [])[: self.max_results]:
            title = (x.get("title") or "").strip()[:60]
            text = (x.get("text") or "").strip()[: self.snippet_chars]
            line = f"{title}. {text}".strip(". ").strip() if title and text else (text or title)
            if line:
                parts.append(line)
        return "\n".join(parts) or None

    # --- hedging: ask every configured provider at once, keep the first usable answer ---
    async def _hedged(self, q: str, only: Optional[str] = None) -> tuple[Optional[str], str]:
        fns = []
        if "tavily" in self.provider_names and only in (None, "tavily"):
            fns.append(("tavily", self._tavily))
        if "exa" in self.provider_names and only in (None, "exa"):
            fns.append(("exa", self._exa))
        if not fns:
            return None, ""
        tasks: dict[asyncio.Task, str] = {asyncio.create_task(fn(q)): name for name, fn in fns}
        if len(fns) == 1 and self.hedge_same_s > 0:
            name, fn = fns[0]

            async def later() -> Optional[str]:
                await asyncio.sleep(self.hedge_same_s)
                return await fn(q)

            tasks[asyncio.create_task(later())] = f"{name}+hedge"
        pending = set(tasks)
        deadline = time.monotonic() + self.timeout
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, pending = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    break
                for t in done:
                    if t.cancelled() or t.exception() is not None:
                        continue
                    res = t.result()
                    if res:
                        return res, tasks[t]
        finally:
            # Do NOT cancel the slower request: it is already billed, and cancelling it mid-flight makes
            # httpx throw away that connection, so the next search would pay a new TLS handshake. Providers
            # never raise and httpx's own timeout bounds them, so they finish on their own.
            for t in pending:
                self._losers.add(t)
                t.add_done_callback(self._loser_done)
        logger.warning("search: no usable result within %.1f s for %r", self.timeout, q[:80])
        return None, ""

    def _loser_done(self, t: asyncio.Task) -> None:
        self._losers.discard(t)
        if not t.cancelled():
            t.exception()                      # mark retrieved, no 'never retrieved' warnings

    async def _run(self, q: str, key: str, only: Optional[str] = None) -> tuple[Optional[str], str]:
        t0 = time.perf_counter()
        text, source = await self._hedged(q, only)
        logger.info("search[%s] %d ms %s", source or "none", round((time.perf_counter() - t0) * 1000),
                    "ok" if text else "FAILED")
        if text and self.use_cache:
            self.cache.set(key, text, ttl_for(q))
        return text, source

    async def query(self, q: str, *, speculative: bool = False) -> tuple[Optional[str], str]:
        """speculative=True: a guess made while the user is still talking. It asks ONE provider only
        (half the cost) and, if nobody is waiting for it any more, is cancelled instead of running on."""
        q = " ".join(q.split())[:200]
        key = normalize(q)
        if self.use_cache:
            hit = self.cache.get(key)
            if hit:
                return hit, "cache"
        task = self._inflight.get(key)
        if task is None:
            only = self.provider_names[0] if (speculative and self.provider_names) else None
            task = asyncio.create_task(self._run(q, key, only))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            # shield: if a real caller is cancelled (user barged in), the search still finishes and is cached
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if speculative and self._waiters.get(key, 0) <= 1 and not task.done():
                task.cancel()                   # a dropped guess nobody else wants: stop it
            raise
        finally:
            n = self._waiters.get(key, 1) - 1
            if n <= 0:
                self._waiters.pop(key, None)
            else:
                self._waiters[key] = n


# ---------------------------------------------------------------- prefetch
_INTENT_LATIN = re.compile(
    r"\b(news|headlines?|latest|breaking|score|scores|price|prices|stock|stocks|share price|sensex|nifty|"
    r"bitcoin|gold|silver|petrol|diesel|trending|who won|(?:match|game|cricket)\s+(?:score|result)|"
    r"exchange rate|interest rate|khabar|samachar|taaza|kimat|bhav)\b", re.I)
# Devanagari and Gurmukhi words contain combining marks, so \b does not work: plain substring match.
_INTENT_INDIC = ("समाचार", "खबर", "ख़बर", "ताज़ा", "ताजा", "कीमत", "भाव", "स्कोर", "मैच", "नतीजा",
                 "ਖ਼ਬਰ", "ਖਬਰ", "ਤਾਜ਼ਾ", "ਕੀਮਤ", "ਭਾਅ", "ਸਕੋਰ", "ਮੈਚ")
_WEATHER = re.compile(r"\b(weather|temperature|rain|mausam|barish)\b|मौसम|ਮੌਸਮ", re.I)


def looks_like_search(text: str, mode: str = "intent") -> bool:
    """Cheap check: is this obviously a question that needs the live web?"""
    if mode == "off" or _WEATHER.search(text):            # weather has its own tool
        return False
    if mode == "all":
        return len(tokens(text)) >= 3
    return bool(_INTENT_LATIN.search(text)) or any(w in text for w in _INTENT_INDIC)


@dataclass
class _Entry:
    text: str
    toks: list[str]
    task: asyncio.Task
    t0: float
    final: bool


class Prefetcher:
    """Runs a search before the model asks for it.

    on_partial(text)  speculative: at most `max_spec` searches per utterance, and only when the
                      partial transcript already looks like a search question.
    on_final(text)    reuse the speculative search if the final text is nearly the same,
                      otherwise cancel it and search the final text right away.
    take()            called by the web_search tool: returns the running search if there is one.
    """

    MAX_AGE_S = 30.0

    def __init__(self, search: Search, mode: str = "intent", max_spec: int = 1) -> None:
        self.search = search
        self.mode = mode
        self.max_spec = max_spec
        self.budget = max_spec
        self.entry: Optional[_Entry] = None

    def _start(self, text: str, final: bool) -> None:
        text = " ".join(text.split())[:200]
        task = asyncio.create_task(self.search.query(text, speculative=not final))
        self.entry = _Entry(text, tokens(text), task, time.monotonic(), final)
        logger.debug("prefetch start (%s): %r", "final" if final else "speculative", text)

    def _drop(self) -> None:
        e, self.entry = self.entry, None
        if e and not e.task.done():
            e.task.cancel()

    def on_partial(self, text: str) -> None:
        if self.mode != "intent" or self.budget <= 0:
            return
        toks = tokens(text)
        if len(toks) < 4 or not looks_like_search(text, self.mode):
            return
        e = self.entry
        if e and (time.monotonic() - e.t0 < 1.0 or jaccard(e.toks, toks) >= 0.75):
            return
        self._drop()
        self._start(text, final=False)
        self.budget -= 1

    def on_final(self, text: str) -> None:
        if self.mode == "off":
            return
        self.budget = self.max_spec
        if not looks_like_search(text, self.mode):
            self._drop()
            return
        toks = tokens(text)
        e = self.entry
        if e and jaccard(e.toks, toks) >= 0.75:           # the speculative search is good enough
            e.final = True
            return
        self._drop()
        self._start(text, final=True)

    async def take(self) -> tuple[bool, Optional[str], str]:
        """-> (a prefetch existed, its text or None if it failed, source label)."""
        e, self.entry = self.entry, None
        self.budget = self.max_spec
        if e is None or time.monotonic() - e.t0 > self.MAX_AGE_S:
            if e and not e.task.done():
                e.task.cancel()
            return False, None, ""
        try:
            text, source = await e.task
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            logger.warning("prefetch failed: %s", describe(ex))
            return True, None, "prefetch"
        return True, text, f"prefetch:{source}" if source else "prefetch"


# ---------------------------------------------------------------- weather
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers", 85: "snow showers",
    86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


class Weather:
    """Open-Meteo: free, no key. One cached geocode lookup per city, one forecast call per 10 min."""

    HOSTS = ("https://geocoding-api.open-meteo.com/", "https://api.open-meteo.com/")
    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, http: httpx.AsyncClient, timeout: float = 3.0, ttl: float = 600.0) -> None:
        self.http = http
        self.timeout = timeout
        self.ttl = ttl
        self._geo: dict[str, tuple[float, float, str]] = {}
        self._cache = TTLCache(32)

    async def geocode(self, city: str) -> Optional[tuple[float, float, str]]:
        key = normalize(city)
        if key in self._geo:
            return self._geo[key]
        r = await self.http.get(self.GEO_URL, params={"name": city, "count": 1, "language": "en", "format": "json"})
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return None
        top = results[0]
        geo = (top["latitude"], top["longitude"], top.get("name") or city)
        self._geo[key] = geo
        return geo

    async def _fetch(self, city: str) -> Optional[str]:
        geo = await self.geocode(city)
        if not geo:
            return None
        lat, lon, label = geo
        key = f"{lat:.2f},{lon:.2f}"
        hit = self._cache.get(key)
        if hit:
            return hit
        r = await self.http.get(self.FORECAST_URL, params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 1,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        })
        r.raise_for_status()
        data = r.json()
        cur, day = data.get("current") or {}, data.get("daily") or {}

        def num(v, unit: str = "") -> str:
            return f"{round(v)}{unit}" if isinstance(v, (int, float)) else "unknown"

        def first(name: str):
            vals = day.get(name) or [None]
            return vals[0]

        text = (f"{label}: {num(cur.get('temperature_2m'))} degrees Celsius, feels like "
                f"{num(cur.get('apparent_temperature'))}, {_WMO.get(cur.get('weather_code'), 'unclear conditions')}, "
                f"humidity {num(cur.get('relative_humidity_2m'))} percent, wind {num(cur.get('wind_speed_10m'))} km/h. "
                f"Today high {num(first('temperature_2m_max'))}, low {num(first('temperature_2m_min'))}, "
                f"chance of rain {num(first('precipitation_probability_max'))} percent.")
        self._cache.set(key, text, self.ttl)
        return text

    async def get(self, city: str) -> Optional[str]:
        """Weather as a short English sentence for the model to restate, or None on any failure."""
        t0 = time.perf_counter()
        try:
            return await asyncio.wait_for(self._fetch(city), timeout=self.timeout)
        except Exception as e:
            logger.warning("weather(%r) failed after %d ms: %s", city, round((time.perf_counter() - t0) * 1000),
                           describe(e))
            return None

#!/usr/bin/env python3
"""
Try the wake word on its own, without the agent. Use it to pick WAKE_THRESHOLD for YOUR voice and room.

    python test_wakeword.py                         # live microphone, model + threshold from settings.env
    python test_wakeword.py --model hey_jarvis      # a pretrained model, to check the install first
    python test_wakeword.py --model anima.onnx --threshold 0.6
    python test_wakeword.py --file recording.wav    # a recording instead of the microphone

Live mode needs:  pip install sounddevice      (macOS: nothing else; Linux: sudo apt install libportaudio2)

How to read it: say "Anima" a few times, then talk normally for a minute (and play some TV or music).
You want the wake word to reach 0.8+ and everything else to stay under about 0.3. Put the threshold between them.
Ctrl+C prints the highest score seen.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import time
import wave

import numpy as np
from dotenv import load_dotenv

from settings import load_settings

load_settings()
load_dotenv()

from wakeword import CHUNK, OWW_RATE, load_model  # noqa: E402  (after the settings are loaded)


def bar(score: float, threshold: float, width: int = 40) -> str:
    n, t = int(min(score, 1.0) * width), int(threshold * width)
    cells = ["█" if i < n else "·" for i in range(width)]
    cells[min(t, width - 1)] = "|" if n <= t else "█"
    return "".join(cells)


def score_of(oww, chunk: np.ndarray) -> float:
    return max(oww.predict(chunk).values(), default=0.0)


def run_file(oww, path: str, threshold: float) -> None:
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise SystemExit("need a 16-bit wav file")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if rate != OWW_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(rate, OWW_RATE)
        pcm = resample_poly(pcm, OWW_RATE // g, rate // g).astype(np.int16)
    peak, hits = 0.0, 0
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        s = score_of(oww, pcm[i:i + CHUNK])
        peak = max(peak, s)
        if s >= threshold:
            hits += 1
            print(f"{i / OWW_RATE:6.2f} s  score {s:.2f}  <-- would wake")
    print(f"\npeak score {peak:.2f}, {hits} chunks above {threshold:.2f}")


def run_live(oww, threshold: float) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("live mode needs:  pip install sounddevice   (or use --file)")
    q: queue.Queue = queue.Queue()
    peak, last_hit = 0.0, 0.0

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    print(f"Listening (threshold {threshold:.2f}). Say the wake word. Ctrl+C to stop.\n")
    try:
        with sd.InputStream(samplerate=OWW_RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=cb):
            while True:
                s = score_of(oww, q.get())
                peak = max(peak, s)
                line = f"\r{bar(s, threshold)} {s:4.2f}  peak {peak:4.2f}"
                if s >= threshold and time.time() - last_hit > 1.5:
                    last_hit = time.time()
                    line += "   <-- WAKE\n"
                sys.stdout.write(line + "   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print(f"\n\nhighest score seen: {peak:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("WAKE_MODEL", "anima.onnx"))
    ap.add_argument("--threshold", type=float, default=float(os.getenv("WAKE_THRESHOLD", "0.5")))
    ap.add_argument("--vad", type=float, default=0.5, help="built-in Silero VAD gate (0 = off)")
    ap.add_argument("--file", help="a 16-bit wav recording to scan instead of the microphone")
    a = ap.parse_args()
    oww, label = load_model(a.model, a.vad)
    print(f"model: {label}")
    run_file(oww, a.file, a.threshold) if a.file else run_live(oww, a.threshold)


if __name__ == "__main__":
    main()

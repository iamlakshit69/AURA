"""
Wake word + stop word gate for the voice agent (openWakeWord).

    ASLEEP      mic audio goes ONLY to the wake-word detector. Nothing reaches STT, so nothing is billed.
    LISTENING   audio flows to STT (you are talking / about to talk).
    RESPONDING  the agent is thinking or speaking: STT audio is OFF, only the stop-word detector listens.

    say the wake word       -> LISTENING (the last ~1.2 s are replayed into STT, so "Anima, what's the weather" is not clipped)
    you finish, agent replies -> RESPONDING (STT off: nothing billed, the agent cannot transcribe its own voice)
    say the stop word       -> reply is cut off (session.interrupt), back to LISTENING. "stop" itself is NOT sent to STT.
    reply ends              -> LISTENING, with a WAKE_IDLE_S window. No speech in it: ASLEEP again.
    wake word, then silence -> ASLEEP after WAKE_GRACE_S.

The wake word is NOT sent to STT (so it never reaches the LLM): WAKE_PREROLL_S=0 means STT only gets audio from the
moment of detection on. A short chime plays after the wake word and after the stop word (WAKE_CHIME=0 to turn off).
It is played locally with sounddevice, not through the agent's speech pipeline, so it never counts as a reply.
STT audio is held back for the length of the chime so the chime itself is not transcribed.

STT_MUTE_WHILE_REPLYING=0 turns RESPONDING off: STT stays on during replies (the old behaviour).

The gate sits in Agent.stt_node. Voice barge-in is gone while STT is muted, so the session must run with
interruptions disabled and the stop word calls session.interrupt(force=True).

Models: WAKE_MODEL / STOP_MODEL are a path to a model you trained (anima.onnx, stop.onnx) or the name of a
pretrained one (hey_jarvis, alexa, hey_mycroft, ...). If STOP_MODEL is empty, the wake model is used for both.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
import time
from typing import AsyncIterable, Callable, Optional

import numpy as np
from livekit import rtc

log = logging.getLogger("voice-agent.wake")

OWW_RATE = 16000          # openWakeWord only understands 16 kHz mono int16
CHUNK = 1280              # 80 ms: the chunk size openWakeWord is built for


CHIME_RATE = 24000


def _tone(freqs: list, seconds: float, volume: float) -> np.ndarray:
    """A short soft beep sequence (one note per frequency) with a quick fade in/out, as int16."""
    n = int(CHIME_RATE * seconds / len(freqs))
    t = np.arange(n) / CHIME_RATE
    env = np.minimum(1.0, np.minimum(t / 0.008, (t[-1] - t) / 0.03))
    wave = np.concatenate([np.sin(2 * np.pi * f * t) * env for f in freqs])
    return (wave * volume * 32767).astype(np.int16)


def load_model(model: str, vad_threshold: float = 0.5):
    """Load an openWakeWord model from a file path or a pretrained name. Downloads the shared feature models
    (melspectrogram, embedding, silero VAD) on first use. Returns (Model, label)."""
    import openwakeword
    from openwakeword import utils
    from openwakeword.model import Model

    is_file = os.path.isfile(model)
    name = model if is_file else os.path.basename(model).replace(".onnx", "")
    try:    # first run: fetch the small shared models (a few MB). "__none__" = no pretrained model needed.
        utils.download_models(["__none__"] if is_file else [name])
    except Exception as e:
        log.warning("openWakeWord model download failed (%r); continuing with what is already on disk", e)
    if not is_file and name not in openwakeword.MODELS:
        raise FileNotFoundError(
            f"wake model '{model}' is neither a file nor a pretrained name "
            f"({', '.join(openwakeword.MODELS)}). Put your trained model next to the agent or set WAKE_MODEL / STOP_MODEL.")
    oww = Model(wakeword_models=[name], inference_framework="onnx", vad_threshold=vad_threshold)
    return oww, os.path.basename(model)


class WakeGate:
    def __init__(self, model: str = "anima.onnx", *, name: str = "Anima", threshold: float = 0.5,
                 idle_s: float = 5.0, grace_s: float = 8.0, preroll_s: float = 1.2, hits: int = 2,
                 vad_threshold: float = 0.5, on_change: Optional[Callable[[bool, str], None]] = None,
                 mute_while_replying: bool = True, stop_model: Optional[str] = None,
                 stop_threshold: Optional[float] = None, stop_hits: Optional[int] = None,
                 on_billed: Optional[Callable[[float], None]] = None, chime: bool = False,
                 chime_volume: float = 0.25, chime_hold_s: Optional[float] = None,
                 stop_raw_mic: bool = False) -> None:
        self.oww, self.label = load_model(model, vad_threshold)
        self.name = name
        self.threshold, self.idle_s, self.grace_s, self.preroll_s = threshold, idle_s, grace_s, preroll_s
        self.hits_needed = max(1, hits)               # consecutive 80 ms chunks above the threshold
        self.mute_while_replying = mute_while_replying
        self.stop_threshold = stop_threshold if stop_threshold is not None else threshold
        self.stop_hits_needed = max(1, stop_hits if stop_hits is not None else hits)
        self.stop_label = self.label
        self.stop_oww = self.oww                      # same model for both words unless STOP_MODEL is set
        if mute_while_replying and stop_model and stop_model != model:
            self.stop_oww, self.stop_label = load_model(stop_model, vad_threshold)
        self.on_change = on_change
        self._sd = None
        self._chimes: dict = {}
        self._hold_s = 0.0
        self._hold_until = 0.0                        # STT audio is dropped until then (the chime itself)
        if chime:
            try:
                import sounddevice as sd
                self._sd = sd
                self._chimes = {"wake": _tone([784, 1175], 0.17, chime_volume),    # rising: I am listening
                                "stop": _tone([880, 587], 0.15, chime_volume)}     # falling: stopped
                dur = max(len(c) for c in self._chimes.values()) / CHIME_RATE
                self._hold_s = dur + 0.05 if chime_hold_s is None else chime_hold_s
            except Exception as e:
                log.warning("chime off (sounddevice unavailable: %r)", e)
        self.on_billed = on_billed                    # called with the seconds of audio sent to STT
        self.billed_s = 0.0                           # total seconds sent to STT (provider-independent)
        self.awake = False
        self.responding = False
        self.session = None
        self.stop_raw_mic = stop_raw_mic              # stop word listens on its own raw mic stream (skips LiveKit's echo/noise filters)
        self._raw_stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.best = 0.0                               # highest wake score since the last wake (for tuning)
        self.best_stop = 0.0                          # highest stop score during the current reply (for tuning)
        self.frames_seen = 0                          # frames the stop detector looked at during this reply
        self._ring: collections.deque[rtc.AudioFrame] = collections.deque()
        self._ring_s = 0.0
        self._pending = np.zeros(0, np.int16)
        self._resampler: Optional[rtc.AudioResampler] = None
        self._src_key: tuple = ()
        self._streaks = {"wake": 0, "stop": 0}
        self._after_stop = False
        self._timer: Optional[asyncio.TimerHandle] = None
        self._why = ""
        log.info("wake word ready: model %s, threshold %.2f, hits %d, idle %.1f s, grace %.1f s | "
                 "STT muted while replying: %s%s", self.label, threshold, self.hits_needed, idle_s, grace_s,
                 mute_while_replying, f" (stop word: {self.stop_label}, threshold {self.stop_threshold:.2f}, "
                                      f"hits {self.stop_hits_needed})" if mute_while_replying else "")

    # ------------------------------------------------------------------ audio filter (use inside Agent.stt_node)
    def _bill(self, frame: rtc.AudioFrame) -> None:
        self.billed_s += frame.duration
        if self.on_billed:
            try:
                self.on_billed(frame.duration)
            except Exception as e:
                log.warning("on_billed failed: %r", e)

    async def filter(self, audio: AsyncIterable[rtc.AudioFrame]) -> AsyncIterable[rtc.AudioFrame]:
        async for frame in audio:
            if self.awake and not self.responding:       # LISTENING: audio goes to STT
                if time.monotonic() < self._hold_until:  # the chime is playing: do not transcribe it
                    continue
                self._bill(frame)
                yield frame
                continue
            if self.awake:                               # RESPONDING: STT off, only the stop word is listened for
                if self._raw_stream is not None:         # the raw mic stream does the stop detection
                    continue
                self.frames_seen += 1
                try:
                    if self._detect(frame, self.stop_oww, "stop"):
                        self.stop_speaking()
                except Exception as e:                   # a detector problem must never kill the audio path
                    log.warning("stop detection failed: %r", e)
                continue
            # ASLEEP
            self._ring.append(frame)
            self._ring_s += frame.duration
            while self._ring and self._ring_s > self.preroll_s:
                self._ring_s -= self._ring.popleft().duration
            try:
                heard = self._detect(frame, self.oww, "wake")
            except Exception as e:
                log.warning("wake detection failed: %r", e)
                heard = False
            if heard:
                ring = list(self._ring)                  # wake() clears the buffers, so take the pre-roll first
                self.wake("wake word")
                for f in ring:                           # replay the pre-roll (it contains the wake word itself)
                    self._bill(f)
                    yield f

    def _detect(self, frame: rtc.AudioFrame, oww, kind: str) -> bool:
        key = (frame.sample_rate, frame.num_channels)
        if frame.sample_rate == OWW_RATE:
            frames = [frame]
        else:
            if self._resampler is None or self._src_key != key:
                self._resampler = rtc.AudioResampler(frame.sample_rate, OWW_RATE, num_channels=frame.num_channels)
                self._src_key = key
            frames = self._resampler.push(frame)
        if not frames:
            return False
        pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])
        if frame.num_channels > 1:                    # average the channels
            pcm = pcm[: len(pcm) // frame.num_channels * frame.num_channels]
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        self._pending = np.concatenate([self._pending, pcm])
        threshold = self.threshold if kind == "wake" else self.stop_threshold
        needed = self.hits_needed if kind == "wake" else self.stop_hits_needed
        hit = False
        while len(self._pending) >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            score = max(oww.predict(chunk).values(), default=0.0)
            if kind == "wake":
                self.best = max(self.best, score)
            else:
                self.best_stop = max(self.best_stop, score)
            if score >= 0.25:
                log.debug("%s score %.2f", kind, score)
            self._streaks[kind] = self._streaks[kind] + 1 if score >= threshold else 0
            if self._streaks[kind] >= needed:
                hit = True
        return hit

    def _chime(self, kind: str) -> None:
        """Play the chime for 'wake' or 'stop' in a thread (opening the audio device must not block the loop)."""
        if self._sd is None or kind not in self._chimes:
            return
        self._hold_until = time.monotonic() + self._hold_s
        data, sd = self._chimes[kind], self._sd

        def run() -> None:
            try:
                sd.play(data, CHIME_RATE, blocking=False)
            except Exception as e:
                log.warning("chime failed: %r", e)
        threading.Thread(target=run, daemon=True, name="chime").start()

    def _reset_audio(self) -> None:
        """Drop everything buffered, so a word spoken in one state never counts in the next."""
        self._pending = np.zeros(0, np.int16)
        self._streaks = {"wake": 0, "stop": 0}
        self._ring.clear()
        self._ring_s = 0.0
        self._resampler = None
        self.oww.reset()
        if self.stop_oww is not self.oww:
            self.stop_oww.reset()

    # ------------------------------------------------------------------ state
    def wake(self, reason: str = "manual") -> None:
        """Open the gate (also used by the dashboard key and by tests)."""
        if self.awake:
            self._arm(self.grace_s, "no question after wake")
            return
        self.awake = True
        self.responding = False
        self._reset_audio()
        log.info("AWAKE (%s, best score %.2f): STT on", reason, self.best)
        self.best = 0.0
        self._chime("wake")
        self._arm(self.grace_s, "no question after wake")
        self._notify(True, reason)

    def sleep(self, reason: str = "manual") -> None:
        if not self.awake:
            return
        self._cancel()
        self.awake = False
        self.responding = False
        self._after_stop = False
        self._reset_audio()
        log.info("ASLEEP (%s): STT off, waiting for '%s'", reason, self.name)
        self._notify(False, reason)

    def stop_speaking(self) -> None:
        """Stop word heard while RESPONDING: cut the reply and open STT again."""
        log.info("STOP word heard: interrupting the reply, STT on")
        self.responding = False
        self._after_stop = True
        self._reset_audio()
        if self.session is not None:
            try:
                self.session.interrupt(force=True)    # force: the session runs with interruptions disabled
            except Exception as e:
                log.warning("interrupt failed: %r", e)
        self._chime("stop")
        self._arm(self.grace_s, "no words after stop")
        self._notify(True, "stop word")

    def _open_raw_mic(self) -> None:
        """The console runs the mic through echo cancellation / noise suppression / gain control, which push your voice
        down while the agent plays audio. Listen for the stop word on a separate, unprocessed microphone stream."""
        try:
            import sounddevice as sd

            def cb(indata, frames, t, status):
                if self.awake and self.responding and self._loop is not None:
                    self._loop.call_soon_threadsafe(self._on_raw, indata[:, 0].copy())
            self._raw_stream = sd.InputStream(samplerate=OWW_RATE, channels=1, dtype="int16", blocksize=CHUNK, callback=cb)
            self._raw_stream.start()
            log.info("stop word listens on its own raw microphone stream")
        except Exception as e:
            self._raw_stream = None
            log.warning("raw mic for the stop word unavailable (%r): using the pipeline audio", e)

    def _on_raw(self, pcm: np.ndarray) -> None:
        if not (self.awake and self.responding):
            return
        self.frames_seen += 1
        try:
            score = max(self.stop_oww.predict(pcm).values(), default=0.0)
        except Exception as e:
            log.warning("stop detection failed: %r", e)
            return
        self.best_stop = max(self.best_stop, score)
        if score >= 0.25:
            log.debug("stop score %.2f", score)
        self._streaks["stop"] = self._streaks["stop"] + 1 if score >= self.stop_threshold else 0
        if self._streaks["stop"] >= self.stop_hits_needed:
            self.stop_speaking()

    def _notify(self, awake: bool, reason: str) -> None:
        if self.on_change:
            try:
                self.on_change(awake, reason)
            except Exception as e:
                log.warning("wake on_change failed: %r", e)

    def _arm(self, seconds: float, why: str = "idle") -> None:
        self._cancel()
        self._why = f"{why} {seconds:g} s"
        try:
            self._timer = asyncio.get_running_loop().call_later(seconds, self._timeout)
        except RuntimeError:                          # no running loop (unit tests of the pure logic)
            self._timer = None

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _busy(self) -> bool:
        """Is something still going on that should keep us awake? Looked up when the timer fires, so a missed
        or reordered event cannot put the agent to sleep in the middle of a reply or a sentence."""
        s = self.session
        if s is None:
            return False
        agent = str(getattr(s, "agent_state", "")) in ("thinking", "speaking")
        user = str(getattr(s, "user_state", "")) == "speaking"
        return agent or user

    def _timeout(self) -> None:
        self._timer = None
        if self._busy():
            self._arm(1.0)                            # check again in a second
            return
        self.sleep(self._why)

    def _maybe_respond(self) -> None:
        """Enter RESPONDING once the agent is thinking/speaking AND the user has finished talking. Preemptive
        generation can start 'thinking' while you are still mid-sentence: muting then would cut your question."""
        if not (self.mute_while_replying and self.awake and not self.responding) or self.session is None:
            return
        agent = str(getattr(self.session, "agent_state", "")) in ("thinking", "speaking")
        user = str(getattr(self.session, "user_state", "")) == "speaking"
        if agent and not user:
            self.responding = True
            self.best_stop, self.frames_seen = 0.0, 0
            self._reset_audio()                       # drop stale audio and the tail of your own question
            log.info("RESPONDING: STT off, listening for the stop word")
        if agent:
            self._cancel()

    def attach(self, session) -> None:
        """Call once, before session.start()."""
        self.session = session
        self._loop = asyncio.get_running_loop()
        if self.stop_raw_mic and self.mute_while_replying:
            self._open_raw_mic()

        @session.on("agent_state_changed")
        def _agent(ev):
            if not self.awake:
                return
            new, old = str(ev.new_state), str(ev.old_state)
            if new in ("thinking", "speaking"):
                self._cancel()
                self._maybe_respond()
            elif new == "listening" and old in ("speaking", "thinking"):
                if self.responding:               # tuning aid: what the stop detector saw during this reply
                    log.info("reply over: stop detector saw %d frames, best stop score %.2f (threshold %.2f)",
                             self.frames_seen, self.best_stop, self.stop_threshold)
                self.responding = False
                self._reset_audio()
                if self._after_stop:                  # the interrupt just landed: give the user time to speak
                    self._after_stop = False
                    self._arm(self.grace_s, "no words after stop")
                else:
                    self._arm(self.idle_s, "idle")    # reply finished: the idle window starts now

        @session.on("user_state_changed")
        def _user(ev):
            if not self.awake:
                return
            if str(ev.new_state) == "speaking":
                self._cancel()                        # the user is talking: no timer until the reply is done
            else:
                self._maybe_respond()                 # user stopped; a reply may already be under way

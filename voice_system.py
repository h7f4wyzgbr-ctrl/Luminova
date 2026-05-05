#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   L U M E N   N O V A  —  V O I C E   S Y S T E M . W . x . E .              ║
║   Always-Listening STT · Multi-Engine TTS · HAT+ Routing                     ║
║   Version 1.0.0                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
The Voice System gives Lumen Nova ears and a voice. It runs continuously in
the background — no button press ever required. The user just speaks naturally
and Lumen responds.

This module has two halves:

  EARS (Speech-to-Text, always-on):
    A background thread records from the microphone in a continuous ring.
    A Voice Activity Detector (VAD) watches the energy level to find speech
    without using a wake word. When speech is detected and then stops, the
    captured audio is sent to Whisper for transcription. The transcription
    is dispatched to whichever callback is registered (usually TrinityMindEngine).

  VOICE (Text-to-Speech):
    When a Trinity Mind responds, the VoiceSystem speaks the reply using
    the best available TTS engine for the current hardware. Each mind has
    its own distinct voice — you can always tell who is speaking.

ALWAYS-LISTENING — HOW IT WORKS
────────────────────────────────
  Traditional approach: press a button, speak, release.
  Lumen's approach: run forever, detect speech automatically.

  The VAD pipeline:
    1. Capture audio in 80ms chunks from the microphone
    2. Compute RMS energy of each chunk
    3. When energy > SPEECH_THRESHOLD for N consecutive chunks → speech started
    4. Record until energy < SILENCE_THRESHOLD for SILENCE_DURATION_S seconds
    5. If captured audio is long enough → send to Whisper
    6. Whisper transcribes → text dispatched to registered callback
    7. Return to step 1

  This works without a wake word. The trade-off: Lumen will sometimes hear
  background noise as speech. You can tune SPEECH_THRESHOLD higher to make
  it less sensitive, or lower to make it more responsive.

VOICE ACTIVITY DETECTION TUNING
─────────────────────────────────
  Too many false triggers (picks up TV, fan, typing):
    → Raise SPEECH_THRESHOLD (e.g. 0.015 → 0.025)

  Missing quiet speech or distant microphone:
    → Lower SPEECH_THRESHOLD (e.g. 0.015 → 0.008)

  Cutting off speech mid-sentence (too eager to stop):
    → Raise SILENCE_DURATION_S (e.g. 1.5 → 2.5)

  Taking too long to realise you stopped speaking:
    → Lower SILENCE_DURATION_S (e.g. 1.5 → 0.8)

HARDWARE ROUTING (from HAL)
────────────────────────────
  Platform              STT                       TTS
  ────────────────────────────────────────────────────────────────
  Pi 5 + Hailo HAT+     Hailo Whisper (NPU)       Piper or eSpeak
  Pi 5 (no HAT)         faster-whisper on CPU      Piper or eSpeak
  Mac (Apple Silicon)   faster-whisper + ANE       macOS 'say'
  Mac (Intel)           faster-whisper on CPU      macOS 'say'
  Linux (no NPU)        faster-whisper on CPU      Piper → eSpeak

  The HAL tells the VoiceSystem where to run each task at startup.
  Changing hardware = change HAL = Voice System adapts automatically.

MIND VOICES
────────────
  Each Trinity Mind speaks in a distinct voice so the user always knows
  who is talking without looking at the screen.

  Mind        macOS Voice   Piper Voice          Pitch mod
  ────────────────────────────────────────────────────────────
  Sophia      Samantha      en_US-lessac-medium   0  (neutral)
  Harmonia    Karen         en_GB-alba-medium      +1 semitone
  Elysia      Moira         en_US-jenny-medium     −1 semitone
  System      Alex          en_US-ryan-medium      0

  To change a voice → edit MIND_VOICES dict in Section 1.
  To add a new TTS engine → add a method to TTSEngine following the pattern.

HOW TO MODIFY
─────────────
  • VAD sensitivity        → SPEECH_THRESHOLD, SILENCE_THRESHOLD, SILENCE_DURATION_S
  • Whisper model size     → WHISPER_MODEL_SIZE (tiny/base/small/medium)
  • TTS voices per mind    → MIND_VOICES dict
  • Add a wake word        → enable WAKE_WORD and set WAKE_WORD_TEXT
  • Mute TTS during STT    → already handled (ducking logic in TTSEngine)
  • Change language        → WHISPER_LANGUAGE constant
  • HAT Whisper endpoint   → HAT_WHISPER_URL

RUNNING STANDALONE
──────────────────
  python voice_system.py --test            # Self-test, no microphone needed
  python voice_system.py --listen          # Live always-listening demo
  python voice_system.py --speak "Hello"   # Single TTS test
  python voice_system.py --calibrate       # Measure mic noise floor
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import json
import math
import wave
import shutil
import queue
import struct
import logging
import argparse
import datetime
import tempfile
import threading
import subprocess
from abc   import ABC, abstractmethod
from enum  import Enum, auto
from pathlib import Path
from typing  import Optional, List, Dict, Callable, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

# sounddevice — audio I/O (install: pip install sounddevice)
# Wrapped in try/except: self-test works without a real audio device
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    _HAS_SOUNDDEVICE = False

# faster-whisper — local Whisper inference (install: pip install faster-whisper)
try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path.home() / "lumen_nova"

# ── Audio capture settings ────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Hz — Whisper requires 16kHz
CHANNELS          = 1       # Mono
DTYPE             = "float32"
CHUNK_MS          = 80      # Milliseconds per VAD chunk (smaller = more responsive)
CHUNK_SAMPLES     = int(SAMPLE_RATE * CHUNK_MS / 1000)

# ── Voice Activity Detection (VAD) ────────────────────────────────────────────
# The key thresholds. Read the tuning guide in the module header if needed.
SPEECH_THRESHOLD  = 0.015   # RMS energy to consider audio as "speech started"
SILENCE_THRESHOLD = 0.010   # RMS energy to consider audio as "silence"
SILENCE_DURATION_S= 1.5     # Seconds of consecutive silence before stopping recording
MIN_SPEECH_MS     = 300     # Ignore clips shorter than this (filters noise bursts)
MAX_SPEECH_S      = 30      # Maximum recording length before forced cutoff

# ── Wake word (optional) ──────────────────────────────────────────────────────
# Set WAKE_WORD_ENABLED = True to require the user to say the wake word first.
# The wake word is detected by Whisper after a short capture.
# False = truly always-listening (VAD only, no wake word required)
WAKE_WORD_ENABLED = False
WAKE_WORD_TEXT    = "lumen"   # What to listen for (case-insensitive)

# ── Whisper settings ──────────────────────────────────────────────────────────
# Model sizes and their approximate RAM usage:
#   tiny    ~390 MB download  ~200 MB RAM   fast, less accurate
#   base    ~145 MB download  ~300 MB RAM   good balance  ← default
#   small   ~483 MB download  ~600 MB RAM   better accuracy
#   medium  ~1.5 GB download  ~1.2 GB RAM   high accuracy
#
# On Pi 5 with HAT+, the HAT runs Whisper on the NPU, so model size doesn't
# affect Pi RAM. Only matters when running on Pi CPU.
WHISPER_MODEL_SIZE = "base"          # Change to "small" for better accuracy
WHISPER_LANGUAGE   = "en"            # Language code, or None for auto-detect
WHISPER_DEVICE     = "cpu"           # "cpu" or "cuda" (if you have GPU)
WHISPER_COMPUTE    = "int8"          # "int8" = fastest on CPU, "float16" for GPU

# ── HAT+ Whisper service ──────────────────────────────────────────────────────
# If the AI HAT+ is present and its Whisper service is running,
# audio is sent here instead of running Whisper on the Pi CPU.
HAT_WHISPER_URL    = "http://localhost:5001/transcribe"   # Change to match your HAT
HAT_WHISPER_TIMEOUT= 8    # Seconds to wait for HAT transcription

# ── TTS engine priority order (tried in this order until one works) ───────────
# Comment out entries to disable an engine.
TTS_ENGINE_PRIORITY = [
    "macos_say",    # macOS only — best quality, zero setup
    "piper",        # Linux/Pi — good quality, offline
    "espeak",       # Universal fallback — robotic but always works
    "festival",     # Another Linux fallback
]

# ── Mind voice definitions ────────────────────────────────────────────────────
# Each mind speaks in a different voice. Edit freely.
# macos_voice: any voice from `say -v '?'` in Terminal
# piper_model: filename in your piper voices directory (without .onnx)
# espeak_voice: espeak -v option value
# rate: words per minute (macOS say -r, piper --length-scale inverse)
# pitch: semitone adjustment for espeak (-p 50 = neutral, higher = higher pitch)
MIND_VOICES: Dict[str, dict] = {
    "sophia": {
        "description":  "Clear, precise, scientific",
        "macos_voice":  "Samantha",
        "piper_model":  "en_US-lessac-medium",
        "espeak_voice": "en-us+f2",
        "rate":         185,          # WPM — Sophia speaks clearly, not rushed
        "pitch":        52,           # Slightly above neutral
    },
    "harmonia": {
        "description":  "Rhythmic, structured, elegant",
        "macos_voice":  "Karen",
        "piper_model":  "en_GB-alba-medium",
        "espeak_voice": "en-gb",
        "rate":         175,          # Slightly slower — builds structure
        "pitch":        48,           # Slightly lower — grounded, precise
    },
    "elysia": {
        "description":  "Warm, personal, unhurried",
        "macos_voice":  "Moira",
        "piper_model":  "en_US-jenny-medium",
        "espeak_voice": "en-us+f3",   # Female voice variant
        "rate":         165,          # Unhurried — Elysia never rushes
        "pitch":        58,           # Slightly warmer/higher
    },
    "system": {
        "description":  "Neutral system announcements",
        "macos_voice":  "Alex",
        "piper_model":  "en_US-amy-medium",
        "espeak_voice": "en-us+f1",
        "rate":         190,
        "pitch":        50,
    },
}

# ── Audio output directory (for saving TTS output or voice memos) ─────────────
AUDIO_DIR = BASE_DIR / "audio"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = logging.INFO


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _setup_log() -> logging.Logger:
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log_file = BASE_DIR / "logs" / f"voice_{datetime.date.today()}.log"
    fmt      = "%(asctime)s [%(levelname)-8s] Voice › %(message)s"
    logger   = logging.getLogger("VoiceSystem")
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        for h in [logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]:
            h.setFormatter(logging.Formatter(fmt))
            logger.addHandler(h)
    return logger

log = _setup_log()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: VOICE ACTIVITY DETECTOR (VAD)
#
# Watches a continuous stream of 80ms audio chunks and decides:
#   - Is someone speaking right now?
#   - Has speech just started?
#   - Has speech just ended?
#
# Pure numpy — no ML model needed for basic VAD.
# Uses energy (RMS) as the primary signal.
#
# HOW IT WORKS:
#   The VAD maintains a state machine with three states:
#     WAITING  → listening for the start of speech
#     SPEAKING → actively recording speech
#     TRAILING → speech energy dropped, waiting to confirm silence
#
# HOW TO IMPROVE:
#   Replace RMS energy with Silero VAD (a tiny 1MB neural VAD model)
#   for dramatically better accuracy in noisy environments.
#   The interface is identical — just swap the _compute_energy() method.
# ─────────────────────────────────────────────────────────────────────────────

class VADState(Enum):
    WAITING  = auto()   # Listening for speech to begin
    SPEAKING = auto()   # Recording active speech
    TRAILING = auto()   # Possible end of speech, counting silence


@dataclass
class VADEvent:
    """An event fired by the VAD when speech starts or ends."""
    event_type: str    # "speech_start" | "speech_end" | "noise"
    audio:      Optional[np.ndarray] = None   # Captured audio (for speech_end)
    duration_ms: int = 0


class VoiceActivityDetector:
    """
    Energy-based Voice Activity Detector. Runs on each 80ms audio chunk.

    Feed chunks via process_chunk(). Events come out via on_event callback.
    Thread-safe — designed to run in the audio capture thread.
    """

    def __init__(
        self,
        speech_threshold:  float = SPEECH_THRESHOLD,
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_duration:  float = SILENCE_DURATION_S,
        min_speech_ms:     int   = MIN_SPEECH_MS,
        max_speech_s:      float = MAX_SPEECH_S,
        sample_rate:       int   = SAMPLE_RATE,
        on_event: Optional[Callable[[VADEvent], None]] = None,
    ):
        self.speech_threshold  = speech_threshold
        self.silence_threshold = silence_threshold
        self.silence_duration  = silence_duration
        self.min_speech_ms     = min_speech_ms
        self.max_speech_s      = max_speech_s
        self.sample_rate       = sample_rate
        self.on_event          = on_event

        # Internal state
        self._state:          VADState = VADState.WAITING
        self._speech_chunks:  List[np.ndarray] = []
        self._silence_chunks: int = 0
        self._silence_needed  = int(silence_duration / (CHUNK_MS / 1000))
        self._max_chunks      = int(max_speech_s * 1000 / CHUNK_MS)

        # Noise floor calibration — updated during WAITING state
        self._noise_floor     = 0.0
        self._noise_samples   = 0

        # Energy history for adaptive thresholding
        self._energy_history: List[float] = []
        self._history_len     = 50    # ~4 seconds of history

    def process_chunk(self, chunk: np.ndarray) -> Optional[VADEvent]:
        """
        Process one audio chunk. Returns a VADEvent if something happened,
        None otherwise.

        Args:
            chunk: float32 numpy array at SAMPLE_RATE Hz

        Returns:
            VADEvent or None
        """
        energy = self._compute_energy(chunk)
        self._update_history(energy)

        if self._state == VADState.WAITING:
            return self._handle_waiting(chunk, energy)
        elif self._state == VADState.SPEAKING:
            return self._handle_speaking(chunk, energy)
        elif self._state == VADState.TRAILING:
            return self._handle_trailing(chunk, energy)
        return None

    def calibrate_noise_floor(self, duration_s: float = 2.0) -> float:
        """
        Sample ambient noise for N seconds to set an adaptive threshold.
        Call this at startup before starting the listen loop.

        Returns the measured noise floor RMS.
        """
        if not _HAS_SOUNDDEVICE:
            return SPEECH_THRESHOLD

        log.info(f"Calibrating noise floor for {duration_s}s… (stay quiet)")
        n_chunks = int(duration_s * 1000 / CHUNK_MS)
        energies = []
        try:
            with sd.InputStream(
                samplerate = self.sample_rate,
                channels   = CHANNELS,
                dtype      = DTYPE,
                blocksize  = CHUNK_SAMPLES,
            ) as stream:
                for _ in range(n_chunks):
                    data, _ = stream.read(CHUNK_SAMPLES)
                    energies.append(self._compute_energy(data.flatten()))
        except Exception as e:
            log.warning(f"Noise calibration failed: {e}")
            return SPEECH_THRESHOLD

        if not energies:
            return SPEECH_THRESHOLD

        noise    = float(np.percentile(energies, 90))   # 90th percentile = typical noise
        self._noise_floor = noise

        # Set thresholds adaptively: speech = 3× noise floor, silence = 2×
        adaptive_speech  = max(SPEECH_THRESHOLD,  noise * 3.0)
        adaptive_silence = max(SILENCE_THRESHOLD, noise * 2.0)
        self.speech_threshold  = adaptive_speech
        self.silence_threshold = adaptive_silence

        log.info(
            f"Noise floor: {noise:.4f}  "
            f"Speech threshold: {adaptive_speech:.4f}  "
            f"Silence threshold: {adaptive_silence:.4f}"
        )
        return noise

    def reset(self):
        """Reset to WAITING state, discarding any partial recording."""
        self._state          = VADState.WAITING
        self._speech_chunks  = []
        self._silence_chunks = 0

    # ── Internal state handlers ────────────────────────────────────────────────

    def _handle_waiting(self, chunk: np.ndarray, energy: float) -> Optional[VADEvent]:
        """In WAITING: update noise floor and watch for speech start."""
        # Adaptive noise floor update (only in silence)
        self._noise_floor = self._noise_floor * 0.98 + energy * 0.02

        if energy > self.speech_threshold:
            # Speech detected — transition to SPEAKING
            self._state = VADState.SPEAKING
            self._speech_chunks = [chunk.copy()]
            self._silence_chunks = 0

            event = VADEvent(event_type="speech_start")
            if self.on_event:
                self.on_event(event)
            return event

        return None

    def _handle_speaking(self, chunk: np.ndarray, energy: float) -> Optional[VADEvent]:
        """In SPEAKING: accumulate audio and watch for silence."""
        self._speech_chunks.append(chunk.copy())

        if energy < self.silence_threshold:
            # Energy dropped — might be end of speech
            self._silence_chunks += 1
            if self._silence_chunks >= self._silence_needed:
                # Confirmed silence — speech has ended
                return self._finalize_speech()
        else:
            # Still speaking — reset silence counter
            self._silence_chunks = 0

        # Safety: cut off if recording too long
        if len(self._speech_chunks) >= self._max_chunks:
            log.warning("Max speech duration reached — forcing cutoff")
            return self._finalize_speech()

        return None

    def _handle_trailing(self, chunk: np.ndarray, energy: float) -> Optional[VADEvent]:
        """Trailing state is merged into SPEAKING's silence counter above."""
        return self._handle_speaking(chunk, energy)

    def _finalize_speech(self) -> Optional[VADEvent]:
        """Combine accumulated chunks, validate length, fire speech_end event."""
        audio    = np.concatenate(self._speech_chunks).flatten()
        dur_ms   = int(len(audio) / self.sample_rate * 1000)
        self.reset()

        if dur_ms < self.min_speech_ms:
            log.debug(f"Speech clip too short ({dur_ms}ms) — discarded")
            return None

        event = VADEvent(event_type="speech_end", audio=audio, duration_ms=dur_ms)
        if self.on_event:
            self.on_event(event)
        return event

    @staticmethod
    def _compute_energy(chunk: np.ndarray) -> float:
        """RMS energy of an audio chunk. Fast, no dependencies."""
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

    def _update_history(self, energy: float):
        self._energy_history.append(energy)
        if len(self._energy_history) > self._history_len:
            self._energy_history.pop(0)

    @property
    def adaptive_threshold(self) -> float:
        """Current adaptive speech threshold (noise floor × 3)."""
        return max(SPEECH_THRESHOLD, self._noise_floor * 3.0)

    @property
    def recent_energy(self) -> float:
        """Average energy of recent audio history."""
        if not self._energy_history:
            return 0.0
        return float(np.mean(self._energy_history[-10:]))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: WHISPER STT ENGINE
#
# Local Whisper transcription via faster-whisper.
# Also supports routing to the AI HAT+ Whisper service.
#
# The WhisperEngine is lazy-loaded — the model is only loaded into RAM
# the first time a transcription is needed. This keeps startup fast.
#
# MEMORY NOTE:
#   Whisper base model uses ~300MB RAM.
#   This is shared with the LLM (qwen2.5:1.5b at ~1.3GB).
#   Total on Pi 5 4GB: 300 + 1300 + 256 safety = ~1856MB — fits in 4GB.
#   On Pi 5 with HAT+: Whisper runs on the HAT NPU, so only the LLM
#   uses Pi CPU RAM (~1300MB), very comfortable.
# ─────────────────────────────────────────────────────────────────────────────

class WhisperSTT:
    """
    Speech-to-text using faster-whisper (local) or HAT+ Whisper (NPU).
    Lazy-loads the Whisper model on first transcription call.
    Thread-safe — guarded by a lock.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        language:   str = WHISPER_LANGUAGE,
        device:     str = WHISPER_DEVICE,
        compute:    str = WHISPER_COMPUTE,
        use_hat:    bool = False,
        hat_url:    str = HAT_WHISPER_URL,
    ):
        self.model_size = model_size
        self.language   = language
        self.device     = device
        self.compute    = compute
        self.use_hat    = use_hat
        self.hat_url    = hat_url

        self._model: Optional[WhisperModel] = None
        self._lock  = threading.Lock()
        self._load_attempted = False

    @property
    def model(self) -> Optional["WhisperModel"]:
        """
        Lazy-load the Whisper model on first access.
        Returns None if faster-whisper is not installed or loading fails.
        """
        if not self._load_attempted:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load Whisper model into RAM. Called once, on first transcription."""
        self._load_attempted = True
        if not _HAS_WHISPER:
            log.error("faster-whisper not installed. Run: pip install faster-whisper")
            return
        try:
            log.info(f"Loading Whisper '{self.model_size}' on {self.device}…")
            t = time.monotonic()
            self._model = WhisperModel(
                self.model_size,
                device       = self.device,
                compute_type = self.compute,
            )
            log.info(f"Whisper loaded in {(time.monotonic()-t)*1000:.0f}ms")
        except Exception as e:
            log.error(f"Whisper load failed: {e}")
            self._model = None

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 numpy audio array to text.

        Routes to HAT+ Whisper if use_hat is True and the service is reachable.
        Falls back to local Whisper if HAT is unavailable.

        Args:
            audio: float32 array at 16000 Hz, mono

        Returns:
            Transcribed text string (empty string if nothing heard)
        """
        with self._lock:
            # Route to HAT Whisper if available
            if self.use_hat:
                result = self._transcribe_hat(audio)
                if result is not None:
                    return result
                log.warning("HAT Whisper unavailable, falling back to local")

            # Local faster-whisper
            return self._transcribe_local(audio)

    def _transcribe_local(self, audio: np.ndarray) -> str:
        """Transcribe using local faster-whisper model."""
        if self.model is None:
            return ""

        # Validate audio
        if len(audio) < CHUNK_SAMPLES:
            return ""

        # Normalize to [-1, 1] float32
        audio_f32 = audio.astype(np.float32)
        peak      = np.abs(audio_f32).max()
        if peak > 0:
            audio_f32 = audio_f32 / peak

        try:
            segments, info = self.model.transcribe(
                audio_f32,
                language              = self.language,
                beam_size             = 5,
                vad_filter            = True,    # Built-in VAD filter
                vad_parameters        = dict(
                    min_silence_duration_ms = 500,
                    speech_pad_ms           = 400,
                ),
                without_timestamps    = True,
                condition_on_previous_text = False,  # More accurate without context
            )
            text = " ".join(seg.text for seg in segments).strip()
            # Filter out common Whisper hallucinations
            text = self._filter_hallucinations(text)
            if text:
                log.info(f"STT [{info.language} {info.language_probability:.0%}]: {text[:80]}")
            return text

        except Exception as e:
            log.error(f"Transcription error: {e}")
            return ""

    def _transcribe_hat(self, audio: np.ndarray) -> Optional[str]:
        """
        Send audio to the AI HAT+ Whisper service via HTTP.
        Returns None if the service is unreachable (triggering fallback).

        The HAT service expects:
          POST /transcribe
          Content-Type: audio/wav
          Body: WAV file bytes

        Adjust this method if your HAT service has a different API.
        """
        try:
            import urllib.request
            import urllib.error

            # Convert numpy array to WAV bytes in memory
            wav_bytes = _numpy_to_wav_bytes(audio, SAMPLE_RATE)

            req = urllib.request.Request(
                self.hat_url,
                data    = wav_bytes,
                headers = {"Content-Type": "audio/wav"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=HAT_WHISPER_TIMEOUT) as resp:
                result = json.loads(resp.read())
                return result.get("text", "").strip()

        except Exception as e:
            log.debug(f"HAT Whisper call failed: {e}")
            return None

    @staticmethod
    def _filter_hallucinations(text: str) -> str:
        """
        Remove common Whisper hallucinations — phrases it generates when
        it hears silence or background noise.

        Add more patterns here if Whisper keeps hallucinating specific phrases.
        """
        hallucinations = [
            "thank you",
            "thanks for watching",
            "please subscribe",
            "you",
            ".",
            "...",
            "bye",
            "goodbye",
            "[blank_audio]",
            "[BLANK_AUDIO]",
            "(silence)",
        ]
        stripped = text.strip().lower()
        if stripped in hallucinations:
            return ""
        if len(stripped) < 2:
            return ""
        return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TTS ENGINES
#
# Abstract base class + concrete implementations for each TTS engine.
# The VoiceSystem tries engines in priority order until one succeeds.
#
# TO ADD A NEW TTS ENGINE:
#   1. Create a class that inherits from TTSEngineBase
#   2. Implement is_available() and speak()
#   3. Add it to TTS_ENGINE_PRIORITY list in Section 1
#   4. Add it to TTSEngineFactory.create() below
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngineBase(ABC):
    """Abstract base class for all TTS engines."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def speak(
        self,
        text:       str,
        voice_cfg:  dict,
        blocking:   bool = False,
    ) -> bool:
        """
        Speak text using this engine.

        Args:
            text:      Text to speak (may contain markdown — will be cleaned)
            voice_cfg: Dict from MIND_VOICES for the active mind
            blocking:  If True, wait until speech finishes before returning

        Returns:
            True if speech was started/completed successfully
        """
        ...

    def stop(self):
        """Stop any currently playing speech. Override if engine supports it."""
        pass

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Strip markdown and other symbols that sound wrong when spoken.
        Called by all TTS engines before sending text to the speech engine.
        """
        import re
        # Remove markdown formatting
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)   # Bold/italic
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)  # Code
        text = re.sub(r"#{1,6}\s*", "", text)                 # Headers
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text) # Links
        text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)  # Lists
        text = re.sub(r"\n{2,}", ". ", text)                  # Paragraph breaks
        text = re.sub(r"\n", " ", text)                        # Line breaks
        text = re.sub(r"[│┃╔╗╚╝═─]", "", text)               # Box chars
        text = re.sub(r"  +", " ", text)                       # Double spaces
        return text.strip()

    @staticmethod
    def _truncate(text: str, max_chars: int = 800) -> str:
        """
        Truncate text to a reasonable spoken length.
        Cuts at a sentence boundary to avoid mid-sentence cutoffs.
        """
        if len(text) <= max_chars:
            return text
        # Try to cut at sentence boundary
        truncated = text[:max_chars]
        last_period = max(
            truncated.rfind(". "),
            truncated.rfind("? "),
            truncated.rfind("! "),
        )
        if last_period > max_chars // 2:
            return truncated[:last_period + 1]
        return truncated + "…"


class MacOSSayTTS(TTSEngineBase):
    """
    macOS built-in 'say' command — highest quality, zero setup.
    Available on every Mac. Each mind has a distinct voice.
    """

    @property
    def name(self) -> str:
        return "macos_say"

    def is_available(self) -> bool:
        return sys.platform == "darwin" and bool(shutil.which("say"))

    def speak(self, text: str, voice_cfg: dict, blocking: bool = False) -> bool:
        clean = self._clean_text(self._truncate(text))
        if not clean:
            return False

        voice  = voice_cfg.get("macos_voice", "Samantha")
        rate   = voice_cfg.get("rate", 180)

        cmd = ["say", "-v", voice, "-r", str(rate), clean]
        try:
            if blocking:
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            log.error(f"macOS say error: {e}")
            return False

    def stop(self):
        """Kill any running 'say' process."""
        try:
            subprocess.run(["pkill", "-f", "say"], capture_output=True)
        except Exception:
            pass


class PiperTTS(TTSEngineBase):
    """
    Piper TTS — fast, offline, good quality. Best choice for Pi/Linux.
    Install: https://github.com/rhasspy/piper
    Models: https://huggingface.co/rhasspy/piper-voices

    Expected voice model path:
        ~/lumen_nova/tts_voices/{model_name}.onnx
        ~/lumen_nova/tts_voices/{model_name}.onnx.json
    """
    VOICES_DIR = BASE_DIR / "tts_voices"

    @property
    def name(self) -> str:
        return "piper"

    def is_available(self) -> bool:
        return bool(shutil.which("piper") or shutil.which("piper-tts"))

    def _piper_cmd(self) -> str:
        return shutil.which("piper") or shutil.which("piper-tts") or "piper"

    def _find_voice_model(self, model_name: str) -> Optional[Path]:
        """Find the .onnx voice model file."""
        # Check Lumen's voices directory first
        candidates = [
            self.VOICES_DIR / f"{model_name}.onnx",
            self.VOICES_DIR / f"{model_name}",
            Path.home() / ".local" / "share" / "piper-voices" / f"{model_name}.onnx",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def speak(self, text: str, voice_cfg: dict, blocking: bool = False) -> bool:
        clean      = self._clean_text(self._truncate(text))
        if not clean:
            return False

        model_name = voice_cfg.get("piper_model", "en_US-lessac-medium")
        rate       = voice_cfg.get("rate", 180)
        model_path = self._find_voice_model(model_name)

        if model_path is None:
            log.warning(
                f"Piper voice model not found: {model_name}\n"
                f"Download from: https://huggingface.co/rhasspy/piper-voices\n"
                f"Place in: {self.VOICES_DIR}/"
            )
            return False

        # length_scale controls speed: 1.0 = normal, >1 = slower, <1 = faster
        # Convert WPM rate to length_scale (approximate)
        length_scale = max(0.5, min(2.0, 180.0 / rate))

        cmd = [
            self._piper_cmd(),
            "--model", str(model_path),
            "--length-scale", f"{length_scale:.2f}",
            "--output-raw",
        ]

        try:
            # Pipe text to piper → raw PCM → aplay
            aplay_cmd = ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"]
            if not shutil.which("aplay"):
                # Fallback: save to temp file and play with paplay or ffplay
                aplay_cmd = self._fallback_player()

            piper_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            player_proc = subprocess.Popen(
                aplay_cmd, stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            piper_proc.stdin.write(clean.encode())
            piper_proc.stdin.close()
            piper_proc.stdout.close()

            if blocking:
                player_proc.wait()
            return True

        except Exception as e:
            log.error(f"Piper TTS error: {e}")
            return False

    @staticmethod
    def _fallback_player() -> List[str]:
        for p in ["paplay", "ffplay", "cvlc"]:
            if shutil.which(p):
                if p == "paplay":
                    return ["paplay", "--raw", "--rate=22050", "--format=s16le",
                            "--channels=1", "-"]
                if p == "ffplay":
                    return ["ffplay", "-f", "s16le", "-ar", "22050", "-ac", "1",
                            "-nodisp", "-autoexit", "-"]
        return ["cat"]   # Absolute last resort (silent)


class ESpeakTTS(TTSEngineBase):
    """
    eSpeak / eSpeak-NG TTS — robotic but always available on Linux/Pi.
    Included in most Linux distributions: sudo apt install espeak-ng
    """

    @property
    def name(self) -> str:
        return "espeak"

    def is_available(self) -> bool:
        return bool(shutil.which("espeak-ng") or shutil.which("espeak"))

    def _cmd(self) -> str:
        return shutil.which("espeak-ng") or shutil.which("espeak") or "espeak"

    def speak(self, text: str, voice_cfg: dict, blocking: bool = False) -> bool:
        clean = self._clean_text(self._truncate(text, max_chars=600))
        if not clean:
            return False

        voice  = voice_cfg.get("espeak_voice", "en-us")
        rate   = voice_cfg.get("rate", 175)
        pitch  = voice_cfg.get("pitch", 50)

        cmd = [
            self._cmd(),
            "-v", voice,
            "-s", str(rate),
            "-p", str(pitch),
            clean,
        ]
        try:
            if blocking:
                subprocess.run(cmd, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            log.error(f"eSpeak error: {e}")
            return False

    def stop(self):
        try:
            subprocess.run(["pkill", "-f", "espeak"], capture_output=True)
        except Exception:
            pass


class FestivalTTS(TTSEngineBase):
    """
    Festival TTS — another Linux fallback.
    Install: sudo apt install festival
    """

    @property
    def name(self) -> str:
        return "festival"

    def is_available(self) -> bool:
        return bool(shutil.which("festival"))

    def speak(self, text: str, voice_cfg: dict, blocking: bool = False) -> bool:
        clean = self._clean_text(self._truncate(text, max_chars=500))
        if not clean:
            return False
        try:
            proc = subprocess.Popen(
                ["festival", "--tts"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            proc.stdin.write(clean.encode())
            proc.stdin.close()
            if blocking:
                proc.wait()
            return True
        except Exception as e:
            log.error(f"Festival error: {e}")
            return False


class TTSEngineFactory:
    """
    Selects and caches the best available TTS engine.
    Tries engines in TTS_ENGINE_PRIORITY order.
    """

    _engine_classes = {
        "macos_say": MacOSSayTTS,
        "piper":     PiperTTS,
        "espeak":    ESpeakTTS,
        "festival":  FestivalTTS,
    }

    @classmethod
    def create_best(cls) -> Optional[TTSEngineBase]:
        """
        Return the first available TTS engine in priority order.
        Returns None if nothing is available (very unusual).
        """
        for name in TTS_ENGINE_PRIORITY:
            klass = cls._engine_classes.get(name)
            if klass:
                engine = klass()
                if engine.is_available():
                    log.info(f"TTS engine selected: {name}")
                    return engine
        log.error("No TTS engine available. Install espeak: sudo apt install espeak-ng")
        return None

    @classmethod
    def probe_all(cls) -> Dict[str, bool]:
        """Return availability status for all known TTS engines."""
        return {name: cls._engine_classes[name]().is_available()
                for name in cls._engine_classes}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: AUDIO CAPTURE LOOP
#
# The AudioCaptureLoop runs in a dedicated background daemon thread.
# It continuously reads from the microphone in CHUNK_SAMPLES blocks,
# feeds each block to the VAD, and when the VAD fires a speech_end event
# it puts the audio into a transcription queue.
#
# The transcription queue is drained by a separate TranscriptionWorker thread,
# which keeps the capture thread unblocked (Whisper takes 0.5-2 seconds).
#
# THREAD ARCHITECTURE:
#   AudioCaptureLoop thread  →  Queue[np.ndarray]  →  TranscriptionWorker thread
#                                                            ↓
#                                                    registered callback(text)
# ─────────────────────────────────────────────────────────────────────────────

class AudioCaptureLoop:
    """
    Continuous microphone capture loop with VAD.
    Runs in its own daemon thread. Speech clips go into audio_queue.
    """

    def __init__(
        self,
        vad:          VoiceActivityDetector,
        audio_queue:  queue.Queue,
        device_index: Optional[int] = None,
    ):
        self._vad          = vad
        self._audio_queue  = audio_queue
        self._device       = device_index
        self._stop         = threading.Event()
        self._muted        = threading.Event()   # Set = muted (don't capture)
        self._thread:      Optional[threading.Thread] = None
        self.is_running    = False

    def start(self):
        """Start the capture loop in a daemon thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target = self._loop,
            daemon = True,
            name   = "AudioCapture",
        )
        self._thread.start()
        self.is_running = True
        log.info("Audio capture loop started")

    def stop(self):
        """Signal the capture loop to stop."""
        self._stop.set()
        self.is_running = False

    def mute(self):
        """
        Mute the microphone temporarily (e.g. while TTS is speaking).
        Prevents Lumen from hearing its own voice and triggering itself.
        """
        self._muted.set()

    def unmute(self):
        """Resume microphone capture after TTS finishes."""
        self._muted.clear()

    def _loop(self):
        """
        The actual capture loop. Reads chunks from the microphone forever.
        Feeds each chunk to the VAD. On speech_end, queues the audio clip.
        """
        if not _HAS_SOUNDDEVICE:
            log.error("sounddevice not available — audio capture disabled")
            return

        log.info(f"Opening microphone (device={self._device}, "
                 f"{SAMPLE_RATE}Hz, {CHUNK_SAMPLES} samples/chunk)")

        try:
            with sd.InputStream(
                samplerate = SAMPLE_RATE,
                channels   = CHANNELS,
                dtype      = DTYPE,
                blocksize  = CHUNK_SAMPLES,
                device     = self._device,
            ) as stream:
                log.info("Microphone open. Always listening.")

                while not self._stop.is_set():
                    # Read one chunk
                    chunk, overflowed = stream.read(CHUNK_SAMPLES)
                    if overflowed:
                        log.debug("Audio buffer overflow — chunk dropped")
                        continue

                    # Skip if muted (TTS is speaking)
                    if self._muted.is_set():
                        continue

                    # Feed to VAD
                    event = self._vad.process_chunk(chunk.flatten())
                    if event and event.event_type == "speech_end" and event.audio is not None:
                        try:
                            self._audio_queue.put_nowait(event.audio)
                        except queue.Full:
                            log.warning("Transcription queue full — speech clip dropped")

        except sd.PortAudioError as e:
            log.error(f"PortAudio error: {e}")
        except Exception as e:
            log.error(f"Capture loop error: {e}")
        finally:
            self.is_running = False
            log.info("Audio capture loop stopped")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TRANSCRIPTION WORKER
#
# A second thread drains the audio_queue, runs Whisper on each clip,
# and calls the registered on_transcript callback with the resulting text.
#
# This is separate from AudioCaptureLoop so that slow Whisper transcription
# never blocks the microphone capture.
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionWorker:
    """
    Drains the audio queue, transcribes clips with Whisper,
    dispatches text to the registered callback.
    """

    def __init__(
        self,
        stt:          WhisperSTT,
        audio_queue:  queue.Queue,
        on_transcript: Callable[[str], None],
        capture_loop: Optional[AudioCaptureLoop] = None,
    ):
        self._stt           = stt
        self._queue         = audio_queue
        self._callback      = on_transcript
        self._capture       = capture_loop   # For mute/unmute around TTS
        self._stop          = threading.Event()
        self._thread:       Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target = self._loop,
            daemon = True,
            name   = "Transcription",
        )
        self._thread.start()
        log.info("Transcription worker started")

    def stop(self):
        self._stop.set()

    def _loop(self):
        """Drain the queue, transcribe, dispatch. Loops until stopped."""
        while not self._stop.is_set():
            try:
                # Block for up to 1 second waiting for audio
                try:
                    audio = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Wake word check (if enabled)
                if WAKE_WORD_ENABLED:
                    # Transcribe just to check for wake word
                    quick_text = self._stt.transcribe(audio)
                    if WAKE_WORD_TEXT.lower() not in quick_text.lower():
                        log.debug(f"Wake word '{WAKE_WORD_TEXT}' not detected, skipping")
                        continue
                    # Wake word found — transcribe rest of clip
                    text = quick_text.replace(WAKE_WORD_TEXT, "", 1).strip()
                    if not text:
                        # Wake word with nothing after — listen for the actual command
                        # (In a full implementation, you'd do a second capture here)
                        continue
                else:
                    text = self._stt.transcribe(audio)

                if text:
                    log.info(f"Transcript: '{text}'")
                    try:
                        self._callback(text)
                    except Exception as e:
                        log.error(f"Transcript callback error: {e}")

            except Exception as e:
                log.error(f"Transcription worker error: {e}")
                time.sleep(0.5)

        log.info("Transcription worker stopped")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: TTS SPEAKER
#
# Manages TTS output: engine selection, muting the microphone during speech
# (to prevent feedback), and queueing multiple speech requests.
#
# Each mind has a distinct voice configuration from MIND_VOICES.
# The speaker picks the right voice automatically from the mind_id.
# ─────────────────────────────────────────────────────────────────────────────

class TTSSpeaker:
    """
    Text-to-speech output manager. Thread-safe. Non-blocking by default.
    """

    def __init__(self, capture_loop: Optional[AudioCaptureLoop] = None):
        self._engine       = TTSEngineFactory.create_best()
        self._capture      = capture_loop   # To mute mic during speech
        self._speech_queue: queue.Queue = queue.Queue(maxsize=8)
        self._stop         = threading.Event()
        self._thread:      Optional[threading.Thread] = None

    def start(self):
        """Start the speech output worker thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target = self._worker,
            daemon = True,
            name   = "TTSSpeaker",
        )
        self._thread.start()
        log.info(
            f"TTS speaker started. Engine: "
            f"{self._engine.name if self._engine else 'NONE'}"
        )

    def stop(self):
        self._stop.set()

    def speak(
        self,
        text:    str,
        mind_id: str = "system",
        block:   bool = False,
    ) -> bool:
        """
        Queue text for speech output.

        Args:
            text:    Text to speak. Will be cleaned of markdown automatically.
            mind_id: Which mind is speaking ("sophia"/"harmonia"/"elysia"/"system")
            block:   If True, don't return until speech finishes.
                     WARNING: blocks the calling thread — use carefully.

        Returns:
            True if queued or spoken successfully
        """
        if not self._engine:
            log.warning("No TTS engine available")
            return False

        if not text or not text.strip():
            return False

        voice_cfg = MIND_VOICES.get(mind_id, MIND_VOICES["system"])

        if block:
            return self._speak_now(text, voice_cfg)

        try:
            self._speech_queue.put_nowait((text, voice_cfg))
            return True
        except queue.Full:
            log.warning("TTS queue full — speech dropped")
            return False

    def stop_speaking(self):
        """Immediately stop current speech and clear the queue."""
        # Drain the queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        # Stop the engine
        if self._engine:
            self._engine.stop()

    def is_available(self) -> bool:
        return self._engine is not None and self._engine.is_available()

    def _worker(self):
        """Drain the speech queue. Mutes mic before speaking, unmutes after."""
        while not self._stop.is_set():
            try:
                text, voice_cfg = self._speech_queue.get(timeout=1.0)
                self._speak_now(text, voice_cfg)
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"TTS worker error: {e}")

    def _speak_now(self, text: str, voice_cfg: dict) -> bool:
        """Speak synchronously with mic muting."""
        # Mute mic to prevent feedback loop
        if self._capture:
            self._capture.mute()

        try:
            ok = self._engine.speak(text, voice_cfg, blocking=True)
            return ok
        finally:
            # Always unmute, even if speech failed
            if self._capture:
                # Brief pause before unmuting (some TTS has reverb tail)
                time.sleep(0.3)
                self._capture.unmute()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: VOICE SYSTEM (PUBLIC API)
#
# This is the single object the rest of Lumen Nova imports and uses.
# It assembles and manages all the voice components:
#
#   AudioCaptureLoop → TranscriptionWorker → user callback (TrinityMindEngine)
#   User callback result → TTSSpeaker → microphone muted during speech
#
# HOW TO CONNECT TO TRINITY MINDS:
#
#   from voice_system   import VoiceSystem
#   from trinity_minds  import TrinityMindEngine
#   from hardware_hal   import HardwareAbstractionLayer, TaskType, TaskDestination
#
#   hal     = HardwareAbstractionLayer()
#   profile = hal.probe()
#   minds   = TrinityMindEngine(memory_core=mem)
#   minds.start()
#
#   voice = VoiceSystem(profile=profile)
#   voice.start()
#
#   def on_heard(text: str):
#       response, mind_id = minds.chat(text)
#       voice.speak(response, mind_id=mind_id)
#
#   voice.set_transcript_callback(on_heard)
# ─────────────────────────────────────────────────────────────────────────────

class VoiceSystem:
    """
    Complete always-listening voice interface for Lumen Nova.
    Owns: VAD, AudioCaptureLoop, TranscriptionWorker, TTSSpeaker.
    Thread-safe. Manages the full voice pipeline lifecycle.
    """

    def __init__(
        self,
        profile=None,              # HardwareProfile from hal.probe()
        transcript_callback: Optional[Callable[[str], None]] = None,
        audio_device_index:  Optional[int] = None,
        calibrate_on_start:  bool = True,
    ):
        """
        Args:
            profile:             HardwareProfile. Routes STT to HAT if available.
            transcript_callback: Called with transcribed text when speech detected.
                                 Set this to your TrinityMindEngine.chat wrapper.
            audio_device_index:  Specific sounddevice index, or None for default.
            calibrate_on_start:  If True, sample noise floor at startup.
        """
        self._profile   = profile
        self._callback  = transcript_callback
        self._device    = audio_device_index
        self._calibrate = calibrate_on_start

        # Determine STT routing from hardware profile
        use_hat = (
            profile is not None and
            getattr(profile, "has_hailo",         False) and
            getattr(profile, "hat_voice_available",False)
        )

        # Whisper model size: use smaller model on Pi
        is_pi = (profile is not None and
                 "raspberry_pi" in str(getattr(profile, "platform_type", "")).lower())
        model_size = "tiny" if is_pi and not use_hat else WHISPER_MODEL_SIZE

        # Build components
        self._audio_q   = queue.Queue(maxsize=10)

        self._vad = VoiceActivityDetector(
            on_event = self._on_vad_event,
        )
        self._stt = WhisperSTT(
            model_size = model_size,
            use_hat    = use_hat,
        )
        self._capture = AudioCaptureLoop(
            vad          = self._vad,
            audio_queue  = self._audio_q,
            device_index = self._device,
        )
        self._speaker = TTSSpeaker(capture_loop=self._capture)
        self._worker  = TranscriptionWorker(
            stt           = self._stt,
            audio_queue   = self._audio_q,
            on_transcript = self._on_transcript,
            capture_loop  = self._capture,
        )

        # State
        self.is_running   = False
        self._status_callbacks: List[Callable[[str], None]] = []

        log.info(
            f"VoiceSystem created — "
            f"STT: {'HAT Whisper' if use_hat else f'whisper-{model_size}'}, "
            f"TTS: {self._speaker._engine.name if self._speaker._engine else 'none'}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, skip_calibration: bool = False):
        """
        Start the full voice pipeline.

        1. Calibrate noise floor (if enabled and microphone available)
        2. Start TTS speaker
        3. Start audio capture loop
        4. Start transcription worker
        5. Speak a startup greeting

        Args:
            skip_calibration: Skip noise floor calibration (faster startup)
        """
        log.info("VoiceSystem starting…")

        # Calibrate noise floor
        if self._calibrate and not skip_calibration and _HAS_SOUNDDEVICE:
            self._vad.calibrate_noise_floor(duration_s=1.5)

        # Start components in order
        self._speaker.start()
        self._capture.start()
        self._worker.start()

        self.is_running = True
        log.info("VoiceSystem started. Always listening.")
        self._emit_status("listening")

    def stop(self):
        """Stop all voice components cleanly."""
        log.info("VoiceSystem stopping…")
        self._worker.stop()
        self._capture.stop()
        self._speaker.stop()
        self.is_running = False
        log.info("VoiceSystem stopped.")
        self._emit_status("stopped")

    # ── Public interface ──────────────────────────────────────────────────────

    def speak(
        self,
        text:    str,
        mind_id: str = "system",
        block:   bool = False,
    ) -> bool:
        """
        Speak text in the voice of the given mind.

        Args:
            text:    Response text to speak
            mind_id: "sophia" | "harmonia" | "elysia" | "system"
            block:   Wait for speech to finish before returning

        Returns:
            True if speech was queued/completed successfully
        """
        if not text:
            return False
        return self._speaker.speak(text, mind_id=mind_id, block=block)

    def stop_speaking(self):
        """Interrupt and clear any current or queued speech."""
        self._speaker.stop_speaking()

    def set_transcript_callback(self, callback: Callable[[str], None]):
        """
        Register the function called when speech is transcribed.
        Usually: lambda text: minds.chat(text)

        Can be changed at runtime (thread-safe — Python assignment is atomic).
        """
        self._callback = callback

    def add_status_callback(self, callback: Callable[[str], None]):
        """
        Register a function that receives status strings.
        Useful for updating a UI label.

        Status values: "listening", "speech_detected", "transcribing",
                       "processing", "speaking", "stopped", "error"
        """
        self._status_callbacks.append(callback)

    def calibrate(self) -> float:
        """
        Re-calibrate the VAD noise floor.
        Returns the measured noise floor RMS.
        Call this if Lumen is triggering too easily (e.g. moved to a noisier room).
        """
        floor = self._vad.calibrate_noise_floor()
        log.info(f"Noise floor recalibrated: {floor:.4f}")
        return floor

    def list_audio_devices(self) -> List[dict]:
        """
        Return available audio input devices as a list of dicts.
        Useful for picking a specific microphone.
        """
        if not _HAS_SOUNDDEVICE:
            return []
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"],
             "in_ch": d["max_input_channels"],
             "out_ch": d["max_output_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    def set_threshold(self, speech: float = None, silence: float = None):
        """
        Adjust VAD thresholds at runtime (no restart needed).
        See tuning guide in module header.
        """
        if speech  is not None: self._vad.speech_threshold  = speech
        if silence is not None: self._vad.silence_threshold = silence
        log.info(
            f"VAD thresholds updated — "
            f"speech={self._vad.speech_threshold:.4f}, "
            f"silence={self._vad.silence_threshold:.4f}"
        )

    def status(self) -> dict:
        """Return current voice system status dict."""
        return {
            "running":         self.is_running,
            "listening":       self._capture.is_running,
            "tts_available":   self._speaker.is_available(),
            "tts_engine":      self._speaker._engine.name if self._speaker._engine else "none",
            "stt_model":       self._stt.model_size,
            "stt_use_hat":     self._stt.use_hat,
            "vad_threshold":   self._vad.speech_threshold,
            "vad_state":       self._vad._state.name,
            "wake_word":       WAKE_WORD_TEXT if WAKE_WORD_ENABLED else "disabled",
            "audio_device":    self._device,
        }

    def probe_tts(self) -> Dict[str, bool]:
        """Return availability of all TTS engines."""
        return TTSEngineFactory.probe_all()

    # ── Internal callbacks ─────────────────────────────────────────────────────

    def _on_vad_event(self, event: VADEvent):
        """Called by VAD on any speech event. Updates status for listeners."""
        if event.event_type == "speech_start":
            self._emit_status("speech_detected")
        elif event.event_type == "speech_end":
            self._emit_status("transcribing")

    def _on_transcript(self, text: str):
        """
        Called by TranscriptionWorker when a transcription is ready.
        Dispatches to the registered callback.
        """
        self._emit_status("processing")
        if self._callback:
            try:
                self._callback(text)
            except Exception as e:
                log.error(f"Transcript callback error: {e}")
        self._emit_status("listening")

    def _emit_status(self, status: str):
        for cb in self._status_callbacks:
            try:
                cb(status)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a float32 numpy array to WAV file bytes (in memory).
    Used for sending audio to the HAT Whisper service.
    """
    import io
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)    # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _save_wav(audio: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE):
    """Save a numpy audio array as a WAV file."""
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def list_microphones() -> List[dict]:
    """List available microphones. Call before instantiating VoiceSystem."""
    if not _HAS_SOUNDDEVICE:
        return []
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"],
         "default_sr": int(d["default_samplerate"])}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: SELF-TEST
# Tests every component without requiring a microphone or TTS system.
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test():
    _R = "\033[0m"; _B = "\033[1m"
    _G = "\033[92m"; _X = "\033[91m"; _Y = "\033[93m"

    print(f"\n{'═'*62}")
    print(f"  {_B}LUMEN NOVA — Voice System Self-Test{_R}")
    print("═"*62 + "\n")

    ok_all = True

    # ── Test 1: VAD energy computation ───────────────────────────────────────
    print(f"▶  {_B}Test 1: VAD energy computation{_R}")
    vad = VoiceActivityDetector()

    # Silent audio
    silent = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    e_silent = vad._compute_energy(silent)
    assert e_silent == 0.0, f"Silent audio energy should be 0, got {e_silent}"

    # Loud audio
    loud = np.ones(CHUNK_SAMPLES, dtype=np.float32) * 0.5
    e_loud = vad._compute_energy(loud)
    assert e_loud > SPEECH_THRESHOLD, f"Loud audio {e_loud} should exceed threshold {SPEECH_THRESHOLD}"

    print(f"  {_G}✓{_R}  Silent energy: {e_silent:.4f}  (expect 0.0)")
    print(f"  {_G}✓{_R}  Loud energy:   {e_loud:.4f}  (expect > {SPEECH_THRESHOLD})")

    # ── Test 2: VAD state machine ─────────────────────────────────────────────
    print(f"\n▶  {_B}Test 2: VAD state machine{_R}")
    events_seen = []
    vad2 = VoiceActivityDetector(
        speech_threshold  = 0.1,
        silence_threshold = 0.05,
        silence_duration  = 0.24,   # 3 chunks at 80ms = 240ms
        min_speech_ms     = 80,
        on_event          = lambda e: events_seen.append(e.event_type),
    )

    # Simulate: silence → speech × 5 chunks → silence × 4 chunks
    silence_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    speech_chunk  = np.ones(CHUNK_SAMPLES, dtype=np.float32) * 0.3

    for _ in range(3):
        vad2.process_chunk(silence_chunk)
    for _ in range(5):
        vad2.process_chunk(speech_chunk)
    for _ in range(4):
        vad2.process_chunk(silence_chunk)

    assert "speech_start" in events_seen, f"Expected speech_start, got: {events_seen}"
    assert "speech_end"   in events_seen, f"Expected speech_end, got: {events_seen}"
    print(f"  {_G}✓{_R}  Events fired: {events_seen}")

    # ── Test 3: Whisper hallucination filter ──────────────────────────────────
    print(f"\n▶  {_B}Test 3: Whisper hallucination filter{_R}")
    stt = WhisperSTT()
    cases = [
        ("thank you",  ""),
        ("you",        ""),
        (".",          ""),
        ("Hello there, Lumen!", "Hello there, Lumen!"),
        ("What is quantum entanglement?", "What is quantum entanglement?"),
    ]
    for inp, expected in cases:
        result = stt._filter_hallucinations(inp)
        ok     = result == expected
        ok_all = ok_all and ok
        status = f"{_G}✓{_R}" if ok else f"{_X}✗{_R}"
        print(f"  {status}  '{inp[:30]}' → '{result[:30]}'  (expect '{expected[:30]}')")

    # ── Test 4: TTS text cleaning ─────────────────────────────────────────────
    print(f"\n▶  {_B}Test 4: TTS text cleaning{_R}")
    engine_dummy = MacOSSayTTS()
    dirty  = "**Bold** and _italic_ with a [link](http://x.com) and `code` here."
    clean  = engine_dummy._clean_text(dirty)
    assert "**" not in clean,  "Bold markers should be removed"
    assert "http" not in clean, "Links should be removed"
    assert "`" not in clean,   "Code backticks should be removed"
    print(f"  {_G}✓{_R}  Input:  {dirty}")
    print(f"  {_G}✓{_R}  Cleaned: {clean}")

    # ── Test 5: TTS text truncation ───────────────────────────────────────────
    print(f"\n▶  {_B}Test 5: TTS text truncation{_R}")
    long_text = "This is a sentence. " * 50   # 1000 chars
    truncated = engine_dummy._truncate(long_text, max_chars=100)
    assert len(truncated) <= 120, f"Truncated text too long: {len(truncated)}"
    print(f"  {_G}✓{_R}  {len(long_text)} chars → {len(truncated)} chars (truncated at sentence boundary)")

    # ── Test 6: TTS engine availability ──────────────────────────────────────
    print(f"\n▶  {_B}Test 6: TTS engine availability{_R}")
    engines = TTSEngineFactory.probe_all()
    best    = TTSEngineFactory.create_best()
    for name, avail in engines.items():
        mark = f"{_G}✓{_R}" if avail else f"{_Y}–{_R}"
        note = " ← selected" if best and best.name == name else ""
        print(f"  {mark}  {name:15} {'available' if avail else 'not available'}{note}")
    if not best:
        print(f"  {_Y}⚠{_R}  No TTS engine available (install espeak: sudo apt install espeak-ng)")

    # ── Test 7: numpy → WAV conversion ────────────────────────────────────────
    print(f"\n▶  {_B}Test 7: numpy → WAV bytes{_R}")
    test_audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    wav_bytes  = _numpy_to_wav_bytes(test_audio, 16000)
    assert wav_bytes[:4] == b"RIFF", "WAV header incorrect"
    assert len(wav_bytes) > 1000, f"WAV too short: {len(wav_bytes)} bytes"
    print(f"  {_G}✓{_R}  {len(test_audio)} samples → {len(wav_bytes)} WAV bytes")
    print(f"  {_G}✓{_R}  WAV header: {wav_bytes[:4].decode()}")

    # ── Test 8: VoiceSystem init (no hardware) ────────────────────────────────
    print(f"\n▶  {_B}Test 8: VoiceSystem initialization (no hardware){_R}")
    received = []
    vs = VoiceSystem(
        profile             = None,
        transcript_callback = lambda t: received.append(t),
        calibrate_on_start  = False,
    )
    status = vs.status()
    assert not status["running"], "Should not be running before start()"
    print(f"  {_G}✓{_R}  VoiceSystem created")
    print(f"  {_G}✓{_R}  STT model: {status['stt_model']}")
    print(f"  {_G}✓{_R}  TTS engine: {status['tts_engine']}")
    print(f"  {_G}✓{_R}  Wake word: {status['wake_word']}")
    print(f"  {_G}✓{_R}  HAT routing: {status['stt_use_hat']}")

    # ── Test 9: Mind voices ───────────────────────────────────────────────────
    print(f"\n▶  {_B}Test 9: Mind voice configurations{_R}")
    for mid, cfg in MIND_VOICES.items():
        assert "macos_voice"  in cfg, f"Missing macos_voice for {mid}"
        assert "espeak_voice" in cfg, f"Missing espeak_voice for {mid}"
        assert "rate"         in cfg, f"Missing rate for {mid}"
        print(f"  {_G}✓{_R}  {mid:10}  mac={cfg['macos_voice']:12} "
              f"espeak={cfg['espeak_voice']:10} rate={cfg['rate']}")

    # ── Test 10: Audio device list ────────────────────────────────────────────
    print(f"\n▶  {_B}Test 10: Audio device enumeration{_R}")
    devs = list_microphones()
    if devs:
        print(f"  {_G}✓{_R}  {len(devs)} microphone(s) found:")
        for d in devs[:3]:
            print(f"     [{d['index']}] {d['name']} ({d['channels']}ch, {d['default_sr']}Hz)")
    else:
        print(f"  {_Y}–{_R}  No microphones found (expected in this environment)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    if ok_all:
        print(f"  {_G}{_B}✓ All Voice System tests passed!{_R}")
    else:
        print(f"  {_X}{_B}✗ Some tests failed — check output above{_R}")
    print("═"*62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: CLI ENTRYPOINT
#
#   python voice_system.py --test
#   python voice_system.py --listen
#   python voice_system.py --speak "Hello, I am Sophia"
#   python voice_system.py --speak "Hello" --mind sophia
#   python voice_system.py --calibrate
#   python voice_system.py --devices
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lumen Nova — Voice System")
    parser.add_argument("--test",      action="store_true",   help="Run self-test")
    parser.add_argument("--listen",    action="store_true",   help="Live always-listening demo")
    parser.add_argument("--speak",     type=str, default=None, help="Speak a line of text")
    parser.add_argument("--mind",      type=str, default="sophia",
                        choices=list(MIND_VOICES.keys()), help="Which mind's voice to use")
    parser.add_argument("--calibrate", action="store_true",   help="Calibrate noise floor")
    parser.add_argument("--devices",   action="store_true",   help="List audio devices")
    parser.add_argument("--model",     type=str, default=WHISPER_MODEL_SIZE,
                        help="Whisper model size (tiny/base/small)")
    args = parser.parse_args()

    if args.test:
        _run_self_test()

    elif args.devices:
        devs = list_microphones()
        if devs:
            print(f"\n  {len(devs)} microphone(s):")
            for d in devs:
                print(f"  [{d['index']}] {d['name']:40} {d['channels']}ch  {d['default_sr']}Hz")
        else:
            print("  No microphones found.")

    elif args.speak:
        speaker = TTSSpeaker()
        speaker.start()
        voice_cfg = MIND_VOICES.get(args.mind, MIND_VOICES["system"])
        print(f"  Speaking as {args.mind}: \"{args.speak}\"")
        speaker.speak(args.speak, mind_id=args.mind, block=True)
        time.sleep(0.5)

    elif args.calibrate or args.listen:
        vs = VoiceSystem(calibrate_on_start=args.calibrate or args.listen)

        if args.calibrate:
            floor = vs._vad.calibrate_noise_floor(duration_s=3.0)
            print(f"\n  Noise floor: {floor:.4f}")
            print(f"  Speech threshold set to: {vs._vad.speech_threshold:.4f}")
            print(f"  Silence threshold set to: {vs._vad.silence_threshold:.4f}\n")

        if args.listen:
            _G = "\033[92m"; _R = "\033[0m"; _Y = "\033[93m"

            heard_texts = []
            def _on_heard(text: str):
                heard_texts.append(text)
                print(f"\n  {_G}Heard:{_R} \"{text}\"")

            vs.set_transcript_callback(_on_heard)
            vs.add_status_callback(
                lambda s: print(f"  [{s}]", end="\r", flush=True)
            )

            print(f"\n  {_Y}Always listening — speak naturally. Ctrl-C to stop.{_R}")
            print(f"  STT: {vs.status()['stt_model']}  |  "
                  f"TTS: {vs.status()['tts_engine']}  |  "
                  f"VAD threshold: {vs._vad.speech_threshold:.4f}\n")

            vs.start(skip_calibration=not args.calibrate)

            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print(f"\n\n  Stopping…")
                vs.stop()
                print(f"  Heard {len(heard_texts)} utterances.")

    else:
        parser.print_help()

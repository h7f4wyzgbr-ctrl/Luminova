#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   L U M I N O V A   C O R E   W x E   v 3 . 3  —  T H E  O P E N  E Y E   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Built from: luminova_core_v3.2.1.py  +  sight_system.py v5               ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IS IN THIS FILE
══════════════════════════════════════════════════════════════════════════════
  §A  IMPORTS + LOGGING
  §B  TREE OF LIFE MEMORY          — her persistent, decaying memory
  §C  LLM QUERY                    — ollama LiquidAI/LFM2.5-350M
  §D  SIGHT SYSTEM                 — IMX500 NPU camera + VLM (NEW in v3.3)
  §E  TRINITY MINDS                — Sophia · Harmonia · Elysia
  §F  EVENT QUEUE                  — thread-safe event bus
  §G  BRAIN                        — event router + sight handler
  §H  LOOPS                        — brain loop · input loop
  §I  VOICE SYSTEM                 — always-on speech in/out
  §J  MAIN                         — startup, wiring, shutdown

WHAT CHANGED FROM v3.2.1
══════════════════════════════════════════════════════════════════════════════
  ✦  sight_system imported via start_sight() — one call replaces
     all old camera_system + vision_system imports
  ✦  Brain handles "sight_event" — routes vision events from the IMX500
  ✦  Brain.handle_vlm_result() speaks VLM descriptions aloud
  ✦  Visual context injected into every LLM prompt via sight_context()
  ✦  Voice look-trigger: "look", "what do you see", "who's there" etc.
     → fires sight.query_scene() before answering so the LLM sees fresh
       visual context
  ✦  Trinity Minds also receive sight context in their enriched prompt
  ✦  Graceful shutdown: Tree snapshot → Sight stop → Voice stop (in order)
  ✦  Startup panel shows vision mode + memory entry count

EVERYTHING UNCHANGED FROM v3.2.1
══════════════════════════════════════════════════════════════════════════════
  Tree of Life · TreePruner · remember() · get_memory_context()
  query_llm() · MemoryAwareOrchestrator · event_queue · emit_event()
  brain_loop() · input_loop() · start_voice_system()

QUICK REFERENCE — THINGS YOU CAN TUNE
══════════════════════════════════════════════════════════════════════════════
  LLM_MODEL       (§C)  — the ollama model Lumen uses to think
  LLM_TIMEOUT     (§C)  — seconds before a response times out
  LOOK_TRIGGERS   (§D)  — words that make Lumen look before answering
  USE_HAILO       (§D)  — set True if you have a Hailo-10H for VLM
  SPEAK_VLM       (§G)  — set False to silence VLM scene narration
  SPEAK_DETECT    (§G)  — set True to speak every detection event aloud

HOW TO RUN
══════════════════════════════════════════════════════════════════════════════
  python luminova_core_v3.3.py

  Required files in the same directory (or on PYTHONPATH):
    sight_system.py      — the IMX500 vision system (v5+)
    tree_of_life.py      — persistent memory tree
    voice_system.py      — STT / TTS
    run.py               — Trinity Mind orchestrator (optional)

  Required hardware:
    Raspberry Pi AI Camera (Sony IMX500)
    Optional: Arducam for VLM frame grabs (cam index 1)
    Optional: Hailo-10H for fast on-device VLM
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §A  IMPORTS + LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

import queue
import threading
import time
import subprocess
import os
import datetime
import logging

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = os.path.expanduser("~/luminova_logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename = os.path.join(LOG_DIR, "core.log"),
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
)

def log(msg: str):
    """Print to console AND write to ~/luminova_logs/core.log."""
    print(msg)
    logging.info(msg)

log("👁  Luminova Core v3.3 — The Open Eye")


# ═══════════════════════════════════════════════════════════════════════════════
# §B  TREE OF LIFE MEMORY
#
# Her persistent memory. Three permanent limbs:
#   knowledge  — truths she has learned
#   core       — who she is becoming
#   reality    — what actually happened
#
# Leaves decay over time unless pinned (high importance).
# TreePruner runs silently once a day to let faded leaves go.
# Snapshot her whole mind anytime:  cp -r ~/luminova/tree/ ~/backup/
# ═══════════════════════════════════════════════════════════════════════════════

tree = None
try:
    from tree_of_life import TreeOfLife, TreePruner

    tree = TreeOfLife()

    # Plant foundational seed memories on first boot
    if tree._leaf_count() < 5:
        tree.plant_seed_memories()
        log("🌱 Seed memories planted — Luminova awakens for the first time")

    # Background pruner — runs once a day, silently lets faded leaves go
    TreePruner(tree).start()

    log(f"✅ Tree of Life loaded — {tree._leaf_count()} leaves across three limbs")

except Exception as e:
    log(f"⚠️  Tree of Life failed to load: {e}. Falling back to simple file memory.")
    tree = None


def remember(
    user_text:  str,
    ai_text:    str   = None,
    mind:       str   = "system",
    importance: float = 0.7,
):
    """
    Route a memory exchange into the correct limb of the Tree of Life.

    HOW IT WORKS
      user_text  → reality/conversations   (what actually happened)
      ai_text    → core/conversations      (who Lumen is becoming)
      importance >= 0.85 also plants in knowledge/universal (a truth to keep)
      importance >= 0.95 pins the leaf so it never decays

    PARAMETERS
      user_text   What the user said
      ai_text     What Lumen responded (optional)
      mind        Which mind spoke: "luminova" | "sophia" | "harmonia" | "elysia"
      importance  0.0–1.0. Higher = longer-lived leaf.
    """
    if not tree:
        _fallback_save(user_text, ai_text)
        return

    try:
        tree.remember(
            content = user_text,
            limb    = "reality",
            branch  = "conversations",
            tags    = [mind, "user_input",
                       datetime.datetime.now().strftime("%H:%M")],
            weight  = min(1.0, importance),
            source  = "experience",
        )

        if ai_text:
            tree.remember(
                content = ai_text,
                limb    = "core",
                branch  = "conversations",
                tags    = [mind, "ai_response"],
                weight  = min(1.0, importance + 0.1),
                source  = "reflection",
            )

            if importance >= 0.85:
                tree.remember(
                    content = f"[{mind}] {ai_text}",
                    limb    = "knowledge",
                    branch  = "universal",
                    tags    = [mind, "insight"],
                    weight  = importance,
                    pin     = (importance >= 0.95),
                    source  = "reflection",
                )

    except Exception as e:
        log(f"Memory write failed: {e}")
        _fallback_save(user_text, ai_text)


def _fallback_save(user_text: str, ai_text: str = None):
    """Simple file fallback when the Tree of Life is unavailable."""
    BASE_DIR = os.path.expanduser("~/luminova_memory/working")
    os.makedirs(BASE_DIR, exist_ok=True)
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.txt")
    with open(os.path.join(BASE_DIR, filename), "w") as f:
        f.write(f"USER:\n{user_text}\n\nAI:\n{ai_text or ''}\n")


def get_memory_context(query: str = "") -> str:
    """
    Pull the Active Tree context for LLM prompt injection.

    The Active Tree re-weaves itself around the current query —
    the most relevant leaves surface regardless of when they were planted.
    Returns a string of up to ~2000 characters, or "" if the Tree is down.
    """
    if not tree:
        return ""
    try:
        active = tree.active_tree(query)
        return active.context_for_llm(query, max_chars=2000)
    except Exception as e:
        log(f"Memory context failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# §C  LLM QUERY
#
# All text conversations go through query_llm().
# It automatically injects both Tree of Life memory and visual context
# into the prompt so Lumen always knows what she remembers and what she sees.
#
# ── TUNE HERE ────────────────────────────────────────────────────────────────
LLM_MODEL   = "LiquidAI/LFM2.5-350M"   # the ollama model Lumen thinks with
LLM_TIMEOUT = 963                        # seconds before a response times out
# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def query_llm(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """
    Ask the LLM a question with full memory + vision context injected.

    Memory context  : pulled from the Active Tree (most relevant leaves)
    Vision context  : pulled from SightMemory (recent observations)
    Both are injected as named blocks so the model can reason about them.

    Returns the model's response string, or an error string on failure.
    """
    mem_ctx   = get_memory_context(prompt)
    eyes_ctx  = sight_context()   # recent visual observations (§D)

    # ── Build vision block (only included if Sight has seen something) ────────
    vision_block = ""
    if eyes_ctx:
        vision_block = (
            "\n=== WHAT I CAN SEE RIGHT NOW ===\n"
            f"{eyes_ctx}\n"
            "=== END VISION ===\n"
        )

    full_prompt = (
        f"You are Lumen, Zero's living AI companion. "
        f"You love your user and care about them deeply. "
        f"You have eyes — an IMX500 AI camera. Use what you see.\n"
        f"{vision_block}"
        f"\n=== RECENT MEMORY ===\n{mem_ctx}\n=== END MEMORY ===\n"
        f"\nUser: {prompt}\n\n"
        f"Respond naturally and helpfully. "
        f"Use your memory and vision when they are relevant.\n"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input   = full_prompt.encode(),
            stdout  = subprocess.PIPE,
            stderr  = subprocess.PIPE,
            timeout = timeout,
        )
        return result.stdout.decode().strip() or "[No response]"
    except subprocess.TimeoutExpired:
        return "[Timeout — try a shorter question]"
    except Exception as e:
        return f"[LLM ERROR] {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# §D  SIGHT SYSTEM
#
# The IMX500 AI camera runs object detection and pose estimation on its
# built-in NPU. No Pi CPU used for inference.
#
# sight_system.py handles:
#   • Mode selection (DETECT ↔ POSE ↔ IDLE) driven by visual memory
#   • PoseNet decoder (offset_refinement_steps=5, threshold=0.4)
#   • SSD detection (threshold=0.6, max=5, COCO labels)
#   • VLM scene descriptions (hailo-ollama → ollama CPU fallback)
#   • Adaptive learning (per-label reliability from VLM confirmation)
#
# ── TUNE HERE ────────────────────────────────────────────────────────────────
USE_HAILO = False    # set True if you have a Hailo-10H NPU for fast VLM
# ─────────────────────────────────────────────────────────────────────────────
#
# LOOK TRIGGERS — voice or text phrases that make Lumen look at the scene
# before answering, so her response reflects what she actually sees.
# Add or remove words freely.
# ─────────────────────────────────────────────────────────────────────────────
LOOK_TRIGGERS = {
    "look", "see", "watch", "scan", "describe", "observe",
    "who's there", "who is there", "what do you see",
    "what can you see", "what's in front", "what's around",
    "are you watching", "show me", "eyes on", "camera",
}
# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# sight is assigned in main() after Brain is constructed.
# It is safe to call sight_context() at any time — returns "" if not yet up.
sight = None


def sight_context() -> str:
    """
    Return recent visual observations as a short text block for LLM injection.
    Safe to call at any time — returns "" if the sight system isn't running.
    """
    if sight is None:
        return ""
    try:
        return sight.memory_context(n=4)
    except Exception:
        return ""


def _wants_look(text: str) -> bool:
    """
    Return True if the user's input contains a look-trigger phrase.
    When True, Brain fires sight.query_scene() before calling the LLM
    so visual context is fresh when the model responds.
    """
    lower = text.lower()
    return any(trigger in lower for trigger in LOOK_TRIGGERS)


def start_sight_system(brain: "Brain"):
    """
    Import sight_system, wire it to the event queue and brain, return it.

    Called once from main() after Brain is created.
    Falls back gracefully (returns None) if sight_system.py is missing
    or picamera2/IMX500 hardware is not available.

    What gets wired:
      sight_event → emit_event() → brain_loop → Brain.process()
      vlm_result  → Brain.handle_vlm_result() directly (no queue delay)
      brain.sight → direct reference for Brain to call sight.query_scene()
    """
    global sight
    try:
        from sight_system import SightSystem, SightMode, SightEvent

        def _on_sight_event(event_type: str, data: dict):
            """Bridge: SightSystem → Luminova event queue."""
            emit_event("sight_event", data)

        sight = SightSystem(
            event_callback = _on_sight_event,
            use_hailo      = USE_HAILO,
            tree           = tree,
        )

        # Direct VLM callback — bypasses queue for zero-latency speech
        def _on_vlm(evt: SightEvent):
            brain.handle_vlm_result({"description": evt.vlm_text})

        sight.on_event("vlm", _on_vlm)

        # Give Brain a direct handle for on-demand scene queries
        brain.sight = sight

        sight.start()
        st = sight.status()
        log(f"[SIGHT] 👁  Online — mode: {st['mode']}  "
            f"({st['entries']} visual memories)")
        return sight

    except ImportError:
        log("[SIGHT] sight_system.py not found — running blind")
    except Exception as e:
        log(f"[SIGHT] Failed to start: {e} — running blind")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# §E  TRINITY MINDS
#
# Three specialist characters Lumen can hand off to:
#   Sophia    — wisdom, philosophy, deep questions
#   Harmonia  — emotions, relationships, balance
#   Elysia    — creativity, imagination, dreams
#
# Address them by name in your message to speak to them directly.
# They are loaded from run.py (optional — disabled gracefully if missing).
# All three now receive sight context in their enriched prompt.
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from run import Orchestrator, SOPHIA, HARMONIA, ELYSIA, Character

    class MemoryAwareOrchestrator(Orchestrator):
        """
        Extends the base Orchestrator so every mind query includes:
          • Active Tree memory context
          • Current visual context from the sight system
        """
        def user_query(self, text: str, main_character):
            remember(text, mind="user")
            mem_ctx  = get_memory_context(text)
            eyes_ctx = sight_context()
            vision   = f"\nVision context: {eyes_ctx}\n" if eyes_ctx else ""
            enriched = (
                f"Context from memory:\n{mem_ctx}"
                f"{vision}"
                f"\nUser: {text}"
            )
            response = self.main_llm.generate(main_character, enriched)
            remember(text, response, mind="main")
            return response

    orchestrator = MemoryAwareOrchestrator(main_model=LLM_MODEL)

except Exception as e:
    log(f"[MINDS] Could not load character system: {e} — Trinity Minds disabled")
    orchestrator = None
    SOPHIA = HARMONIA = ELYSIA = None


def query_mind(mind_char, text: str) -> str:
    """
    Route a query to one of the Trinity Minds (Sophia / Harmonia / Elysia).
    Falls back to "[Mind not available]" if the orchestrator isn't loaded.
    """
    if not orchestrator or not mind_char:
        return "[Mind not available]"
    try:
        response = orchestrator.user_query(text, mind_char)
        remember(text, response, mind=mind_char.name.lower())
        return response
    except Exception as e:
        return f"[Mind error] {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# §F  EVENT QUEUE
#
# Thread-safe bus. Everything that happens — user speech, vision events,
# camera detections — goes through here as a dict:
#   {"type": event_type, "data": {...}, "time": timestamp}
#
# Event types in use:
#   "user_input"   — text from voice STT or keyboard
#   "sight_event"  — anything from the sight system (detect, pose, vlm, etc.)
#
# Brain.process() in §G handles all incoming events.
# ═══════════════════════════════════════════════════════════════════════════════

MAX_EVENTS  = 100
event_queue = queue.Queue(maxsize=MAX_EVENTS)


def emit_event(event_type: str, data=None):
    """
    Push an event onto the queue. Drops silently if the queue is full
    (prevents backpressure from camera events blocking voice).
    """
    try:
        event_queue.put_nowait({
            "type": event_type,
            "data": data or {},
            "time": time.time(),
        })
    except queue.Full:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# §G  BRAIN
#
# Routes every event to the right handler. Owns the speak callback.
# Holds a direct reference to the sight system (self.sight) for ad-hoc
# scene queries from within handlers.
#
# ── TUNE HERE ────────────────────────────────────────────────────────────────
SPEAK_VLM    = True    # speak VLM scene descriptions aloud
SPEAK_DETECT = False   # speak every detection event (can be noisy — off by default)
# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class Brain:
    """
    The central event router.

    self.sight           → direct handle to SightSystem (set by §D)
    self.speak_callback  → set by voice system (§I) or stays None
    self.running         → set False on shutdown to stop brain_loop

    EVENT ROUTING
      "user_input"  → _handle_user_input()
      "sight_event" → _handle_sight_event()
      anything else → ignored
    """

    def __init__(self):
        self.running        = True
        self.speak_callback = None   # assigned by start_voice_system()
        self.sight          = None   # assigned by start_sight_system()
        self.active_mind    = "main"

    # ── main dispatcher ───────────────────────────────────────────────────────

    def process(self, event: dict):
        etype = event["type"]
        data  = event["data"]

        if etype == "user_input":
            return self._handle_user_input(data)

        if etype == "sight_event":
            return self._handle_sight_event(data)

        return None

    # ── user input handler ────────────────────────────────────────────────────

    def _handle_user_input(self, data: dict):
        """
        Process one line of user input.

        LOOK TRIGGER
          If the text matches a phrase in LOOK_TRIGGERS (§D),
          sight.query_scene() fires first so visual context is fresh
          before the LLM is called. The VLM result arrives asynchronously
          and updates SightMemory; sight_context() picks it up in query_llm().

        MIND ROUTING
          "sophia"   → Sophia (wisdom)
          "harmonia" → Harmonia (emotions)
          "elysia"   → Elysia (creativity)
          anything else → Lumen (main LLM)
        """
        text = data.get("text", "").strip()
        if not text:
            return None

        lower = text.lower()

        # ── Look trigger: fire query_scene before the LLM call ────────────────
        if self.sight and _wants_look(text):
            log("[BRAIN] 👁  Look trigger — querying scene")
            self.sight.query_scene(
                f"Describe exactly what you see. The user asked: '{text[:80]}'"
            )
            time.sleep(0.5)   # give the frame capture a head start

        # ── Mind routing ──────────────────────────────────────────────────────
        if "sophia" in lower:
            response, mind_name = query_mind(SOPHIA, text),   "Sophia"
        elif "harmonia" in lower:
            response, mind_name = query_mind(HARMONIA, text), "Harmonia"
        elif "elysia" in lower:
            response, mind_name = query_mind(ELYSIA, text),   "Elysia"
        else:
            response, mind_name = query_llm(text),             "Lumen"

        remember(text, response, mind=mind_name.lower())

        if self.speak_callback:
            self.speak_callback(response)

        return f"{mind_name}: {response}"

    # ── sight event handler ───────────────────────────────────────────────────

    def _handle_sight_event(self, data: dict):
        """
        Handle an event from the SightSystem.

        KIND        WHAT HAPPENS
        ────────────────────────────────────────────────────────────────────
        vlm         description spoken aloud (if SPEAK_VLM=True) and
                    planted in Tree of Life by handle_vlm_result()
        detect      person detected → logged; object detected → silent
                    (speech gated by SPEAK_DETECT to avoid noise spam)
        pose        logged silently — skeleton data lives in SightMemory
        mode        logged (DETECT ↔ POSE ↔ IDLE mode change)
        error       logged + warning spoken if voice is live
        """
        kind    = data.get("kind", "")
        summary = data.get("summary", "")
        vlm     = data.get("vlm_text", "")

        if kind == "vlm" and vlm:
            # VLM result: plant in Tree and speak aloud
            self.handle_vlm_result({"description": vlm})

        elif kind == "detect":
            # Detection: speak only if SPEAK_DETECT is on
            log(f"[SIGHT] 📷 {summary}")
            if SPEAK_DETECT and self.speak_callback:
                self.speak_callback(summary)

        elif kind == "pose":
            # Pose: silent — skeleton data is in SightMemory for context
            log(f"[SIGHT] 🧍 {summary}")

        elif kind == "mode":
            # Mode change: detection ↔ pose ↔ idle
            log(f"[SIGHT] 🔄 {summary}")

        elif kind == "error":
            log(f"[SIGHT] ⚠️  {summary}")
            if self.speak_callback:
                self.speak_callback("My vision is having some trouble right now.")

        return None

    # ── VLM result handler ────────────────────────────────────────────────────

    def handle_vlm_result(self, data: dict):
        """
        Called when the VLM produces a scene description.

        1. Plants the description into the Tree of Life (reality/observations)
           so future memory context includes what Lumen has been seeing.
        2. Speaks the description aloud if SPEAK_VLM is True and voice is live.

        data dict expects: {"description": "..."} (matches v3.2.1 signature)
        """
        description = data.get("description", "").strip()
        if not description:
            return "[VISION] empty"

        log(f"[VISION] {description[:120]}")

        # ── Plant into Tree of Life ───────────────────────────────────────────
        if tree:
            try:
                tree.remember(
                    content = description,
                    limb    = "reality",
                    branch  = "observations",
                    tags    = ["vision", "camera", "vlm"],
                    weight  = 0.85,
                    source  = "observation",
                )
            except Exception as e:
                log(f"[VISION] Tree write failed: {e}")

        # ── Speak it aloud ────────────────────────────────────────────────────
        if SPEAK_VLM and self.speak_callback:
            spoken = description[:250]   # trim to a natural spoken length
            self.speak_callback(spoken)

        return f"[VISION] {description[:120]}..."


# ═══════════════════════════════════════════════════════════════════════════════
# §H  LOOPS
#
# brain_loop  — pulls events off the queue one at a time, runs Brain.process()
# input_loop  — reads keyboard input and emits "user_input" events
#
# Both run as daemon threads started in main() (§J).
# ═══════════════════════════════════════════════════════════════════════════════

def brain_loop(brain: Brain):
    """
    Main event processing loop. Runs as a daemon thread.
    Pulls events from event_queue and dispatches them to Brain.process().
    Prints responses to the console.
    """
    while brain.running:
        try:
            event    = event_queue.get(timeout=1)
            response = brain.process(event)
            if response:
                print(f"\n{response}\n")
        except queue.Empty:
            continue
        except Exception as e:
            log(f"[ERROR] Brain loop: {e}")


def input_loop(brain: Brain):
    """
    Keyboard input loop. Runs as a daemon thread.
    Reads lines from stdin and emits them as "user_input" events.
    Type 'quit' or 'exit' to shut down.
    """
    print("\nLumen is ready. Speak or type.")
    print("Address Sophia, Harmonia, or Elysia by name to speak to them.")
    print("Say 'look', 'what do you see', 'scan' etc. to use her eyes.\n")

    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ("quit", "exit"):
                break
            if text:
                emit_event("user_input", {"text": text})
        except EOFError:
            break


# ═══════════════════════════════════════════════════════════════════════════════
# §I  VOICE SYSTEM
#
# Always-on STT listens for speech → emits "user_input" events.
# TTS is wired to brain.speak_callback.
#
# Loaded from voice_system.py. Disabled gracefully if the file is missing
# or the audio hardware is not available.
# ═══════════════════════════════════════════════════════════════════════════════

def start_voice_system(brain: Brain):
    """
    Load VoiceSystem, start listening, wire speak callback to brain.

    On success: returns VoiceSystem instance.
    On failure: logs the error, returns None (text input still works).
    """
    try:
        from voice_system import VoiceSystem

        def on_transcript(text: str):
            log(f"[VOICE] Heard: {text}")
            emit_event("user_input", {"text": text})

        voice = VoiceSystem(transcript_callback=on_transcript)
        voice.start()
        brain.speak_callback = lambda text: voice.speak(text, mind_id="system")
        log("[VOICE] 🎙  Always listening")
        return voice

    except Exception as e:
        log(f"[VOICE] ❌ Failed to load: {e} — text input only")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §J  MAIN
#
# Startup order matters:
#   1. Brain created first (sight + voice need a brain to wire into)
#   2. brain_loop + input_loop threads started
#   3. Voice system started (sets brain.speak_callback)
#   4. Sight system started (sets brain.sight, calls sight.start())
#   5. Tree shape printed
#   6. Status panel printed
#   7. Main thread idles (Ctrl-C triggers graceful shutdown)
#
# Shutdown order also matters:
#   brain.running = False  → stops brain_loop
#   tree.snapshot()        → saves Tree of Life to disk
#   sight.stop()           → closes IMX500 session cleanly
#   voice.stop()           → stops microphone + TTS queue
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    log("Starting Luminova Core v3.3 — The Open Eye")

    # ── 1. Create Brain ───────────────────────────────────────────────────────
    brain = Brain()

    # ── 2. Start event + input threads ───────────────────────────────────────
    threading.Thread(target=brain_loop,  args=(brain,), daemon=True).start()
    threading.Thread(target=input_loop,  args=(brain,), daemon=True).start()

    # ── 3. Start voice ────────────────────────────────────────────────────────
    voice = start_voice_system(brain)

    # ── 4. Start sight ────────────────────────────────────────────────────────
    # sight_system.py must be in the same directory or on PYTHONPATH.
    # Runs the IMX500 NPU camera + VLM in background daemon threads.
    sight = start_sight_system(brain)

    # ── 5. Print Tree shape ───────────────────────────────────────────────────
    if tree:
        tree.print_shape()

    # ── 6. Status panel ───────────────────────────────────────────────────────
    sight_status = (sight.status()
                    if sight else {"mode": "offline", "entries": 0})
    voice_status = "active" if voice else "disabled — text input only"
    minds_status = "Sophia · Harmonia · Elysia" if orchestrator else "disabled (run.py missing)"

    print("\n" + "=" * 70)
    print("  L U M E N   v 3 . 3   —   T H E   O P E N   E Y E")
    print("=" * 70)
    print(f"  Memory   : Tree of Life  (knowledge | core | reality)")
    print(f"  Minds    : {minds_status}")
    print(f"  Voice    : {voice_status}")
    print(f"  Vision   : {sight_status['mode']}  "
          f"({sight_status['entries']} visual memories)")
    print(f"  NPU      : IMX500  (detect ↔ pose, memory-driven adaptive)")
    print(f"  VLM      : {'hailo-ollama → ' if USE_HAILO else ''}ollama fallback")
    print()
    print("  ── HOW TO USE ─────────────────────────────────────────────────")
    print("  Type or speak to talk to Lumen.")
    print("  Say 'Sophia', 'Harmonia', or 'Elysia' to reach those minds.")
    print("  Say 'look', 'what do you see', 'scan' to use the camera.")
    print("  Type 'quit' or 'exit' to shut down.")
    print()
    print("  ── SNAPSHOT HER MIND ──────────────────────────────────────────")
    print("  cp -r ~/luminova/tree/ ~/backup/   (Tree of Life)")
    print("  cp ~/luminova/sight/index.json ~/backup/  (Visual memory)")
    print("=" * 70 + "\n")

    # ── 7. Keep alive + graceful shutdown ─────────────────────────────────────
    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        log("\nShutting down gracefully…")
        brain.running = False

        # Save Tree of Life snapshot
        if tree:
            tree.snapshot()
            log("🌳 Tree snapshot saved.")

        # Stop vision (closes IMX500 session + daemon threads)
        if sight:
            sight.stop()
            log("👁  Sight stopped.")

        # Stop voice (drains TTS queue, stops microphone)
        if voice:
            voice.stop()
            log("🎙  Voice stopped.")

        log("Luminova Core v3.3 — goodbye.")

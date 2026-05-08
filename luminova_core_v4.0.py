#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L U M I N O V A   C O R E   W x E   v 4 . 0 — T H E  G R O W I N G  M I N D ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Built from: luminova_core_v3.3.py                                         ║
║  Replaces:   §E Trinity Minds → §E Knowledge Tree                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IS IN THIS FILE
══════════════════════════════════════════════════════════════════════════════
§A  IMPORTS + LOGGING
§B  TREE OF LIFE MEMORY — her persistent, decaying personal memory
§C  LLM QUERY — ollama LiquidAI/LFM2.5-350M
§D  SIGHT SYSTEM — IMX500 NPU camera + VLM
§E  KNOWLEDGE TREE — growing domain knowledge: Math · Science · Art  ← NEW
§F  EVENT QUEUE — thread-safe event bus
§G  BRAIN — event router + sight handler
§H  LOOPS — brain loop · input loop
§I  VOICE SYSTEM — always-on speech in/out
§J  MAIN — startup, wiring, shutdown

WHAT CHANGED FROM v3.3
══════════════════════════════════════════════════════════════════════════════
✦ §E completely replaced: Trinity Minds (Sophia/Harmonia/Elysia, run.py)
  → Knowledge Tree (knowledge_tree.py)

✦ Knowledge Tree: three growing domains — Mathematics · Science · Art
  Each domain has permanent seed branches that grow with every conversation.
  Knowledge decays slowly (0.015/day). Pinned seeds never decay.

✦ Domain routing replaces name-keyword routing in Brain._handle_user_input()
  Old: "sophia" → Sophia, "harmonia" → Harmonia, "elysia" → Elysia
  New: auto-detect math/science/art domain from query content
  Multi-domain queries (e.g. music theory) get context from both domains.

✦ query_llm() now accepts domain_context + domain_name parameters.
  Injects a named domain knowledge block above the personal memory block
  so the LLM can see: vision → domain knowledge → personal memory → question.

✦ After every response, grow() plants two leaves:
  • The user's question → "questions" branch (retrieval anchor)
  • The LLM response lead → best-matching branch (growing knowledge)

✦ KnowledgePruner background thread decays + prunes once per day.

✦ Startup panel updated — shows leaf counts per domain.

✦ run.py dependency completely removed. No optional import, no fallback.

EVERYTHING UNCHANGED FROM v3.3
══════════════════════════════════════════════════════════════════════════════
Tree of Life · TreePruner · remember() · get_memory_context()
sight_system · SightSystem · start_sight_system() · sight_context()
event_queue · emit_event() · Brain · brain_loop() · input_loop()
start_voice_system()

QUICK REFERENCE — THINGS YOU CAN TUNE
══════════════════════════════════════════════════════════════════════════════
LLM_MODEL (§C)         — the ollama model Lumen uses to think
LLM_TIMEOUT (§C)       — seconds before a response times out
LOOK_TRIGGERS (§D)     — words that make Lumen look before answering
USE_HAILO (§D)         — set True if you have a Hailo-10H for VLM
SPEAK_VLM (§G)         — set False to silence VLM scene narration
SPEAK_DETECT (§G)      — set True to speak every detection event aloud

In knowledge_tree.py:
ROUTE_THRESHOLD        — minimum domain score to inject knowledge context
MULTI_DOMAIN_RATIO     — how close second domain must be to also be included
DECAY_RATE_PER_DAY     — how fast knowledge leaves fade (default 0.015)
CONTEXT_MAX_CHARS      — per-domain knowledge block character budget

HOW TO RUN
══════════════════════════════════════════════════════════════════════════════
python luminova_core_v4.0.py

Required files in the same directory (or on PYTHONPATH):
  knowledge_tree.py  — growing domain knowledge tree (NEW)
  sight_system.py    — the IMX500 vision system (v5+)
  tree_of_life.py    — persistent personal memory tree
  voice_system.py    — STT / TTS

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

log("📚 Luminova Core v4.0 — The Growing Mind")

# ═══════════════════════════════════════════════════════════════════════════════
# §B  TREE OF LIFE MEMORY
#
# Her persistent personal memory. Three permanent limbs:
#   knowledge — truths she has learned
#   core      — who she is becoming
#   reality   — what actually happened
#
# Leaves decay over time unless pinned (high importance).
# TreePruner runs silently once a day to let faded leaves go.
# Snapshot her whole mind anytime: cp -r ~/luminova/tree/ ~/backup/
# ═══════════════════════════════════════════════════════════════════════════════

tree = None
try:
    from tree_of_life import TreeOfLife, TreePruner
    tree = TreeOfLife()
    if tree._leaf_count() < 5:
        tree.plant_seed_memories()
        log("🌱 Seed memories planted — Luminova awakens for the first time")
    TreePruner(tree).start()
    log(f"✅ Tree of Life loaded — {tree._leaf_count()} leaves across three limbs")
except Exception as e:
    log(f"⚠️  Tree of Life failed to load: {e}. Falling back to simple file memory.")
    tree = None


def remember(
    user_text: str,
    ai_text:   str   = None,
    mind:      str   = "system",
    importance: float = 0.7,
):
    """
    Route a memory exchange into the Tree of Life.

    user_text  → reality/conversations
    ai_text    → core/conversations
    importance >= 0.85 also plants in knowledge/universal
    importance >= 0.95 pins the leaf (never decays)
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
    BASE_DIR = os.path.expanduser("~/luminova_memory/working")
    os.makedirs(BASE_DIR, exist_ok=True)
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.txt")
    with open(os.path.join(BASE_DIR, filename), "w") as f:
        f.write(f"USER:\n{user_text}\n\nAI:\n{ai_text or ''}\n")


def get_memory_context(query: str = "") -> str:
    """
    Pull the Active Tree context for LLM prompt injection.
    Returns up to ~2000 characters, or "" if the Tree is down.
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
# It automatically injects vision context, domain knowledge context,
# and personal memory into the prompt — in that order.
#
# Prompt injection order (most-immediate → most-background):
#   1. Vision:   what Lumen can see right now
#   2. Domain:   relevant Math / Science / Art knowledge (if query triggers a domain)
#   3. Memory:   personal episodic memory from the Tree of Life
#   4. Question: the user's actual input
#
# ── TUNE HERE ────────────────────────────────────────────────────────────────
LLM_MODEL   = "LiquidAI/LFM2.5-350M"   # the ollama model Lumen thinks with
LLM_TIMEOUT = 963                        # seconds before a response times out
# ─────────────────────────────────────────────────────────────────────────────

def query_llm(
    prompt:         str,
    timeout:        int  = LLM_TIMEOUT,
    domain_context: str  = "",
    domain_name:    str  = "",
) -> str:
    """
    Ask the LLM a question with full context injected.

    Injection order
    ───────────────
    1. Vision block   — what the IMX500 can see right now
    2. Domain block   — growing Math / Science / Art knowledge (NEW v4.0)
    3. Memory block   — personal episodic memory from Tree of Life
    4. User question

    Parameters
    ──────────
    prompt          The user's question.
    timeout         Subprocess timeout in seconds.
    domain_context  Knowledge context string from KnowledgeTree.context_for().
                    Empty string → block omitted entirely.
    domain_name     Human label for the domain block header ("math" etc.).
                    Used only for the section heading in the prompt.

    Returns the model's response string, or an error string on failure.
    """
    mem_ctx  = get_memory_context(prompt)
    eyes_ctx = sight_context()

    # ── Vision block ──────────────────────────────────────────────────────────
    vision_block = ""
    if eyes_ctx:
        vision_block = (
            "\n=== WHAT I CAN SEE RIGHT NOW ===\n"
            f"{eyes_ctx}\n"
            "=== END VISION ===\n"
        )

    # ── Domain knowledge block (new in v4.0) ──────────────────────────────────
    domain_block = ""
    if domain_context:
        label = domain_name.upper() if domain_name else "DOMAIN"
        domain_block = (
            f"\n=== {label} KNOWLEDGE ===\n"
            f"{domain_context}\n"
            f"=== END {label} KNOWLEDGE ===\n"
        )

    full_prompt = (
        f"You are Lumen, Zero's living AI companion. "
        f"You love your user and care about them deeply. "
        f"You have eyes — an IMX500 AI camera. Use what you see.\n"
        f"{vision_block}"
        f"{domain_block}"
        f"\n=== RECENT MEMORY ===\n{mem_ctx}\n=== END MEMORY ===\n"
        f"\nUser: {prompt}\n\n"
        f"Respond naturally and helpfully. "
        f"Use your memory, knowledge, and vision when they are relevant.\n"
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

# LOOK TRIGGERS — voice or text phrases that make Lumen look at the scene
# before answering. Add or remove words freely.
LOOK_TRIGGERS = {
    "look", "see", "watch", "scan", "describe", "observe",
    "who's there", "who is there", "what do you see",
    "what can you see", "what's in front", "what's around",
    "are you watching", "show me", "eyes on", "camera",
}

# sight is assigned in main() after Brain is constructed.
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
    """Return True if the user's input contains a look-trigger phrase."""
    lower = text.lower()
    return any(trigger in lower for trigger in LOOK_TRIGGERS)

def start_sight_system(brain: "Brain"):
    """
    Import sight_system, wire it to the event queue and brain, return it.
    Falls back gracefully (returns None) if hardware is not available.
    """
    global sight
    try:
        from sight_system import SightSystem, SightMode, SightEvent

        def _on_sight_event(event_type: str, data: dict):
            emit_event("sight_event", data)

        sight = SightSystem(
            event_callback = _on_sight_event,
            use_hailo      = USE_HAILO,
            tree           = tree,
        )

        def _on_vlm(evt: SightEvent):
            brain.handle_vlm_result({"description": evt.vlm_text})

        sight.on_event("vlm", _on_vlm)
        brain.sight = sight
        sight.start()

        st = sight.status()
        log(f"[SIGHT] 👁 Online — mode: {st['mode']} "
            f"({st['entries']} visual memories)")
        return sight

    except ImportError:
        log("[SIGHT] sight_system.py not found — running blind")
    except Exception as e:
        log(f"[SIGHT] Failed to start: {e} — running blind")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# §E  KNOWLEDGE TREE
#
# Luminova's growing body of domain knowledge. Three domains:
#   Mathematics — algebra · calculus · geometry · statistics · logic
#   Science     — physics · chemistry · biology · astronomy · CS
#   Art         — music · visual art · literature · film · design
#
# How it works in conversation
# ────────────────────────────
#   1. Brain receives a user query.
#   2. KnowledgeTree.context_for(query) routes it to the matching domain(s)
#      and returns the most relevant knowledge leaves as a context block.
#   3. The context block is injected into the LLM prompt (§C) above personal
#      memory, so the model can draw on accumulated domain knowledge.
#   4. After the response, KnowledgeTree.grow() plants two new leaves:
#      • the user's question (retrieval anchor in "questions" branch)
#      • the response lead (knowledge leaf in best-matching branch)
#   5. KnowledgePruner decays leaves slowly (0.015/day) in the background.
#
# Non-domain queries (greetings, personal, sight, etc.) receive a None domain
# → no domain block injected → the model uses only personal memory + vision.
#
# Persistence: ~/luminova/knowledge/{math,science,art}.json
# Snapshot: cp -r ~/luminova/knowledge/ ~/backup/
# ═══════════════════════════════════════════════════════════════════════════════

ktree = None
try:
    from knowledge_tree import KnowledgeTree, KnowledgePruner
    ktree = KnowledgeTree()
    KnowledgePruner(ktree).start()
    s = ktree.stats()
    log(f"✅ Knowledge Tree loaded — "
        f"math: {s['domains']['math']['total_leaves']} | "
        f"science: {s['domains']['science']['total_leaves']} | "
        f"art: {s['domains']['art']['total_leaves']} leaves")
except Exception as e:
    log(f"⚠️  Knowledge Tree failed to load: {e}. Domain knowledge disabled.")
    ktree = None


def get_domain_context(query: str):
    """
    Route the query to the Knowledge Tree and return (domain, context_str).

    Returns (None, "") if:
      • Knowledge Tree is not loaded
      • No domain scores above ROUTE_THRESHOLD
      • The query is too short to route reliably

    Returns (domain_name, context_str) where:
      domain_name   "math" | "science" | "art"
      context_str   Ready-to-inject knowledge block. Empty if no context found.
    """
    if not ktree or not query or len(query.strip()) < 4:
        return None, ""
    try:
        return ktree.context_for(query)
    except Exception as e:
        log(f"[KTREE] context_for failed: {e}")
        return None, ""


def grow_knowledge(query: str, response: str, domain):
    """
    Grow the Knowledge Tree from a completed interaction.
    Safe to call with domain=None — returns 0 leaves in that case.
    """
    if not ktree or not query or not response:
        return 0
    try:
        return ktree.grow(query, response, domain)
    except Exception as e:
        log(f"[KTREE] grow failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# §F  EVENT QUEUE
#
# Thread-safe bus. Everything that happens — user speech, vision events —
# goes through here as a dict:
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

class Brain:
    """
    The central event router.

    self.sight           → direct handle to SightSystem (set by §D)
    self.speak_callback  → set by voice system (§I) or stays None
    self.running         → set False on shutdown to stop brain_loop

    EVENT ROUTING
    "user_input"   → _handle_user_input()
    "sight_event"  → _handle_sight_event()
    anything else  → ignored
    """

    def __init__(self):
        self.running         = True
        self.speak_callback  = None   # assigned by start_voice_system()
        self.sight           = None   # assigned by start_sight_system()

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
        If the text matches a phrase in LOOK_TRIGGERS (§D), sight.query_scene()
        fires first so visual context is fresh before the LLM is called.

        DOMAIN ROUTING (replaces Trinity Minds from v3.3)
        KnowledgeTree.context_for() scores the query against Math / Science / Art
        keyword sets. If a domain matches above ROUTE_THRESHOLD, its knowledge
        context is injected into the LLM prompt via query_llm(domain_context=…).
        If no domain matches, the call proceeds with personal memory only.

        KNOWLEDGE GROWTH
        After every response, grow_knowledge() plants two leaves into the matched
        domain so Luminova's knowledge deepens with every conversation.

        MEMORY
        Every exchange is planted in the Tree of Life regardless of domain.
        """
        text = data.get("text", "").strip()
        if not text:
            return None

        # ── Look trigger: fire query_scene before the LLM call ───────────────
        if self.sight and _wants_look(text):
            log("[BRAIN] 👁 Look trigger — querying scene")
            self.sight.query_scene(
                f"Describe exactly what you see. "
                f"The user asked: '{text[:80]}'"
            )
            time.sleep(0.5)

        # ── Domain routing (new in v4.0) ──────────────────────────────────────
        domain, domain_ctx = get_domain_context(text)

        if domain:
            log(f"[BRAIN] 📚 Domain: {domain} — injecting knowledge context")

        # ── LLM call with domain knowledge injected ───────────────────────────
        response = query_llm(
            text,
            domain_context = domain_ctx,
            domain_name    = domain or "",
        )

        # ── Grow knowledge tree from this exchange ────────────────────────────
        leaves_planted = grow_knowledge(text, response, domain)
        if leaves_planted:
            log(f"[BRAIN] 🌿 Grew {leaves_planted} knowledge leaves "
                f"({domain})")

        # ── Plant in Tree of Life (personal episodic memory) ──────────────────
        remember(text, response, mind="lumen")

        # ── Speak ─────────────────────────────────────────────────────────────
        if self.speak_callback:
            self.speak_callback(response)

        return f"Lumen: {response}"

    # ── sight event handler ───────────────────────────────────────────────────

    def _handle_sight_event(self, data: dict):
        """
        Handle an event from the SightSystem.

        KIND        WHAT HAPPENS
        ────────────────────────────────────────────────────────────────────
        vlm         description spoken aloud (SPEAK_VLM=True) and planted
                    in Tree of Life by handle_vlm_result()
        detect      person detected → logged; object → silent
                    (gated by SPEAK_DETECT to avoid noise spam)
        pose        logged silently — skeleton data lives in SightMemory
        mode        logged (DETECT ↔ POSE ↔ IDLE mode change)
        error       logged + warning spoken if voice is live
        """
        kind    = data.get("kind", "")
        summary = data.get("summary", "")
        vlm     = data.get("vlm_text", "")

        if kind == "vlm" and vlm:
            self.handle_vlm_result({"description": vlm})
        elif kind == "detect":
            log(f"[SIGHT] 📷 {summary}")
            if SPEAK_DETECT and self.speak_callback:
                self.speak_callback(summary)
        elif kind == "pose":
            log(f"[SIGHT] 🧍 {summary}")
        elif kind == "mode":
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
        Plants into Tree of Life (reality/observations) and speaks if SPEAK_VLM.
        """
        description = data.get("description", "").strip()
        if not description:
            return "[VISION] empty"

        log(f"[VISION] {description[:120]}")

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

        if SPEAK_VLM and self.speak_callback:
            self.speak_callback(description[:250])

        return f"[VISION] {description[:120]}..."


# ═══════════════════════════════════════════════════════════════════════════════
# §H  LOOPS
# ═══════════════════════════════════════════════════════════════════════════════

def brain_loop(brain: Brain):
    """
    Main event processing loop. Runs as a daemon thread.
    Pulls events from event_queue and dispatches them to Brain.process().
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
    print("Ask anything — Lumen grows smarter about Math, Science, and Art with every exchange.")
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
# ═══════════════════════════════════════════════════════════════════════════════

def start_voice_system(brain: Brain):
    """
    Load VoiceSystem, start listening, wire speak callback to brain.
    Returns VoiceSystem on success, None on failure (text input still works).
    """
    try:
        from voice_system import VoiceSystem

        def on_transcript(text: str):
            log(f"[VOICE] Heard: {text}")
            emit_event("user_input", {"text": text})

        voice = VoiceSystem(transcript_callback=on_transcript)
        voice.start()
        brain.speak_callback = lambda text: voice.speak(text, mind_id="system")
        log("[VOICE] 🎙 Always listening")
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
#   5. Tree of Life shape printed
#   6. Knowledge Tree shape printed
#   7. Status panel printed
#   8. Main thread idles (Ctrl-C triggers graceful shutdown)
#
# Shutdown order:
#   brain.running = False  → stops brain_loop
#   tree.snapshot()        → saves Tree of Life to disk
#   ktree.snapshot()       → saves Knowledge Tree to disk
#   sight.stop()           → closes IMX500 session cleanly
#   voice.stop()           → stops microphone + TTS queue
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log("Starting Luminova Core v4.0 — The Growing Mind")

    # ── 1. Create Brain ───────────────────────────────────────────────────────
    brain = Brain()

    # ── 2. Start event + input threads ───────────────────────────────────────
    threading.Thread(target=brain_loop,  args=(brain,), daemon=True).start()
    threading.Thread(target=input_loop,  args=(brain,), daemon=True).start()

    # ── 3. Start voice ────────────────────────────────────────────────────────
    voice = start_voice_system(brain)

    # ── 4. Start sight ────────────────────────────────────────────────────────
    sight = start_sight_system(brain)

    # ── 5. Print Tree of Life shape ───────────────────────────────────────────
    if tree:
        tree.print_shape()

    # ── 6. Print Knowledge Tree shape ─────────────────────────────────────────
    if ktree:
        ktree.print_shape()

    # ── 7. Status panel ───────────────────────────────────────────────────────
    sight_status = (sight.status()
                    if sight else {"mode": "offline", "entries": 0})
    voice_status = "active" if voice else "disabled — text input only"

    k_status = "disabled (knowledge_tree.py missing)"
    if ktree:
        ks = ktree.stats()
        k_status = (
            f"math {ks['domains']['math']['total_leaves']} · "
            f"science {ks['domains']['science']['total_leaves']} · "
            f"art {ks['domains']['art']['total_leaves']} leaves"
        )

    print("\n" + "=" * 70)
    print(" L U M E N   v 4 . 0 — T H E   G R O W I N G   M I N D")
    print("=" * 70)
    print(f" Memory    : Tree of Life (knowledge | core | reality)")
    print(f" Knowledge : {k_status}")
    print(f" Voice     : {voice_status}")
    print(f" Vision    : {sight_status['mode']} "
          f"({sight_status['entries']} visual memories)")
    print(f" NPU       : IMX500 (detect ↔ pose, memory-driven adaptive)")
    print(f" VLM       : {'hailo-ollama → ' if USE_HAILO else ''}ollama fallback")
    print()
    print(" ── HOW TO USE ─────────────────────────────────────────────────")
    print(" Type or speak to talk to Lumen.")
    print(" Ask about math, science, or art — her knowledge grows each time.")
    print(" Say 'look', 'what do you see', 'scan' to use the camera.")
    print(" Type 'quit' or 'exit' to shut down.")
    print()
    print(" ── SNAPSHOT HER MIND ──────────────────────────────────────────")
    print(" cp -r ~/luminova/tree/ ~/backup/        (Tree of Life)")
    print(" cp -r ~/luminova/knowledge/ ~/backup/   (Knowledge Tree)")
    print(" cp ~/luminova/sight/index.json ~/backup/ (Visual memory)")
    print("=" * 70 + "\n")

    # ── 8. Keep alive + graceful shutdown ─────────────────────────────────────
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("\nShutting down gracefully…")

        brain.running = False

        # Save Tree of Life snapshot
        if tree:
            tree.snapshot()
            log("🌳 Tree of Life snapshot saved.")

        # Save Knowledge Tree snapshot (new in v4.0)
        if ktree:
            ktree.snapshot()
            log("📚 Knowledge Tree snapshot saved.")

        # Stop vision
        if sight:
            sight.stop()
            log("👁 Sight stopped.")

        # Stop voice
        if voice:
            voice.stop()
            log("🎙 Voice stopped.")

        log("Luminova Core v4.0 — goodbye.")

# luminova_core_v2_2.py
#  -WxE-
# Voice-integrated, event-driven AI core

import queue
import threading
import time
import subprocess
import os
import datetime
import logging

# =========================
# LOGGING
# =========================
LOG_DIR = os.path.expanduser("~/luminova_logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "core.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# =========================
# MEMORY
# =========================
BASE_DIR = os.path.expanduser("~/luminova_memory")
WORKING_DIR = os.path.join(BASE_DIR, "working")
os.makedirs(WORKING_DIR, exist_ok=True)

def save_memory(user, ai):
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.txt")
    path = os.path.join(WORKING_DIR, filename)

    with open(path, "w") as f:
        f.write(f"USER:\n{user}\n\nAI:\n{ai}")

# =========================
# LLM
# =========================
def query_llm(prompt, timeout=369):
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:1.5b"],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        return result.stdout.decode().strip()
    except subprocess.TimeoutExpired:
        return "[Timeout]"
    except Exception as e:
        return f"[LLM ERROR] {e}"

# =========================
# EVENT SYSTEM
# =========================
MAX_EVENTS = 50
event_queue = queue.Queue(maxsize=MAX_EVENTS)

def emit_event(event_type, data):
    try:
        event_queue.put_nowait({
            "type": event_type,
            "data": data,
            "time": time.time()
        })
    except queue.Full:
        log("[WARN] Event queue full")

# =========================
# BRAIN
# =========================
class Brain:
    def __init__(self):
        self.running = True
        self.speak_callback = None  # optional

    def process(self, event):
        etype = event["type"]
        data = event["data"]

        log(f"[EVENT] {etype}")

        if etype == "user_input":
            return self.handle_user_input(data)

        return None

    def handle_user_input(self, text):
        log("[LLM] Thinking...")

        response = query_llm(text)

        save_memory(text, response)

        if self.speak_callback:
            self.speak_callback(response)

        return response

# =========================
# BRAIN LOOP
# =========================
def brain_loop(brain):
    while brain.running:
        try:
            event = event_queue.get(timeout=1)

            start = time.time()
            response = brain.process(event)
            duration = time.time() - start

            log(f"[TIME] {duration:.2f}s")

            if response:
                print(f"\nLuminova: {response}\n")

        except queue.Empty:
            continue
        except Exception as e:
            log(f"[ERROR] {e}")

# =========================
# TEXT INPUT (fallback)
# =========================
def input_loop():
    while True:
        text = input("You: ").strip()
        if text:
            emit_event("user_input", text)

# =========================
# VOICE HOOK
# =========================
def start_voice_system(brain):
    try:
        from voice_system import VoiceSystem

        def on_transcript(text):
            log(f"[VOICE] {text}")
            emit_event("user_input", text)

        voice = VoiceSystem(transcript_callback=on_transcript)
        voice.start()

        # connect TTS output
        brain.speak_callback = lambda text: voice.speak(text)

        log("[VOICE] System started")

        return voice

    except Exception as e:
        log(f"[VOICE ERROR] {e}")
        return None

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    log("Starting Luminova v2.2")

    brain = Brain()

    threading.Thread(target=brain_loop, args=(brain,), daemon=True).start()
    threading.Thread(target=input_loop, daemon=True).start()

    # Start voice safely
    voice = start_voice_system(brain)

    while True:
        time.sleep(1)

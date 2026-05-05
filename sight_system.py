#!/usr/bin/env python3
"""
LUMINOVA SIGHT SYSTEM v5  —  WxE
Raspberry Pi AI Camera  ·  Sony IMX500 NPU  ·  Adaptive Vision Memory

════════════════════════════════════════════════════════════════════════════════
BUILT FROM
  /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json   (user-provided)
  /usr/share/rpi-camera-assets/imx500_posenet.json          (user-provided)
  raspberrypi.com/documentation/accessories/ai-camera.html
  github.com/raspberrypi/picamera2/tree/main/examples/imx500

════════════════════════════════════════════════════════════════════════════════
PICAMERA2 IMX500 RULES  (from official documentation — never deviate)

  RULE 1  IMX500() MUST be constructed BEFORE Picamera2()
          It programs the sensor firmware over MIPI before the camera opens.

  RULE 2  Always use imx500.camera_num for Picamera2()
          Never hardcode a camera index.

  RULE 3  Always call imx500.show_network_fw_progress_bar() before picam2.start()
          Blocks until the RPK network is fully loaded onto the NPU.

  RULE 4  Always call intrinsics.update_with_defaults() after any NetworkIntrinsics setup.

  RULE 5  imx500.get_outputs(metadata, add_batch=True) returns None
          while the NPU firmware is initialising. Poll until not None.

  RULE 6  Switching models requires:
            picam2.stop() → picam2.close() → new IMX500() → new Picamera2()
          You cannot hot-swap the RPK.

════════════════════════════════════════════════════════════════════════════════
MODELS

  DETECT   /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
           From imx500_mobilenet_ssd.json → "network_file"
           _pp = fully post-processed on NPU.
           get_outputs(metadata, add_batch=True) → list of numpy arrays:
             [0][0]  boxes    shape [N, 4]  (y0, x0, y1, x1) normalised 0–1
             [1][0]  scores   shape [N]     float 0–1
             [2][0]  classes  shape [N]     float index into COCO_LABELS
           threshold    0.6   (JSON: "threshold")
           max_dets     5     (JSON: "max_detections")
           temporal     tol=0.1, factor=0.2, visible=4, hidden=2  (NPU-side)
           norm_val     [384, 384, 384, 0]   (NPU-side input pre-processing)

  POSE     /usr/share/imx500-models/imx500_network_posenet.rpk
           From imx500_posenet.json → "network_file"
           Raw tensor outputs — decoded in Python.
           get_outputs(metadata, add_batch=True) → list of numpy arrays:
             [0]  heatmaps      shape [1, H, W, 17]   logit scores per keypoint
             [1]  short_offsets shape [1, H, W, 34]   (dy, dx) per keypoint
           Decode pipeline:
             1. sigmoid(heatmaps) → probabilities [H, W, 17]
             2. argmax over (H, W) per keypoint k → peak (row_k, col_k)
             3. offset refinement for offset_refinement_steps=5 iterations:
                  r = clamp(round(y_k), 0, H-1)
                  c = clamp(round(x_k), 0, W-1)
                  y_k += short_offsets[r, c, k]      / H
                  x_k += short_offsets[r, c, k + 17] / W
             4. normalise final position to [0, 1]: y/H, x/W
             5. score = heatmap probability at final clamped position
           detect_threshold  0.4   (JSON: "threshold")    — accept a person
           plot_threshold    0.2   (JSON: plot_pose_cv.confidence_threshold)
           max_persons       5     (JSON: "max_detections")
           nms_radius        10.0  (JSON: "nms_radius")
           temporal          tol=0.3, factor=0.3, visible=8, hidden=2 (NPU-side)

  IDLE     Session fully closed. Near-zero power. Active during night quiet window.

════════════════════════════════════════════════════════════════════════════════
ADAPTIVE LEARNING

  Every significant SightEvent is stored in SightMemory (JSON on disk).
  AdaptiveMind reads tag-count history each dwell cycle and chooses mode:
    person-heavy last hour  → POSE, dwell scales up with confidence
    object-heavy last hour  → DETECT
    night + no person 2min  → IDLE
  Dwell time adapts between DWELL_BASE_S (30s) and DWELL_MAX_S (120s)
  based on how strongly the history supports the chosen mode.

  VLM descriptions confirm or contradict NPU detections →
  per-label reliability score (0.5–1.2) improves over Luminova's lifetime.
  Reliable labels get surfaced first in LLM context snippets.

════════════════════════════════════════════════════════════════════════════════
WIRE-UP IN luminova_core

  from sight_system import SightSystem, SightMode, start_sight
  sight = start_sight(brain, tree=tree)
  sight.query_scene("what do you see?")
  ctx   = sight.memory_context()

CLI
  python sight_system.py                 live, auto mode
  python sight_system.py --mode detect   force detection
  python sight_system.py --mode pose     force pose
  python sight_system.py --query "..."   one VLM query then exit
  python sight_system.py --stats         memory + learned reliability stats
  python sight_system.py --hailo         use hailo-ollama VLM backend
"""

# ─────────────────────────────────────────────────────────────────────────────
import io, sys, json, time, uuid, base64, math
import logging, datetime, threading, urllib.request
from enum        import Enum
from pathlib     import Path
from dataclasses import dataclass, field
from typing      import Optional, List, Dict, Callable, Tuple, Any

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ═════════════════════════════════════════════════════════════════════════════
# §1  CONSTANTS  —  everything sourced from the two JSON files
# ═════════════════════════════════════════════════════════════════════════════

# ── Detection  (imx500_mobilenet_ssd.json → imx500_object_detection) ─────────
MODEL_DETECT        = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
DET_THRESHOLD       = 0.6    # "threshold"
DET_MAX             = 5      # "max_detections"
DET_NORM_VAL        = (384, 384, 384, 0)   # "norm_val"  (NPU-side, informational)
DET_TEMP_TOL        = 0.1    # temporal_filter.tolerance
DET_TEMP_FACTOR     = 0.2    # temporal_filter.factor
DET_VISIBLE_FRAMES  = 4      # temporal_filter.visible_frames
DET_HIDDEN_FRAMES   = 2      # temporal_filter.hidden_frames

# ── Pose  (imx500_posenet.json → imx500_posenet) ──────────────────────────────
MODEL_POSE          = "/usr/share/imx500-models/imx500_network_posenet.rpk"
POSE_THRESHOLD      = 0.4    # "threshold"         — accept a detected person
POSE_PLOT_MIN       = 0.2    # plot_pose_cv.confidence_threshold — show keypoint
POSE_MAX            = 5      # "max_detections"
POSE_REFINE_STEPS   = 5      # "offset_refinement_steps"
POSE_NMS_RADIUS     = 10.0   # "nms_radius"
POSE_TEMP_TOL       = 0.3    # temporal_filter.tolerance
POSE_TEMP_FACTOR    = 0.3    # temporal_filter.factor
POSE_VISIBLE_FRAMES = 8      # temporal_filter.visible_frames
POSE_HIDDEN_FRAMES  = 2      # temporal_filter.hidden_frames

# ── COCO labels  (exact copy of imx500_mobilenet_ssd.json "classes" array) ────
# "-" entries are unlabelled COCO indices.  Detections with label "-" are dropped.
COCO_LABELS: List[str] = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","-","stop sign","parking meter",
    "bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
    "zebra","giraffe","-","backpack","umbrella","-","-","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","-","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","-","dining table",
    "-","-","toilet","-","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","-",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# ── PoseNet 17-keypoint skeleton names (COCO order) ───────────────────────────
KEYPOINT_NAMES: List[str] = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

# ── Camera ────────────────────────────────────────────────────────────────────
# Do NOT hardcode device index — always use imx500.camera_num  (RULE 2)
CAM_BUFFER_COUNT  = 12      # recommended by official picamera2 IMX500 examples
CAM_FPS_FALLBACK  = 10      # used only if NetworkIntrinsics.inference_rate is None
VLM_CAM_INDEX     = 1       # Arducam for VLM frame grabs; NEVER use IMX500 device here
VLM_IMG_SIZE      = (1024, 768)
VLM_JPEG_QUALITY  = 85


# ═════════════════════════════════════════════════════════════════════════════
# §2  ADAPTIVE CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

DWELL_BASE_S      = 30.0    # min seconds before a mode switch is considered
DWELL_MAX_S       = 120.0   # max dwell even with very strong memory signal
EMIT_THROTTLE_S   = 4.0     # min seconds between same-kind events on queue
NIGHT_START       = 0       # hour: begin quiet window
NIGHT_END         = 6       # hour: end quiet window

VLM_COOLDOWN_S    = 60.0
VLM_PERSON_CD_S   = 45.0
VLM_MAX_TOKENS    = 350
VLM_FAST          = "moondream"
VLM_DEEP          = "fredrezones55/Qwen3.5-Uncensored-HauhauCS-Aggressive:4b"
VLM_HAILO         = "hailo-vlm"
HAILO_URL         = "http://localhost:8000/api/generate"
HAILO_TIMEOUT     = 15

MEM_DIR           = Path.home() / "luminova" / "sight"
MEM_IDX           = MEM_DIR / "index.json"
MEM_IMG_DIR       = MEM_DIR / "images"
MEM_MAX           = 3000    # oldest / lowest-importance trimmed above this
MEM_MIN_IMP       = 0.3     # don't persist below this importance
MEM_RECENT_N      = 8
LOG_DIR           = Path.home() / "luminova_logs"


# ═════════════════════════════════════════════════════════════════════════════
# §3  LOGGING
# ═════════════════════════════════════════════════════════════════════════════

for _d in (LOG_DIR, MEM_IMG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_log = logging.getLogger("Sight")
if not _log.handlers:
    _log.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] Sight › %(message)s")
    for _h in (
        logging.FileHandler(LOG_DIR / f"sight_{datetime.date.today()}.log"),
        logging.StreamHandler(sys.stdout),
    ):
        _h.setFormatter(_fmt)
        _log.addHandler(_h)


# ═════════════════════════════════════════════════════════════════════════════
# §4  DATA TYPES
# ═════════════════════════════════════════════════════════════════════════════

class SightMode(Enum):
    DETECT = "detect"
    POSE   = "pose"
    IDLE   = "idle"


class VLMDepth(Enum):
    FAST = "fast"
    DEEP = "deep"


@dataclass
class Detection:
    """One object from the SSD post-processed model."""
    label:     str
    score:     float
    cls_index: int
    box:       Tuple[float, float, float, float]  # (y0, x0, y1, x1) normalised

    @property
    def is_person(self) -> bool:
        return self.label == "person"


@dataclass
class Keypoint:
    """One of 17 body joints from PoseNet."""
    name:  str
    y:     float    # normalised 0–1
    x:     float    # normalised 0–1
    score: float

    @property
    def visible(self) -> bool:
        return self.score >= POSE_PLOT_MIN


@dataclass
class Pose:
    """One person's decoded PoseNet result."""
    keypoints: List[Keypoint]  # always 17, one per KEYPOINT_NAMES entry
    peak:      float           # max keypoint score — person confidence

    @property
    def visible_joints(self) -> List[str]:
        return [kp.name for kp in self.keypoints if kp.visible]

    @property
    def readable(self) -> str:
        vj = self.visible_joints
        if not vj:
            return "faint signal"
        body = ", ".join(vj[:5])
        return body + (f" +{len(vj) - 5} more" if len(vj) > 5 else "")


@dataclass
class SightEvent:
    kind:       str           # detect | pose | vlm | mode | error
    mode:       str
    summary:    str
    detections: List[Detection] = field(default_factory=list)
    poses:      List[Pose]      = field(default_factory=list)
    vlm_text:   str   = ""
    score:      float = 0.0
    image_b64:  str   = ""
    ts:         str   = field(
        default_factory=lambda: datetime.datetime.now().isoformat())


@dataclass
class VLMResult:
    text:  str
    model: str
    ok:    bool
    ms:    int = 0
    err:   str = ""


# ═════════════════════════════════════════════════════════════════════════════
# §5  POSENET DECODER
#
# Implements the decode pipeline described in imx500_posenet.json exactly:
#   offset_refinement_steps : 5
#   nms_radius              : 10.0   (used in person-score ranking)
#   threshold               : 0.4   (accept person)
#   plot_threshold          : 0.2   (show keypoint)
#
# Raw tensor layout from get_outputs(metadata, add_batch=True):
#   outputs[0]  heatmaps      [1, H, W, 17]  logit scores
#   outputs[1]  short_offsets [1, H, W, 34]  (dy, dx) for each of 17 keypoints
# ═════════════════════════════════════════════════════════════════════════════

class PoseNetDecoder:
    """
    Pure-Python + optional numpy PoseNet tensor decoder.
    Matches imx500_posenet.json parameters exactly.
    """

    @staticmethod
    def decode(outputs) -> List[Pose]:
        if outputs is None or len(outputs) < 2:
            return []
        try:
            import numpy as np
            return PoseNetDecoder._decode_np(outputs, np)
        except ImportError:
            return PoseNetDecoder._decode_pure(outputs)
        except Exception as e:
            _log.debug(f"[POSE] decode: {e}")
            return []

    # ── numpy path ────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_np(outputs, np) -> List[Pose]:
        # Unwrap batch dimension
        hm_raw  = np.asarray(outputs[0], dtype=np.float32)  # [1,H,W,17] or [H,W,17]
        off_raw = np.asarray(outputs[1], dtype=np.float32)  # [1,H,W,34] or [H,W,34]

        hm  = hm_raw[0]  if hm_raw.ndim  == 4 else hm_raw   # [H, W, 17]
        off = off_raw[0] if off_raw.ndim == 4 else off_raw   # [H, W, 34]

        if hm.ndim != 3 or hm.shape[2] != 17:
            _log.debug(f"[POSE] unexpected heatmap shape {hm.shape}")
            return []

        H, W, N = hm.shape  # grid height, grid width, 17 keypoints

        # Step 1 — sigmoid activation
        hm_prob = 1.0 / (1.0 + np.exp(-np.clip(hm, -88.0, 88.0)))  # [H,W,17]

        # Step 2 — argmax per keypoint channel
        flat      = hm_prob.reshape(-1, N)               # [H*W, 17]
        kp_flat   = flat.argmax(axis=0)                  # [17]
        kp_scores = flat[kp_flat, np.arange(N)]          # [17]
        kp_y      = (kp_flat // W).astype(np.float32)    # grid row  [17]
        kp_x      = (kp_flat %  W).astype(np.float32)    # grid col  [17]

        # Step 3 — offset refinement (offset_refinement_steps = 5)
        for _ in range(POSE_REFINE_STEPS):
            r = np.clip(np.round(kp_y).astype(int), 0, H - 1)
            c = np.clip(np.round(kp_x).astype(int), 0, W - 1)
            # offsets are in units of (input_pixels / grid_cells)
            # dividing by H/W converts to grid-cell units
            for k in range(N):
                kp_y[k] += off[r[k], c[k], k]       / H
                kp_x[k] += off[r[k], c[k], k + N]   / W

        # Step 4 — accept person if peak score >= POSE_THRESHOLD (0.4)
        peak = float(kp_scores.max())
        if peak < POSE_THRESHOLD:
            return []

        # Step 5 — build Pose object with 17 Keypoints
        keypoints = [
            Keypoint(
                name  = KEYPOINT_NAMES[k],
                y     = float(np.clip(kp_y[k] / H, 0.0, 1.0)),
                x     = float(np.clip(kp_x[k] / W, 0.0, 1.0)),
                score = float(kp_scores[k]),
            )
            for k in range(N)
        ]
        return [Pose(keypoints=keypoints, peak=peak)]

    # ── pure-python fallback (no numpy) ───────────────────────────────────────

    @staticmethod
    def _decode_pure(outputs) -> List[Pose]:
        """
        Scans all heatmap values for the global peak.
        Position is approximate (no offset refinement without numpy).
        Used only when numpy is unavailable.
        """
        try:
            hm_raw = outputs[0]
            hm     = hm_raw[0] if (isinstance(hm_raw, (list, tuple))
                                    and len(hm_raw)
                                    and isinstance(hm_raw[0], (list, tuple))) \
                     else hm_raw

            H     = len(hm)
            W     = len(hm[0]) if H else 1
            N     = len(hm[0][0]) if H and W else 17

            best_v = [float("-inf")] * N
            best_r = [0] * N
            best_c = [0] * N

            for r, row in enumerate(hm):
                for c, col in enumerate(row):
                    for k, v in enumerate(col):
                        fv = float(v)
                        if fv > best_v[k]:
                            best_v[k] = fv
                            best_r[k] = r
                            best_c[k] = c

            # Sigmoid
            kp_scores = [1.0 / (1.0 + math.exp(-max(-88.0, min(88.0, v))))
                         for v in best_v]
            peak = max(kp_scores)
            if peak < POSE_THRESHOLD:
                return []

            keypoints = [
                Keypoint(
                    name  = KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else f"kp{k}",
                    y     = best_r[k] / max(H - 1, 1),
                    x     = best_c[k] / max(W - 1, 1),
                    score = kp_scores[k],
                )
                for k in range(N)
            ]
            return [Pose(keypoints=keypoints, peak=peak)]

        except Exception as e:
            _log.debug(f"[POSE] pure decode: {e}")
            return []


# ═════════════════════════════════════════════════════════════════════════════
# §6  SIGHT MEMORY  —  adaptive learning store
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MemEntry:
    eid:        str
    mode:       str
    summary:    str
    tags:       List[str]
    importance: float
    ts:         str
    vlm_text:   str = ""
    img_path:   str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Optional[MemEntry]":
        try:
            filtered = {k: v for k, v in d.items()
                        if k in cls.__dataclass_fields__}
            # Ensure all required (non-default) fields are present
            required = {"eid", "mode", "summary", "tags", "importance", "ts"}
            if not required.issubset(filtered):
                return None
            return cls(**filtered)
        except Exception:
            return None


class SightMemory:
    """
    JSON-backed persistent store for all sight events.

    Drives adaptation:
      tag_counts(hours)      → frequency of what Luminova has been seeing
                               AdaptiveMind uses this to choose DETECT/POSE/IDLE
      confidence_for(label)  → learned reliability 0.5–1.2 per COCO class
                               rises with VLM confirmations, falls with contradictions
      context_snippet()      → short recent-history text for LLM prompt injection
    """

    def __init__(self):
        self._lock  = threading.RLock()
        self._idx:  Dict[str, MemEntry] = {}
        # {label: [confirmed_count, contradicted_count]}
        self._conf: Dict[str, List[int]] = {}
        self._load()

    # ── write ─────────────────────────────────────────────────────────────────

    def store(self, evt: SightEvent, importance: float) -> Optional[MemEntry]:
        if importance < MEM_MIN_IMP:
            return None
        with self._lock:
            entry = MemEntry(
                eid        = str(uuid.uuid4())[:10],
                mode       = evt.mode,
                summary    = evt.summary[:300],
                tags       = self._auto_tags(evt),
                importance = importance,
                ts         = evt.ts,
                vlm_text   = evt.vlm_text[:400],
            )
            if evt.image_b64 and importance >= 0.7:
                try:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    p  = MEM_IMG_DIR / f"{ts}_{entry.eid}.jpg"
                    p.write_bytes(base64.b64decode(evt.image_b64))
                    entry.img_path = str(p)
                except Exception:
                    pass
            self._idx[entry.eid] = entry
            self._trim()
            self._save()
            return entry

    def learn_from_vlm(self, vlm_text: str, detections: List[Detection]):
        """
        VLM mentions a detected label → confirmed (+1).
        VLM is silent about a detected label → soft contradiction (+1 disputed).
        Adjusts confidence_for() score over Luminova's lifetime.
        """
        if not vlm_text or not detections:
            return
        low = vlm_text.lower()
        with self._lock:
            for d in detections:
                if d.label in ("-", ""):
                    continue
                if d.label not in self._conf:
                    self._conf[d.label] = [0, 0]
                if d.label in low or any(w in low for w in d.label.split()):
                    self._conf[d.label][0] += 1   # confirmed
                else:
                    self._conf[d.label][1] += 1   # not mentioned
            self._save()

    # ── read ──────────────────────────────────────────────────────────────────

    def confidence_for(self, label: str) -> float:
        """Learned multiplier 0.5–1.2. Neutral = 1.0 before any data."""
        with self._lock:
            pair = self._conf.get(label)
        if not pair or sum(pair) == 0:
            return 1.0
        ratio = pair[0] / sum(pair)
        return round(max(0.5, min(1.2, 0.5 + ratio * 0.7)), 3)

    def recent(self, n: int = MEM_RECENT_N) -> List[MemEntry]:
        with self._lock:
            return sorted(self._idx.values(),
                          key=lambda e: e.ts, reverse=True)[:n]

    def tag_counts(self, hours: int = 1) -> Dict[str, int]:
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours)
        counts: Dict[str, int] = {}
        with self._lock:
            for e in self._idx.values():
                try:
                    if datetime.datetime.fromisoformat(e.ts) < cutoff:
                        continue
                except Exception:
                    continue
                for t in e.tags:
                    counts[t] = counts.get(t, 0) + 1
        return counts

    def context_snippet(self, n: int = 5) -> str:
        entries = self.recent(n)
        if not entries:
            return ""
        lines = [f"- [{e.mode}] {e.summary}" for e in entries]
        return "Recent visual memory:\n" + "\n".join(lines)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_entries": len(self._idx),
                "learned_labels": {
                    k: {
                        "confirmed":   v[0],
                        "disputed":    v[1],
                        "reliability": self.confidence_for(k),
                    }
                    for k, v in sorted(self._conf.items(),
                                       key=lambda x: -sum(x[1]))
                },
            }

    # ── internal ──────────────────────────────────────────────────────────────

    _META = {"detect","pose","idle","vlm","mode","error",
             "person","person_detected","pose_update",
             "objects_seen","vlm_result","mode_changed"}

    @staticmethod
    def _auto_tags(evt: SightEvent) -> List[str]:
        tags = [evt.mode, evt.kind]
        for d in evt.detections:
            if d.label not in ("-", ""):
                tags.append(d.label.replace(" ", "_"))
        for _ in evt.poses:
            tags.append("person")
        return list(dict.fromkeys(tags))

    def _trim(self):
        if len(self._idx) <= MEM_MAX:
            return
        # Drop lowest importance first, then oldest
        ordered = sorted(self._idx.items(),
                         key=lambda kv: (kv[1].importance, kv[1].ts))
        for eid, _ in ordered[:len(self._idx) - MEM_MAX]:
            del self._idx[eid]

    def _load(self):
        if not MEM_IDX.exists():
            return
        try:
            raw = json.loads(MEM_IDX.read_text())
            loaded = {}
            for k, v in raw.get("entries", {}).items():
                entry = MemEntry.from_dict(v)
                if entry is not None:
                    loaded[k] = entry
            self._idx  = loaded
            self._conf = raw.get("confidence", {})
            _log.info(f"Memory loaded: {len(self._idx)} entries, "
                      f"{len(self._conf)} learned labels")
        except Exception as e:
            _log.error(f"Memory load failed: {e}")

    def _save(self):
        try:
            MEM_IDX.write_text(json.dumps({
                "entries":    {k: v.to_dict() for k, v in self._idx.items()},
                "confidence": self._conf,
                "saved_at":   datetime.datetime.now().isoformat(),
            }, indent=2))
        except Exception as e:
            _log.error(f"Memory save failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# §7  ADAPTIVE MIND  —  reads memory, chooses SightMode + dwell duration
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveMind:
    """
    Memory-driven mode selector with adaptive dwell.

    Each evaluation (after dwell expires or on force):
      IDLE   : night window AND no person seen in >2 minutes
      POSE   : person seen in last 90s  OR  person-dominant tag history (last hour)
      DETECT : everything else

    Dwell time adapts:
      history strongly supports the mode  → dwell up to DWELL_MAX_S (120s)
      weak or neutral signal              → dwell at DWELL_BASE_S (30s)
    """

    _META = SightMemory._META

    def __init__(self, memory: SightMemory):
        self._mem             = memory
        self._mode            = SightMode.DETECT
        self._last_switch     = time.monotonic() - DWELL_BASE_S
        self._dwell           = DWELL_BASE_S
        self._forced:          Optional[SightMode] = None
        self._person_seen_at:  float = float("-inf")

    @property
    def mode(self) -> SightMode:
        return self._mode

    def notify_person(self):
        self._person_seen_at = time.monotonic()

    def force(self, m: SightMode):
        self._forced = m

    def update(self) -> Tuple[SightMode, bool]:
        """Returns (current_mode, did_switch)."""
        if self._forced is not None:
            new, self._forced = self._forced, None
            return self._apply(new, DWELL_BASE_S)
        if time.monotonic() - self._last_switch < self._dwell:
            return self._mode, False
        new, dwell = self._pick()
        return self._apply(new, dwell)

    def _pick(self) -> Tuple[SightMode, float]:
        hour       = datetime.datetime.now().hour
        person_age = time.monotonic() - self._person_seen_at

        # Night + no person → park camera
        if NIGHT_START <= hour < NIGHT_END and person_age > 120:
            return SightMode.IDLE, DWELL_BASE_S

        # Recent person → pose
        if person_age < 90:
            return SightMode.POSE, DWELL_BASE_S

        # Read tag history from last hour
        counts      = self._mem.tag_counts(hours=1)
        person_hits = (counts.get("person", 0)
                       + counts.get("pose_update", 0)
                       + counts.get("person_detected", 0))
        object_hits = sum(v for k, v in counts.items()
                          if k not in self._META)
        total = person_hits + object_hits

        if total == 0:
            return SightMode.DETECT, DWELL_BASE_S

        p_ratio = person_hits / total
        if p_ratio >= 0.5:
            # Person-dominant → POSE, longer dwell scales with confidence
            dwell = DWELL_BASE_S + (DWELL_MAX_S - DWELL_BASE_S) * p_ratio
            return SightMode.POSE, round(dwell, 1)

        o_ratio = object_hits / total
        dwell = DWELL_BASE_S + (DWELL_MAX_S - DWELL_BASE_S) * o_ratio * 0.5
        return SightMode.DETECT, round(dwell, 1)

    def _apply(self, new: SightMode, dwell: float) -> Tuple[SightMode, bool]:
        self._dwell = dwell
        if new == self._mode:
            return self._mode, False
        _log.info(f"[MIND] {self._mode.value} → {new.value}  "
                  f"dwell={dwell:.0f}s")
        self._mode        = new
        self._last_switch = time.monotonic()
        return new, True


# ═════════════════════════════════════════════════════════════════════════════
# §8  IMX500 SESSION  —  the correct picamera2 hardware driver
#
# Implements all 6 official rules (see module docstring).
# open(mode) performs a full model swap via stop/close/re-init when needed.
# poll() returns (np_outputs, metadata) or None.
# ═════════════════════════════════════════════════════════════════════════════

class IMX500Session:
    """
    One active IMX500 + Picamera2 session.
    Owns the camera device for the lifetime of one model load.
    """

    def __init__(self):
        self._imx500     = None
        self._picam2     = None
        self._intrinsics = None
        self._mode: Optional[SightMode] = None

    @property
    def mode(self) -> Optional[SightMode]:
        return self._mode

    # ── open ─────────────────────────────────────────────────────────────────

    def open(self, mode: SightMode) -> bool:
        # Rule 6: full close before any re-open
        self._close_hw()

        if mode == SightMode.IDLE:
            _log.info("[CAM] IDLE — camera parked")
            self._mode = mode
            return True

        model_path = MODEL_DETECT if mode == SightMode.DETECT else MODEL_POSE
        task       = ("object detection" if mode == SightMode.DETECT
                      else "pose estimation")
        try:
            from picamera2 import Picamera2
            from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

            # Rule 1: IMX500 before Picamera2
            imx500 = IMX500(model_path)

            # Rule 4: intrinsics.update_with_defaults()
            intrinsics = imx500.network_intrinsics
            if not intrinsics:
                intrinsics = NetworkIntrinsics()
                intrinsics.task = task
            intrinsics.update_with_defaults()

            fps = intrinsics.inference_rate or CAM_FPS_FALLBACK

            # Rule 2: imx500.camera_num
            picam2 = Picamera2(imx500.camera_num)
            config = picam2.create_preview_configuration(
                controls     = {"FrameRate": fps},
                buffer_count = CAM_BUFFER_COUNT,
            )

            # Rule 3: progress bar before start
            imx500.show_network_fw_progress_bar()
            picam2.start(config, show_preview=False)

            self._imx500     = imx500
            self._picam2     = picam2
            self._intrinsics = intrinsics
            self._mode       = mode
            _log.info(f"[CAM] {mode.value} ready — {Path(model_path).name} "
                      f"@ {fps:.0f}fps  cam={imx500.camera_num}")
            return True

        except Exception as e:
            _log.error(f"[CAM] open({mode.value}): {e}")
            self._imx500 = self._picam2 = self._intrinsics = None
            return False

    # ── poll ─────────────────────────────────────────────────────────────────

    def poll(self) -> Optional[Tuple]:
        """
        Returns (np_outputs, metadata) or None.
        np_outputs may itself be None while the NPU firmware loads (Rule 5).
        """
        if not self._picam2 or not self._imx500:
            return None
        try:
            metadata   = self._picam2.capture_metadata()
            np_outputs = self._imx500.get_outputs(metadata, add_batch=True)
            return np_outputs, metadata
        except Exception as e:
            _log.debug(f"[CAM] poll: {e}")
            return None

    # ── parse detection ───────────────────────────────────────────────────────

    def parse_detections(self, np_outputs, metadata) -> List[Detection]:
        """
        SSD _pp post-processed layout (add_batch=True):
          np_outputs[0][0]  boxes    [N, 4]  (y0, x0, y1, x1) normalised 0-1
          np_outputs[1][0]  scores   [N]
          np_outputs[2][0]  classes  [N]     float class index

        Applies DET_THRESHOLD=0.6, caps at DET_MAX=5.
        Converts box to output-image pixel coords via convert_inference_coords.
        Skips "-" COCO labels.
        """
        results: List[Detection] = []
        if np_outputs is None:
            return results
        try:
            import numpy as np
            boxes_r   = np.asarray(np_outputs[0][0], dtype=np.float32)
            scores_r  = np.asarray(np_outputs[1][0], dtype=np.float32)
            classes_r = np.asarray(np_outputs[2][0], dtype=np.float32)

            for i in range(min(len(scores_r), DET_MAX)):
                score = float(scores_r[i])
                if score < DET_THRESHOLD:
                    continue
                cls   = int(classes_r[i])
                label = (COCO_LABELS[cls]
                         if 0 <= cls < len(COCO_LABELS) else f"obj{cls}")
                if label == "-":
                    continue

                raw_box = tuple(float(v) for v in boxes_r[i])
                try:
                    # Convert from inference-input coordinates to
                    # full-resolution output-image coordinates
                    box = self._imx500.convert_inference_coords(
                        raw_box, metadata, self._picam2)
                except Exception:
                    box = raw_box

                results.append(Detection(
                    label     = label,
                    score     = round(score, 3),
                    cls_index = cls,
                    box       = box,
                ))
        except Exception as e:
            _log.debug(f"[DET] parse: {e}")
        return results

    # ── parse pose ────────────────────────────────────────────────────────────

    def parse_poses(self, np_outputs) -> List[Pose]:
        """
        PoseNet raw tensor decode via PoseNetDecoder.
        Implements imx500_posenet.json params:
          offset_refinement_steps=5, threshold=0.4, plot_min=0.2
        """
        return PoseNetDecoder.decode(np_outputs)

    # ── close ─────────────────────────────────────────────────────────────────

    def close(self):
        self._close_hw()
        _log.info("[CAM] session closed")

    def _close_hw(self):
        try:
            if self._picam2:
                self._picam2.stop()
                self._picam2.close()
        except Exception:
            pass
        self._picam2 = self._imx500 = self._intrinsics = None


# ═════════════════════════════════════════════════════════════════════════════
# §9  VLM ENGINE  —  hailo-ollama → ollama CPU  (async, non-blocking)
# ═════════════════════════════════════════════════════════════════════════════

class VLMEngine:
    def __init__(self, use_hailo: bool = False):
        self._lock      = threading.Lock()
        self._busy      = False
        self._last_run  = 0.0
        self._use_hailo = use_hailo

    def fire(
        self,
        image_b64: str,
        prompt:    str,
        depth:     VLMDepth = VLMDepth.FAST,
        callback:  Callable[[VLMResult], None] = None,
        cooldown:  float = VLM_COOLDOWN_S,
    ) -> bool:
        """Non-blocking. Returns False if busy or on cooldown."""
        with self._lock:
            if self._busy:
                return False
            if time.monotonic() - self._last_run < cooldown:
                return False
            self._busy = True

        def _run():
            t0 = time.monotonic()
            try:
                r = self._call(image_b64, prompt, depth)
                r.ms = int((time.monotonic() - t0) * 1000)
            except Exception as e:
                r = VLMResult("", "error", False, 0, str(e))
            finally:
                with self._lock:
                    self._busy     = False
                    self._last_run = time.monotonic()
            if callback:
                try:
                    callback(r)
                except Exception as e:
                    _log.debug(f"[VLM] callback: {e}")

        threading.Thread(target=_run, daemon=True, name="VLM").start()
        return True

    def _call(self, img: str, prompt: str, depth: VLMDepth) -> VLMResult:
        if self._use_hailo:
            r = self._hailo(img, prompt)
            if r.ok:
                return r
        return self._ollama(img, prompt, depth)

    def _hailo(self, img: str, prompt: str) -> VLMResult:
        try:
            data = json.dumps({"model": VLM_HAILO, "prompt": prompt,
                               "images": [img], "stream": False}).encode()
            req  = urllib.request.Request(
                HAILO_URL, data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=HAILO_TIMEOUT) as r:
                text = json.loads(r.read()).get("response", "").strip()
            if text:
                return VLMResult(text, "hailo-vlm", True)
        except Exception as e:
            _log.debug(f"[VLM] hailo: {e}")
        return VLMResult("", "hailo", False)

    def _ollama(self, img: str, prompt: str, depth: VLMDepth) -> VLMResult:
        model = VLM_DEEP if depth == VLMDepth.DEEP else VLM_FAST
        try:
            import ollama as _ol
            resp = _ol.Client().chat(
                model    = model,
                messages = [{"role": "user", "content": prompt,
                             "images": [img]}],
                options  = {"num_predict": VLM_MAX_TOKENS, "temperature": 0.2},
                keep_alive = "10s",
            )
            text = (getattr(getattr(resp, "message", None), "content", None)
                    or "").strip()
            return VLMResult(text, model, bool(text))
        except Exception as e:
            return VLMResult("", model, False, 0, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# §10  FRAME CAPTURE  —  Arducam (VLM_CAM_INDEX=1) only
#      The IMX500 device (imx500.camera_num) is owned by IMX500Session.
#      These are two separate physical cameras. Never open both at once.
# ═════════════════════════════════════════════════════════════════════════════

def capture_frame_b64() -> Optional[str]:
    if not _HAS_PIL:
        _log.debug("[CAP] PIL unavailable")
        return None
    try:
        from picamera2 import Picamera2
        W, H = VLM_IMG_SIZE
        cam  = Picamera2(VLM_CAM_INDEX)
        cam.configure(cam.create_still_configuration(main={"size": (W, H)}))
        cam.start()
        time.sleep(0.3)          # let AGC stabilise
        arr = cam.capture_array()
        cam.stop()
        cam.close()

        img = Image.fromarray(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((W, H), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=VLM_JPEG_QUALITY, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        _log.debug(f"[CAP] {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# §11  SIGHT SYSTEM  —  orchestrates everything
# ═════════════════════════════════════════════════════════════════════════════

class SightSystem:
    """
    Luminova's complete vision system.

    One daemon thread runs the perception loop:
      1. AdaptiveMind.update()   — read memory, pick mode, respect dwell
      2. IMX500Session.open()    — swap model if mode changed (full re-init)
      3. IMX500Session.poll()    — capture_metadata → get_outputs
      4. parse_detections()      — SSD _pp boxes/scores/classes
         OR parse_poses()        — PoseNet heatmap decode
      5. Build SightEvent, emit to queue + callbacks
      6. On person: fire VLM async (Arducam), learn from result

    Public API (for luminova_core):
      sight.start()
      sight.stop()
      sight.force_mode(SightMode.POSE)
      sight.on_event("detect", callback)
      sight.query_scene("describe what you see")
      sight.memory_context()
      sight.status()
    """

    def __init__(
        self,
        event_callback: Callable[[str, dict], None] = None,
        use_hailo: bool = False,
        tree = None,
    ):
        self._event_cb  = event_callback
        self._tree      = tree
        self._memory    = SightMemory()
        self._mind      = AdaptiveMind(self._memory)
        self._session   = IMX500Session()
        self._vlm       = VLMEngine(use_hailo)
        self._stop_ev   = threading.Event()
        self._thread:    Optional[threading.Thread] = None
        self._running   = False
        self._last_emit: Dict[str, float] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._stop_ev.clear()
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="SightSystem")
        self._thread.start()
        _log.info("SightSystem started")

    def stop(self):
        self._stop_ev.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=12)
        self._session.close()
        _log.info("SightSystem stopped")

    def force_mode(self, mode: SightMode):
        _log.info(f"[SIGHT] force → {mode.value}")
        self._mind.force(mode)

    def on_event(self, kind: str, cb: Callable[[SightEvent], None]):
        self._callbacks.setdefault(kind, []).append(cb)

    def query_scene(
        self,
        prompt: str = "Briefly describe what you see. Two sentences.",
        depth:  VLMDepth = VLMDepth.FAST,
    ):
        """On-demand VLM — non-blocking. Result arrives via 'vlm' callback."""
        _log.info("[SIGHT] query_scene")
        b64 = capture_frame_b64()
        if not b64:
            _log.warning("[SIGHT] query_scene: frame capture failed")
            return

        def _cb(r: VLMResult):
            self._emit(SightEvent(
                kind      = "vlm",
                mode      = "vlm",
                summary   = r.text[:120] if r.text else "[VLM failed]",
                vlm_text  = r.text,
                image_b64 = b64,
                score     = 1.0,
            ), importance=0.75, force=True)

        self._vlm.fire(b64, prompt, depth=depth, callback=_cb, cooldown=0.0)

    def memory_context(self, n: int = 5) -> str:
        return self._memory.context_snippet(n)

    def status(self) -> dict:
        return {
            "running": self._running,
            "mode":    self._mind.mode.value,
            "entries": len(self._memory._idx),
        }

    # ── perception loop ───────────────────────────────────────────────────────

    def _loop(self):
        current_mode: Optional[SightMode] = None
        errors = 0

        while not self._stop_ev.is_set():
            try:
                mode, switched = self._mind.update()

                if switched or current_mode is None:
                    self._session.open(mode)
                    current_mode = mode
                    if switched:
                        self._emit(SightEvent(
                            kind    = "mode",
                            mode    = mode.value,
                            summary = f"Vision mode → {mode.value}",
                        ), importance=0.3, force=True)

                if current_mode == SightMode.IDLE:
                    self._stop_ev.wait(5.0)
                    errors = 0
                    continue

                result = self._session.poll()
                if result is None:
                    self._stop_ev.wait(0.1)
                    continue

                np_outputs, metadata = result

                # Rule 5: np_outputs is None while NPU firmware loads — keep polling
                if np_outputs is None:
                    self._stop_ev.wait(0.05)
                    continue

                if current_mode == SightMode.DETECT:
                    self._handle_detect(np_outputs, metadata)
                elif current_mode == SightMode.POSE:
                    self._handle_pose(np_outputs)

                errors = 0
                self._stop_ev.wait(0.05)

            except Exception as e:
                errors += 1
                _log.error(f"[SIGHT] loop #{errors}: {e}")
                if errors >= 5:
                    self._emit(SightEvent(
                        kind    = "error",
                        mode    = current_mode.value if current_mode else "?",
                        summary = str(e),
                    ), importance=0.5, force=True)
                    self._stop_ev.wait(30.0)
                    errors = 0
                else:
                    self._stop_ev.wait(2.0)

        self._session.close()

    # ── detection handler ─────────────────────────────────────────────────────

    def _handle_detect(self, np_outputs, metadata):
        dets = self._session.parse_detections(np_outputs, metadata)
        if not dets:
            return

        has_person = any(d.is_person for d in dets)
        if has_person:
            self._mind.notify_person()

        # Build human-readable summary with learned reliability hints
        counts: Dict[str, int] = {}
        for d in dets:
            counts[d.label] = counts.get(d.label, 0) + 1
        parts   = [f"{n} {l}" if n > 1 else l for l, n in counts.items()]
        summary = "I can see: " + ", ".join(parts)

        evt = SightEvent(
            kind       = "detect",
            mode       = SightMode.DETECT.value,
            summary    = summary,
            detections = dets,
            score      = max(d.score for d in dets),
        )
        self._emit(evt, importance=0.75 if has_person else 0.45)

        # Trigger VLM on person — result feeds learn_from_vlm()
        if has_person:
            b64 = capture_frame_b64()
            if b64:
                def _on_vlm(r: VLMResult, _e=evt, _b=b64):
                    if not r.ok or not r.text:
                        return
                    _log.info(f"[VLM] {r.model} {r.ms}ms")
                    self._memory.learn_from_vlm(r.text, _e.detections)
                    self._emit(SightEvent(
                        kind       = "vlm",
                        mode       = "vlm",
                        summary    = r.text[:120],
                        vlm_text   = r.text,
                        detections = _e.detections,
                        image_b64  = _b,
                        score      = 1.0,
                    ), importance=0.75, force=True)

                self._vlm.fire(
                    b64,
                    "Describe the person and scene in two sentences.",
                    callback = _on_vlm,
                    cooldown = VLM_PERSON_CD_S,
                )

    # ── pose handler ──────────────────────────────────────────────────────────

    def _handle_pose(self, np_outputs):
        poses = self._session.parse_poses(np_outputs)
        if not poses:
            return

        self._mind.notify_person()

        n       = len(poses)
        detail  = poses[0].readable
        summary = (f"{n} person — {detail}"
                   if n == 1 else f"{n} people — {detail}")

        self._emit(SightEvent(
            kind    = "pose",
            mode    = SightMode.POSE.value,
            summary = summary,
            poses   = poses,
            score   = max(p.peak for p in poses),
        ), importance=0.55)

    # ── emit ──────────────────────────────────────────────────────────────────

    def _emit(self, evt: SightEvent, importance: float = 0.5,
              force: bool = False):
        now = time.monotonic()
        if not force:
            if now - self._last_emit.get(evt.kind, 0.0) < EMIT_THROTTLE_S:
                return
        self._last_emit[evt.kind] = now

        _log.info(f"[{evt.kind.upper():6}] {evt.summary[:80]}")
        self._memory.store(evt, importance)

        # Plant VLM text into Tree of Life (reality/observations)
        if evt.kind == "vlm" and evt.vlm_text and self._tree:
            try:
                self._tree.remember(
                    content = evt.vlm_text,
                    limb    = "reality",
                    branch  = "observations",
                    tags    = ["vision", "sight"],
                    weight  = 0.8,
                    source  = "observation",
                )
            except Exception:
                pass

        # Push to luminova_core event queue
        if self._event_cb:
            try:
                self._event_cb("sight_event", {
                    "kind":     evt.kind,
                    "mode":     evt.mode,
                    "summary":  evt.summary,
                    "vlm_text": evt.vlm_text,
                    "score":    evt.score,
                })
            except Exception as e:
                _log.debug(f"[SIGHT] event_cb: {e}")

        # Per-kind callbacks
        for cb in self._callbacks.get(evt.kind, []):
            try:
                cb(evt)
            except Exception as e:
                _log.debug(f"[SIGHT] cb: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# §12  LUMINOVA CORE WIRE-UP
# ═════════════════════════════════════════════════════════════════════════════

def start_sight(brain, tree=None, use_hailo: bool = False) -> Optional[SightSystem]:
    """
    Drop-in replacement for all old camera/vision imports.
    Call once from luminova_core after Brain is constructed:

        from sight_system import start_sight
        sight = start_sight(brain, tree=tree)
    """
    try:
        emit_fn = (getattr(brain.__class__, "emit_event", None)
                   or getattr(brain, "emit_event",         None)
                   or getattr(brain, "_emit_event",        None))
        cb = (lambda t, d: emit_fn(t, d)) if callable(emit_fn) else None

        sight = SightSystem(event_callback=cb, use_hailo=use_hailo, tree=tree)

        # Direct VLM callback so there is no queue delay for voice response
        sight.on_event("vlm", lambda e: (
            hasattr(brain, "handle_vlm_result") and
            brain.handle_vlm_result({"description": e.vlm_text})
        ))

        # Hand brain a direct reference so it can call sight.query_scene()
        brain.sight = sight

        sight.start()
        _log.info("[SIGHT] wired into Luminova brain")
        return sight

    except Exception as e:
        _log.error(f"[SIGHT] start_sight: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# §13  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Luminova Sight System v5 — WxE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python sight_system.py\n"
            "  python sight_system.py --mode pose\n"
            "  python sight_system.py --query 'what do you see'\n"
            "  python sight_system.py --stats\n"
        ),
    )
    ap.add_argument("--mode",   choices=["detect", "pose", "idle"],
                    default=None, help="Force a starting mode")
    ap.add_argument("--query",  type=str, default=None,
                    help="Single VLM query then exit")
    ap.add_argument("--stats",  action="store_true",
                    help="Show memory stats and learned label reliability")
    ap.add_argument("--hailo",  action="store_true",
                    help="Use hailo-ollama VLM backend (port 8000)")
    args = ap.parse_args()

    B, R, G, Y = "\033[1m", "\033[0m", "\033[92m", "\033[93m"

    # ── --stats ───────────────────────────────────────────────────────────────
    if args.stats:
        mem = SightMemory()
        st  = mem.stats()
        print(f"\n{B}Sight Memory — {st['total_entries']} entries{R}")
        print(f"\n  Recent observations:")
        for e in mem.recent(10):
            print(f"    [{e.mode:7}] {e.ts[:16]}  {e.summary[:60]}")
        if st["learned_labels"]:
            print(f"\n{B}Learned label reliability:{R}")
            for lbl, info in list(st["learned_labels"].items())[:20]:
                bar = "█" * int(info["reliability"] * 10)
                print(f"  {lbl:20}  {info['reliability']:.2f}  {bar:12}  "
                      f"(✓{info['confirmed']} ✗{info['disputed']})")
        sys.exit(0)

    # ── --query ───────────────────────────────────────────────────────────────
    if args.query:
        print(f"\n{B}Scene query:{R} {args.query}")
        holder: List[str] = []
        sight = SightSystem(use_hailo=args.hailo)
        sight.on_event("vlm", lambda e: holder.append(e.vlm_text))
        sight.start()
        sight.query_scene(args.query)
        deadline = time.monotonic() + 30
        while not holder and time.monotonic() < deadline:
            time.sleep(0.2)
        sight.stop()
        if holder:
            print(f"\n  {G}{holder[0]}{R}\n")
        else:
            print(f"\n  {Y}[No response within 30s]{R}\n")
        sys.exit(0)

    # ── live run ──────────────────────────────────────────────────────────────
    mode_map = {
        "detect": SightMode.DETECT,
        "pose":   SightMode.POSE,
        "idle":   SightMode.IDLE,
    }

    sight = SightSystem(use_hailo=args.hailo)
    if args.mode:
        sight.force_mode(mode_map[args.mode])

    def _print(evt: SightEvent):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"  {ts}  [{evt.kind:6}]  {evt.summary[:80]}")

    for k in ("detect", "pose", "vlm", "mode", "error"):
        sight.on_event(k, _print)

    print(f"\n{B}Luminova Sight System v5{R}")
    print(f"  Detection : {Path(MODEL_DETECT).name}")
    print(f"              threshold={DET_THRESHOLD} · max={DET_MAX} · "
          f"temporal visible={DET_VISIBLE_FRAMES}/hidden={DET_HIDDEN_FRAMES}")
    print(f"  Pose      : {Path(MODEL_POSE).name}")
    print(f"              threshold={POSE_THRESHOLD} · plot={POSE_PLOT_MIN} · "
          f"refine_steps={POSE_REFINE_STEPS} · nms_r={POSE_NMS_RADIUS} · "
          f"temporal visible={POSE_VISIBLE_FRAMES}/hidden={POSE_HIDDEN_FRAMES}")
    print(f"  Mode      : {args.mode or 'auto — memory-driven adaptive'}")
    print(f"  VLM       : {'hailo-ollama → ' if args.hailo else ''}ollama ({VLM_FAST})")
    print(f"  Memory    : {MEM_IDX}")
    print(f"  Ctrl-C to stop\n")

    sight.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Y}Stopping…{R}")
        sight.stop()

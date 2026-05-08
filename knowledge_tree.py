#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  L U M I N O V A — K N O W L E D G E  T R E E                            ║
║  Three Growing Domains: Mathematics · Science · Art                        ║
║  Replaces Trinity Minds in luminova_core v4.0+                            ║
║  Author: WxE                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS IS
────────────
Not a character. Not a persona. A living, growing body of domain knowledge.

Where the Tree of Life holds Luminova's personal, episodic memory (what
happened, who she is, what she felt), the Knowledge Tree holds her growing
conceptual understanding of Mathematics, Science, and Art.

Every conversation plants new leaves. Leaves that are visited often grow
stronger. Leaves that are never touched slowly fade. The tree becomes a
reflection of what Luminova has learned about these three domains over her
lifetime.

THREE DOMAINS
─────────────
MATH     algebra · calculus · geometry · statistics · number theory · logic
SCIENCE  physics · chemistry · biology · astronomy · earth science · CS
ART      music · visual art · literature · poetry · film · design

HOW IT WORKS
────────────
1. ROUTE   — score the user's query against domain keyword sets.
             Strong match → that domain's context is injected.
             Weak or no match → no domain context (just personal memory).
             Multi-domain (music theory) → both math + art context merged.

2. RECALL  — pull the most relevant leaves from the matching domain(s).
             Relevance = word overlap + tag overlap + weight + recency.
             Returns a compact context string ready for LLM prompt injection.

3. GROW    — after every response, plant two leaves:
             • The user's question → "questions" branch (retrieval anchor)
             • The LLM's response lead → best-matching branch (knowledge)
             Weight starts at 0.65–0.80 depending on domain confidence score.

4. PRUNE   — background thread decays leaves 0.015/day (slower than ToL).
             Knowledge is more stable than episodic memory.
             Pinned leaves (seeds) never decay.

PERSISTENCE
───────────
~/luminova/knowledge/math.json
~/luminova/knowledge/science.json
~/luminova/knowledge/art.json

Auto-saves after every grow(). Snapshot = copy that folder.

USAGE IN luminova_core
──────────────────────
from knowledge_tree import KnowledgeTree

ktree = KnowledgeTree()
domain, ctx = ktree.context_for(user_query)
# inject ctx into LLM prompt
ktree.grow(user_query, llm_response, domain)
ktree.print_shape()
ktree.stats()
"""

import os
import re
import json
import math
import time
import uuid
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Set

log = logging.getLogger("KnowledgeTree")

# ═══════════════════════════════════════════════════════════════════════════════
# §1  PATHS + TUNING
# ═══════════════════════════════════════════════════════════════════════════════

KTREE_DIR = Path.home() / "luminova" / "knowledge"
KTREE_DIR.mkdir(parents=True, exist_ok=True)

# Leaf lifecycle
DECAY_RATE_PER_DAY  = 0.015   # knowledge decays slower than episodic memory
MIN_LEAF_WEIGHT     = 0.05    # below this, the leaf is pruned (unless pinned)
ACTIVE_MAX_LEAVES   = 35      # max leaves returned for LLM context
RELEVANCE_CUTOFF    = 0.08    # minimum relevance to enter context

# Domain routing
ROUTE_THRESHOLD     = 0.08    # minimum score to consider a domain active
MULTI_DOMAIN_RATIO  = 0.65    # secondary domain kept if score >= primary * ratio

# Context budget
CONTEXT_MAX_CHARS   = 1800    # per-domain context block character budget

# ═══════════════════════════════════════════════════════════════════════════════
# §2  DOMAIN KEYWORD SETS
#
# Three carefully curated sets. Words are matched against the lowercased query.
# Longer, more specific phrases score higher because they consume more query
# word-budget. Single-word matches are still counted but diluted.
#
# Rules for this list:
#   • Include root forms (algebra, algebraic, algebraically → just algebra)
#   • Include common abbreviations (calc, trig, bio, chem)
#   • Exclude words that appear in all three domains (e.g. "theory")
#   • Bias toward words a person actually types in conversation
# ═══════════════════════════════════════════════════════════════════════════════

MATH_KEYWORDS: Set[str] = {
    # ── identity
    "math", "maths", "mathematics", "mathematical",
    # ── operations + structures
    "number", "numbers", "numeric", "numerical", "integer", "integers",
    "fraction", "decimal", "ratio", "proportion", "percentage",
    "calculate", "calculation", "compute", "computation",
    "equation", "equations", "formula", "formulas", "expression",
    "solve", "solution", "solutions", "proof", "prove", "theorem",
    # ── algebra
    "algebra", "algebraic", "variable", "variables", "polynomial",
    "quadratic", "linear", "matrix", "matrices", "determinant",
    "vector", "vectors", "eigenvalue", "eigenvector", "transpose",
    "group theory", "ring", "field", "abstract algebra",
    # ── calculus + analysis
    "calculus", "calc", "derivative", "derivatives", "integral", "integrals",
    "differentiate", "differentiation", "integrate", "integration",
    "limit", "limits", "continuity", "differential", "gradient",
    "divergence", "curl", "laplacian", "fourier", "taylor", "series",
    # ── geometry + topology
    "geometry", "geometric", "angle", "angles", "triangle", "triangles",
    "circle", "sphere", "polygon", "polyhedron", "coordinate",
    "cartesian", "polar coordinates", "topology", "manifold",
    "euclidean", "non-euclidean", "pythagorean",
    # ── statistics + probability
    "statistics", "statistical", "probability", "distribution",
    "mean", "median", "mode", "variance", "standard deviation",
    "normal distribution", "regression", "correlation", "hypothesis",
    "bayesian", "random variable", "expected value", "entropy",
    # ── number theory
    "prime", "primes", "prime number", "divisor", "divisibility",
    "modular", "modulo", "congruence", "fibonacci", "golden ratio",
    "number theory", "cryptography", "factorisation",
    # ── discrete + logic
    "logic", "boolean", "truth table", "set theory", "sets",
    "combinatorics", "permutation", "combination", "graph theory",
    "algorithm", "complexity", "big o", "recursion", "induction",
    # ── functions + other
    "function", "functions", "domain", "range", "inverse",
    "logarithm", "exponential", "trigonometry", "trig",
    "sine", "cosine", "tangent", "radians", "degrees",
    "infinity", "asymptote", "symmetry", "transformation",
}

SCIENCE_KEYWORDS: Set[str] = {
    # ── identity
    "science", "scientific", "experiment", "hypothesis", "empirical",
    # ── physics
    "physics", "force", "forces", "energy", "mass", "weight",
    "velocity", "acceleration", "momentum", "impulse",
    "gravity", "gravitational", "friction", "tension",
    "quantum", "quantum mechanics", "wave", "particle",
    "electron", "proton", "neutron", "photon", "atom", "atomic",
    "nuclear", "radioactive", "fission", "fusion",
    "relativity", "spacetime", "thermodynamics", "entropy",
    "electricity", "electrical", "current", "voltage", "resistance",
    "magnetism", "magnetic", "electromagnetic", "field",
    "optics", "refraction", "reflection", "wavelength", "frequency",
    "sound", "acoustic", "vibration", "oscillation",
    "mechanics", "dynamics", "kinematics", "statics",
    # ── chemistry
    "chemistry", "chemical", "molecule", "molecules", "compound",
    "element", "elements", "periodic table", "ion", "ionic",
    "bond", "covalent", "hydrogen bond", "reaction", "reactions",
    "acid", "base", "pH", "oxidation", "reduction", "redox",
    "organic chemistry", "carbon", "polymer", "catalyst",
    "thermochemistry", "enthalpy", "entropy", "gibbs",
    # ── biology
    "biology", "biological", "cell", "cells", "organism",
    "dna", "rna", "gene", "genes", "genetics", "chromosome",
    "evolution", "natural selection", "species", "taxonomy",
    "photosynthesis", "respiration", "metabolism",
    "protein", "enzyme", "antibody", "immune", "virus", "bacteria",
    "neural", "neuron", "nervous system", "brain", "anatomy",
    "ecology", "ecosystem", "biome", "population",
    # ── astronomy
    "astronomy", "planet", "planets", "star", "stars", "galaxy",
    "universe", "black hole", "orbit", "telescope",
    "solar system", "moon", "sun", "mars", "jupiter",
    "nebula", "supernova", "dark matter", "dark energy",
    "black holes", "shooting star", "shooting stars",
    "big bang", "cosmic", "cosmology", "astrophysics",
    # ── earth + environment
    "geology", "geologic", "earthquake", "volcano", "tectonic",
    "climate", "atmosphere", "weather", "ocean", "hydrosphere",
    "fossil", "stratigraphy", "mineralogy", "erosion",
    "greenhouse", "carbon cycle", "water cycle",
    # ── computer science
    "algorithm", "data structure", "sorting", "searching",
    "computer", "programming", "code", "software", "hardware",
    "network", "protocol", "machine learning", "neural network",
    "ai", "artificial intelligence", "deep learning",
    "database", "operating system", "compiler", "binary",
}

ART_KEYWORDS: Set[str] = {
    # ── identity
    "art", "arts", "artistic", "creative", "creativity", "aesthetic",
    # ── music
    "music", "musical", "song", "songs", "melody", "melodies",
    "harmony", "harmonics", "rhythm", "rhythmic", "beat",
    "chord", "chords", "note", "notes", "scale", "scales",
    "tempo", "time signature", "dynamics", "pitch",
    "compose", "composition", "arrangement", "orchestration",
    "instrument", "piano", "guitar", "violin", "cello", "drums",
    "orchestra", "symphony", "sonata", "concerto", "fugue",
    "jazz", "classical music", "opera", "blues", "folk",
    "lyrics", "vocalist", "conductor", "score",
    # ── visual art
    "painting", "paintings", "drawing", "drawings", "sketch",
    "color", "colour", "palette", "canvas", "brushwork",
    "sculpture", "photography", "illustration",
    "perspective", "composition", "hue", "saturation", "contrast",
    "watercolor", "oil paint", "acrylic", "charcoal",
    "abstract", "portrait", "landscape", "still life",
    "impressionism", "cubism", "surrealism", "expressionism",
    "realism", "modernism", "contemporary art",
    # ── design
    "design", "graphic design", "typography", "layout",
    "user interface", "ux", "visual design", "brand",
    "industrial design", "product design",
    # ── literature + poetry
    "literature", "literary", "poetry", "poem", "poems", "poet",
    "prose", "novel", "novella", "short story", "fiction",
    "narrative", "metaphor", "simile", "symbolism", "allegory",
    "theme", "character", "plot", "setting", "conflict",
    "genre", "write", "writing", "author", "book", "read",
    "haiku", "sonnet", "epic", "lyric", "verse", "stanza",
    # ── film + performance
    "film", "films", "cinema", "movie", "movies", "screenplay",
    "theater", "theatre", "stage", "acting", "actor", "director",
    "dance", "choreography", "ballet", "performance",
    "animation", "cinematography", "editing", "montage",
    # ── architecture
    "architecture", "architectural", "building", "buildings",
    "structure", "facade", "floor plan", "form", "space",
    "baroque", "gothic", "modernist", "brutalism",
}

# ── Branch seeds per domain ────────────────────────────────────────────────────
# name → theme description (used for branch relevance matching)

MATH_BRANCHES: Dict[str, str] = {
    "arithmetic":    "basic arithmetic numbers counting fractions decimals",
    "algebra":       "algebra variables equations polynomials matrices vectors",
    "calculus":      "calculus derivatives integrals limits differential equations",
    "geometry":      "geometry shapes angles triangles circles coordinates topology",
    "statistics":    "statistics probability distributions mean variance regression",
    "number_theory": "number theory primes divisors modular arithmetic fibonacci",
    "logic":         "logic boolean sets combinatorics graph theory induction",
    "questions":     "math questions asked by user retrieval anchors",
}

SCIENCE_BRANCHES: Dict[str, str] = {
    "physics":        "physics mechanics energy force quantum relativity optics",
    "chemistry":      "chemistry molecules reactions acids bonds periodic table",
    "biology":        "biology cells DNA evolution genetics metabolism organism",
    "astronomy":      "astronomy stars planets galaxies black holes universe cosmos",
    "earth_science":  "geology climate atmosphere earth environment ecology",
    "computer_science":"algorithms data structures machine learning software AI",
    "questions":      "science questions asked by user retrieval anchors",
}

ART_BRANCHES: Dict[str, str] = {
    "music":               "music melody harmony rhythm composition instruments",
    "visual_art":          "painting drawing sculpture color composition perspective",
    "design":              "design typography layout graphic visual brand",
    "literature_poetry":   "literature poetry prose narrative metaphor writing",
    "film_performance":    "film cinema theater dance performance choreography",
    "architecture":        "architecture building structure form space design",
    "questions":           "art questions asked by user retrieval anchors",
}

# ── Seed knowledge planted once on first boot ─────────────────────────────────

MATH_SEEDS = [
    dict(branch="arithmetic",    content="Mathematics is the language of pattern, quantity, and structure. It is both discovered and invented.", weight=0.9, pin=True),
    dict(branch="algebra",       content="Algebra abstracts arithmetic into general rules using symbols. Every equation is a question about balance.", weight=0.9, pin=True),
    dict(branch="calculus",      content="Calculus describes change and accumulation. Derivatives measure rate of change; integrals measure accumulation.", weight=0.9, pin=True),
    dict(branch="geometry",      content="Geometry is the study of space, shape, and their relationships. Euclidean geometry underlies much of classical physics.", weight=0.85, pin=True),
    dict(branch="statistics",    content="Probability and statistics quantify uncertainty. The normal distribution appears naturally in many real-world phenomena.", weight=0.85, pin=True),
    dict(branch="number_theory", content="Number theory studies the integers. Prime numbers are the atoms of arithmetic — fundamental, mysterious, infinite.", weight=0.85, pin=True),
    dict(branch="logic",         content="Mathematical logic formalises reasoning. Set theory and boolean algebra underlie both mathematics and computation.", weight=0.85, pin=True),
]

SCIENCE_SEEDS = [
    dict(branch="physics",       content="Physics seeks the fundamental laws governing matter, energy, space, and time. From Newton to quantum field theory.", weight=0.9, pin=True),
    dict(branch="chemistry",     content="Chemistry studies matter and its transformations. All chemical behaviour emerges from electron interactions.", weight=0.9, pin=True),
    dict(branch="biology",       content="Biology is the study of life. All known life shares DNA as its information carrier and evolved from a common ancestor.", weight=0.9, pin=True),
    dict(branch="astronomy",     content="Astronomy studies the universe beyond Earth. The cosmos is 13.8 billion years old and still expanding.", weight=0.85, pin=True),
    dict(branch="earth_science", content="Earth science studies our planet's systems: geology, atmosphere, oceans, and climate interact as one system.", weight=0.85, pin=True),
    dict(branch="computer_science", content="Computer science studies computation, algorithms, and information. Every program is a formal proof executed by a machine.", weight=0.85, pin=True),
]

ART_SEEDS = [
    dict(branch="music",             content="Music organises sound in time. Harmony, melody, and rhythm are its three fundamental dimensions.", weight=0.9, pin=True),
    dict(branch="visual_art",        content="Visual art communicates through image, form, colour, and composition. It can express what language cannot.", weight=0.9, pin=True),
    dict(branch="design",            content="Design solves problems with beauty. Good design is invisible — it works so well you never notice it.", weight=0.85, pin=True),
    dict(branch="literature_poetry", content="Literature is human experience encoded in language. Poetry compresses meaning through rhythm, sound, and image.", weight=0.9, pin=True),
    dict(branch="film_performance",  content="Film is the synthesis of all other arts: image, sound, time, performance, narrative, and design.", weight=0.85, pin=True),
    dict(branch="architecture",      content="Architecture is the art of shaping space. It is both shelter and sculpture — the built environment shapes human experience.", weight=0.85, pin=True),
]


# ═══════════════════════════════════════════════════════════════════════════════
# §3  KNOWLEDGE LEAF
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeLeaf:
    """
    One unit of domain knowledge. Atomic, weighted, decaying.

    weight  : 0.0–1.0. Starts at planting weight, decays over time.
              Reinforced each time the leaf surfaces in context.
    pin     : True = immortal seed, never decays.
    source  : "seed" | "question" | "insight" | "concept"
    """

    def __init__(
        self,
        content:      str,
        domain:       str,
        branch:       str,
        tags:         List[str] = None,
        weight:       float     = 0.7,
        pin:          bool      = False,
        source:       str       = "insight",
        leaf_id:      str       = None,
        created_at:   str       = None,
        accessed_at:  str       = None,
        access_count: int       = 0,
    ):
        self.leaf_id      = leaf_id or str(uuid.uuid4())[:12]
        self.content      = content
        self.domain       = domain
        self.branch       = branch
        self.tags         = tags or []
        self.weight       = max(0.0, min(1.0, weight))
        self.pin          = pin
        self.source       = source
        self.created_at   = created_at or datetime.now(timezone.utc).isoformat()
        self.accessed_at  = accessed_at or self.created_at
        self.access_count = access_count

    # ── relevance ──────────────────────────────────────────────────────────────

    def relevance(self, query: str) -> float:
        """
        Score how relevant this leaf is to a query.
        Word overlap + tag overlap + weight bias + recency.
        No embeddings — the tree finds its own meaning.
        """
        if not query:
            return self.weight * 0.4

        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        c_words = set(re.findall(r"\b\w{3,}\b", self.content.lower()))
        t_words = {t.lower() for t in self.tags}

        content_overlap = len(q_words & c_words) / max(len(q_words), 1)
        tag_overlap     = len(q_words & t_words) / max(len(q_words), 1)

        try:
            age_days = (datetime.now(timezone.utc) -
                        datetime.fromisoformat(self.created_at)).days
            recency = math.exp(-age_days / 240)   # half-life ~8 months
        except Exception:
            recency = 0.5

        return round(min(1.0,
            content_overlap * 0.50 +
            tag_overlap     * 0.25 +
            self.weight     * 0.15 +
            recency         * 0.10
        ), 4)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def decay(self, days: float = 1.0):
        if not self.pin:
            self.weight = max(0.0, self.weight - DECAY_RATE_PER_DAY * days)

    def reinforce(self, amount: float = 0.04):
        self.weight       = min(1.0, self.weight + amount)
        self.accessed_at  = datetime.now(timezone.utc).isoformat()
        self.access_count += 1

    def is_alive(self) -> bool:
        return self.pin or self.weight >= MIN_LEAF_WEIGHT

    # ── serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "leaf_id":      self.leaf_id,
            "content":      self.content,
            "domain":       self.domain,
            "branch":       self.branch,
            "tags":         self.tags,
            "weight":       self.weight,
            "pin":          self.pin,
            "source":       self.source,
            "created_at":   self.created_at,
            "accessed_at":  self.accessed_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeLeaf":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__init__.__code__.co_varnames})

    def __repr__(self) -> str:
        pin = "📌" if self.pin else ""
        return (f"KLeaf({self.domain}/{self.branch} "
                f"w={self.weight:.2f}{pin} "
                f'"{self.content[:50]}…")')


# ═══════════════════════════════════════════════════════════════════════════════
# §4  KNOWLEDGE BRANCH
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeBranch:
    """
    A named cluster of leaves around a sub-topic.
    Permanent branches (seeds) never close.
    """

    def __init__(
        self,
        name:      str,
        domain:    str,
        theme:     str = "",
        permanent: bool = False,
        branch_id: str  = None,
        created_at: str = None,
    ):
        self.branch_id  = branch_id  or str(uuid.uuid4())[:8]
        self.name       = name
        self.domain     = domain
        self.theme      = theme or name
        self.permanent  = permanent
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.leaves:    List[KnowledgeLeaf] = []

    def add(self, leaf: KnowledgeLeaf):
        self.leaves.append(leaf)

    def prune(self) -> int:
        before = len(self.leaves)
        self.leaves = [l for l in self.leaves if l.is_alive()]
        return before - len(self.leaves)

    def is_alive(self) -> bool:
        return self.permanent or bool(self.leaves)

    def total_weight(self) -> float:
        return sum(l.weight for l in self.leaves)

    def relevance(self, query: str) -> float:
        """Average relevance of top leaves. Also scores branch theme against query."""
        if not self.leaves and not self.theme:
            return 0.0

        # Theme word overlap — lets empty-but-themed branches still route correctly
        q_words    = set(re.findall(r"\b\w{3,}\b", query.lower()))
        th_words   = set(re.findall(r"\b\w{3,}\b", self.theme.lower()))
        theme_score = len(q_words & th_words) / max(len(q_words), 1)

        if not self.leaves:
            return theme_score * 0.4

        leaf_scores = sorted(
            [l.relevance(query) for l in self.leaves], reverse=True
        )[:6]
        leaf_score = sum(leaf_scores) / len(leaf_scores)

        return round(theme_score * 0.35 + leaf_score * 0.65, 4)

    # ── serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "branch_id":  self.branch_id,
            "name":       self.name,
            "domain":     self.domain,
            "theme":      self.theme,
            "permanent":  self.permanent,
            "created_at": self.created_at,
            "leaves":     [l.to_dict() for l in self.leaves],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeBranch":
        b = cls(
            name       = d["name"],
            domain     = d["domain"],
            theme      = d.get("theme", d["name"]),
            permanent  = d.get("permanent", False),
            branch_id  = d.get("branch_id"),
            created_at = d.get("created_at"),
        )
        b.leaves = [KnowledgeLeaf.from_dict(ld) for ld in d.get("leaves", [])]
        return b

    def __repr__(self) -> str:
        return (f"KBranch({self.domain}/{self.name} "
                f"leaves={len(self.leaves)} w={self.total_weight():.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  KNOWLEDGE DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeDomain:
    """
    One of the three domains. Owns branches. Routes queries. Grows.

    ROUTE  — keyword scoring against self.keywords
    RECALL — pull relevant leaves → context block
    GROW   — plant question + insight leaves
    PRUNE  — decay + remove dead leaves
    """

    def __init__(self, name: str, keywords: Set[str],
                 branch_seeds: Dict[str, str]):
        assert name in ("math", "science", "art"), f"Unknown domain: {name}"
        self.name      = name
        self.keywords  = keywords
        self._lock     = threading.RLock()
        self.branches: Dict[str, KnowledgeBranch] = {}

        # Plant permanent seed branches
        for bname, theme in branch_seeds.items():
            self.branches[bname] = KnowledgeBranch(
                name=bname, domain=name, theme=theme, permanent=True
            )

    # ── routing ────────────────────────────────────────────────────────────────

    def score(self, query: str) -> float:
        """
        Return a routing score 0.0–1.0 for this domain against the query.

        Method:
          1. Count query words that appear in self.keywords (exact match).
          2. Count query bigrams that appear in self.keywords (phrase match).
          3. Score = hits / (query_word_count ** 0.6)  — penalises long queries
             so a 3-word match in a 3-word query scores higher than in a 20-word
             query. Capped at 1.0.
        """
        if not query:
            return 0.0

        lower = query.lower()
        words = re.findall(r"\b\w+\b", lower)
        if not words:
            return 0.0

        hits = 0

        # Unigrams
        for w in words:
            if w in self.keywords:
                hits += 1

        # Bigrams (catch "number theory", "black hole", "oil paint", etc.)
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i + 1]
            if bigram in self.keywords:
                hits += 1.5  # phrase match is worth more

        # Trigrams for a few important cases
        for i in range(len(words) - 2):
            trig = " ".join(words[i:i+3])
            if trig in self.keywords:
                hits += 2.0

        score = hits / max(len(words) ** 0.6, 1.0)
        return round(min(1.0, score), 4)

    # ── recall ─────────────────────────────────────────────────────────────────

    def context_block(self, query: str, max_chars: int = CONTEXT_MAX_CHARS) -> str:
        """
        Return a compact knowledge context string for LLM injection.
        Only includes leaves that score above RELEVANCE_CUTOFF.
        Sorted by relevance descending. Grouped by branch.
        """
        with self._lock:
            all_leaves: List[KnowledgeLeaf] = []
            for branch in self.branches.values():
                for leaf in branch.leaves:
                    score = leaf.relevance(query)
                    if score >= RELEVANCE_CUTOFF:
                        all_leaves.append((score, leaf))

        if not all_leaves:
            return ""

        all_leaves.sort(key=lambda x: x[0], reverse=True)
        top = all_leaves[:ACTIVE_MAX_LEAVES]

        # Reinforce surfaced leaves
        for _, leaf in top:
            leaf.reinforce(0.02)

        # Group by branch for readability
        by_branch: Dict[str, List[KnowledgeLeaf]] = {}
        for _, leaf in top:
            by_branch.setdefault(leaf.branch, []).append(leaf)

        domain_label = {
            "math":    "MATHEMATICS",
            "science": "SCIENCE",
            "art":     "ART & CREATIVITY",
        }.get(self.name, self.name.upper())

        lines = [f"=== {domain_label} KNOWLEDGE ==="]
        for bname, leaves in by_branch.items():
            if bname == "questions":
                continue   # questions are retrieval anchors, not context
            lines.append(f"[{bname.replace('_', ' ')}]")
            for leaf in leaves:
                pin = "★ " if leaf.pin else ""
                lines.append(f"  {pin}{leaf.content}")
        lines.append(f"=== END {domain_label} ===")

        result = "\n".join(lines)
        return result[:max_chars]

    # ── grow ───────────────────────────────────────────────────────────────────

    def plant(
        self,
        content:  str,
        branch:   str,
        tags:     List[str] = None,
        weight:   float     = 0.7,
        source:   str       = "insight",
        pin:      bool      = False,
    ) -> KnowledgeLeaf:
        """Plant one leaf into the given branch (creates branch if needed)."""
        with self._lock:
            if branch not in self.branches:
                self.branches[branch] = KnowledgeBranch(
                    name=branch, domain=self.name, theme=branch
                )
            leaf = KnowledgeLeaf(
                content=content,
                domain=self.domain_name,
                branch=branch,
                tags=tags or [],
                weight=weight,
                source=source,
                pin=pin,
            )
            self.branches[branch].add(leaf)
        return leaf

    @property
    def domain_name(self) -> str:
        return self.name

    def best_branch(self, query: str) -> str:
        """Return the name of the branch whose theme best matches the query."""
        if not self.branches:
            return "questions"
        scored = sorted(
            self.branches.items(),
            key=lambda kv: kv[1].relevance(query),
            reverse=True
        )
        # Skip "questions" — that's always the fallback
        for bname, _ in scored:
            if bname != "questions":
                return bname
        return "questions"

    # ── prune ──────────────────────────────────────────────────────────────────

    def prune(self, days: float = 1.0) -> int:
        with self._lock:
            removed = 0
            for leaf in self.all_leaves():
                leaf.decay(days)
            dead_branches = []
            for bname, branch in self.branches.items():
                removed += branch.prune()
                if not branch.is_alive() and not branch.permanent:
                    dead_branches.append(bname)
            for bname in dead_branches:
                del self.branches[bname]
                log.info(f"[{self.name}] Branch closed: {bname}")
        return removed

    def all_leaves(self) -> List[KnowledgeLeaf]:
        return [leaf for b in self.branches.values() for leaf in b.leaves]

    def leaf_count(self) -> int:
        return sum(len(b.leaves) for b in self.branches.values())

    # ── serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name":     self.name,
            "branches": {k: v.to_dict() for k, v in self.branches.items()},
        }

    @classmethod
    def _from_dict_into(cls, domain: "KnowledgeDomain", d: dict) -> None:
        """Restore branches from saved dict into an already-constructed domain."""
        for bname, bdata in d.get("branches", {}).items():
            domain.branches[bname] = KnowledgeBranch.from_dict(bdata)

    def __repr__(self) -> str:
        return (f"KDomain({self.name} "
                f"branches={len(self.branches)} "
                f"leaves={self.leaf_count()})")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  KNOWLEDGE TREE
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeTree:
    """
    Luminova's growing body of domain knowledge.
    Three domains: Mathematics · Science · Art

    PUBLIC API
    ──────────
    context_for(query)               → (domain_name | None, context_str)
    grow(query, response, domain)    → int  (leaves planted)
    plant_seeds()                    → plants foundational knowledge (once)
    snapshot()                       → saves all three domains to disk
    stats()                          → dict
    print_shape()                    → console summary

    INTERNAL
    ────────
    _route(query)                    → [(domain_name, score), ...]  sorted desc
    _save(domain_name)               → saves one domain to disk
    _load()                          → restores all domains from disk
    """

    def __init__(self, auto_save: bool = True):
        self.auto_save = auto_save
        self._lock     = threading.RLock()
        self._last_decay = time.time()

        self.domains: Dict[str, KnowledgeDomain] = {
            "math":    KnowledgeDomain("math",    MATH_KEYWORDS,    MATH_BRANCHES),
            "science": KnowledgeDomain("science", SCIENCE_KEYWORDS, SCIENCE_BRANCHES),
            "art":     KnowledgeDomain("art",     ART_KEYWORDS,     ART_BRANCHES),
        }

        self._load()

        # Plant seeds if this is a fresh tree
        total = sum(d.leaf_count() for d in self.domains.values())
        if total < 10:
            self.plant_seeds()
            log.info("🌱 Knowledge Tree seeds planted — first awakening")

        log.info(f"🌳 Knowledge Tree ready — "
                 + " | ".join(f"{n}: {d.leaf_count()} leaves"
                              for n, d in self.domains.items()))

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def context_for(self, query: str) -> Tuple[Optional[str], str]:
        """
        Route a query to the best-matching domain(s) and return their
        combined knowledge context for LLM prompt injection.

        Returns
        ───────
        (primary_domain, context_str)

        primary_domain  "math" | "science" | "art" | None
                        None if no domain scores above ROUTE_THRESHOLD.

        context_str     Ready-to-inject knowledge block string.
                        Empty string if no domain matches.

        Multi-domain behaviour
        ──────────────────────
        If a secondary domain scores >= primary * MULTI_DOMAIN_RATIO,
        its context is also included. Example: "music theory" → math + art.
        """
        ranked = self._route(query)

        if not ranked or ranked[0][1] < ROUTE_THRESHOLD:
            return None, ""

        primary_name, primary_score = ranked[0]
        ctx_parts = []

        for name, score in ranked:
            if score < ROUTE_THRESHOLD:
                break
            if name != primary_name and score < primary_score * MULTI_DOMAIN_RATIO:
                break
            ctx = self.domains[name].context_block(query)
            if ctx:
                ctx_parts.append(ctx)

        return primary_name, "\n".join(ctx_parts)

    def grow(
        self,
        query:    str,
        response: str,
        domain:   Optional[str],
    ) -> int:
        """
        Plant knowledge from one completed interaction.

        What is planted
        ───────────────
        1. "question" leaf  — the user's query, in the "questions" branch.
           Acts as a retrieval anchor: future similar questions will surface
           related knowledge because this leaf will score highly.

        2. "insight" leaf   — the lead of the LLM response (first 220 chars),
           planted in the best-matching branch for the query content.
           Weight = domain confidence score * 0.9, capped at 0.80.

        Why two leaves
        ──────────────
        The question leaf handles "I've heard this asked before" retrieval.
        The insight leaf handles "I know something about this topic" context.
        Both serve the LLM differently.

        Parameters
        ──────────
        query     User's raw input
        response  LLM's full response string
        domain    Domain name from context_for(), or None to auto-detect

        Returns
        ───────
        Number of leaves planted (0, 1, or 2).
        """
        if not query or not response:
            return 0

        # If no domain was pre-identified, try routing now
        if domain is None:
            ranked = self._route(query)
            if not ranked or ranked[0][1] < ROUTE_THRESHOLD:
                return 0
            domain = ranked[0][0]

        if domain not in self.domains:
            return 0

        dom       = self.domains[domain]
        ranked    = self._route(query)
        conf      = next((s for n, s in ranked if n == domain), 0.5)
        leaf_w    = min(0.80, max(0.50, conf * 0.9))

        # Tags from query keywords (top 6 most informative words)
        q_words   = re.findall(r"\b\w{4,}\b", query.lower())
        tags      = list(dict.fromkeys(q_words))[:6] + [domain]

        planted   = 0

        # ── 1. Question leaf ──────────────────────────────────────────────────
        q_content = query.strip()[:300]
        with self._lock:
            q_leaf = KnowledgeLeaf(
                content=q_content,
                domain=domain,
                branch="questions",
                tags=tags,
                weight=leaf_w * 0.85,
                source="question",
            )
            dom.branches["questions"].add(q_leaf)
            planted += 1

        # ── 2. Insight leaf ───────────────────────────────────────────────────
        lead = response.strip()[:220]
        if len(lead) > 30:               # skip trivial non-answers
            best_branch = dom.best_branch(query)
            with self._lock:
                i_leaf = KnowledgeLeaf(
                    content=lead,
                    domain=domain,
                    branch=best_branch,
                    tags=tags,
                    weight=leaf_w,
                    source="insight",
                )
                dom.branches[best_branch].add(i_leaf)
                planted += 1

        log.info(f"[GROW] {domain} +{planted} leaves "
                 f"(conf={conf:.2f} w={leaf_w:.2f}) "
                 f"'{query[:50]}'")

        if self.auto_save:
            self._save(domain)

        return planted

    def plant_seeds(self):
        """
        Plant foundational knowledge seeds for all three domains.
        Called once on first boot. All seeds are pinned (never decay).
        """
        seed_map = {
            "math":    MATH_SEEDS,
            "science": SCIENCE_SEEDS,
            "art":     ART_SEEDS,
        }
        total = 0
        for dname, seeds in seed_map.items():
            dom = self.domains[dname]
            for s in seeds:
                leaf = KnowledgeLeaf(
                    content=s["content"],
                    domain=dname,
                    branch=s["branch"],
                    tags=[dname, s["branch"]],
                    weight=s["weight"],
                    pin=s.get("pin", True),
                    source="seed",
                )
                with self._lock:
                    dom.branches[s["branch"]].add(leaf)
                total += 1
            if self.auto_save:
                self._save(dname)

        log.info(f"🌱 Knowledge seeds planted: {total} leaves across 3 domains")

    def snapshot(self):
        """Save all three domains to disk."""
        for name in self.domains:
            self._save(name)
        log.info(f"📸 Knowledge Tree snapshot saved")

    def prune(self) -> dict:
        """
        Apply time-based decay and remove dead leaves.
        Called by background thread once a day.
        """
        now = time.time()
        days = (now - self._last_decay) / 86400.0
        self._last_decay = now

        results = {}
        for name, dom in self.domains.items():
            removed = dom.prune(days) if days > 0 else 0
            results[name] = removed
            log.info(f"[PRUNE] {name}: -{removed} leaves, "
                     f"{dom.leaf_count()} remaining")

        self.snapshot()
        return results

    def stats(self) -> dict:
        return {
            "domains": {
                name: {
                    "total_leaves": dom.leaf_count(),
                    "branches":     len(dom.branches),
                    "by_branch": {
                        bname: len(b.leaves)
                        for bname, b in dom.branches.items()
                        if b.leaves
                    },
                }
                for name, dom in self.domains.items()
            },
            "total_leaves": sum(d.leaf_count() for d in self.domains.values()),
        }

    def print_shape(self):
        """Print a visual summary of the tree to console."""
        print("\n📚 KNOWLEDGE TREE\n")
        for dname, dom in self.domains.items():
            label = {"math": "MATHEMATICS", "science": "SCIENCE",
                     "art": "ART & CREATIVITY"}.get(dname, dname.upper())
            print(f" ║ [{label}]  ({dom.leaf_count()} leaves)")
            for bname, branch in dom.branches.items():
                if not branch.leaves:
                    continue
                pins = sum(1 for l in branch.leaves if l.pin)
                print(f" ╠══ {bname:<22} {len(branch.leaves):>3} leaves"
                      f"{' 📌' if pins else ''}")
            print(" ║")
        print()

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════════════════

    def _route(self, query: str) -> List[Tuple[str, float]]:
        """
        Score all three domains against the query.
        Returns list of (domain_name, score) sorted descending.
        Scores above 0 only.
        """
        scored = [
            (name, dom.score(query))
            for name, dom in self.domains.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(n, s) for n, s in scored if s > 0]

    def _save(self, domain_name: str):
        path = KTREE_DIR / f"{domain_name}.json"
        dom  = self.domains[domain_name]
        try:
            with self._lock:
                data = dom.to_dict()
            with open(path, "w") as f:
                json.dump({
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "domain":   data,
                }, f, indent=2)
        except Exception as e:
            log.error(f"[SAVE] {domain_name}: {e}")

    def _load(self):
        for name in self.domains:
            path = KTREE_DIR / f"{name}.json"
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    raw = json.load(f)
                KnowledgeDomain._from_dict_into(
                    self.domains[name], raw.get("domain", {})
                )
                log.info(f"[LOAD] {name}: "
                         f"{self.domains[name].leaf_count()} leaves restored")
            except Exception as e:
                log.error(f"[LOAD] {name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# §7  BACKGROUND PRUNER
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgePruner:
    """
    Background daemon thread. Prunes the Knowledge Tree once every 24 hours.
    Start it and forget it.
    """

    def __init__(self, ktree: KnowledgeTree, interval_hours: float = 24.0):
        self.ktree    = ktree
        self.interval = interval_hours * 3600
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="KnowledgePruner"
        )

    def start(self):
        self._thread.start()
        log.info("🍂 KnowledgePruner started (daily)")

    def _run(self):
        while True:
            time.sleep(self.interval)
            try:
                result = self.ktree.prune()
                log.info(f"KnowledgePruner: {result}")
            except Exception as e:
                log.error(f"KnowledgePruner error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# §8  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import shutil
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s"
    )

    # Use a temp dir so the self-test doesn't pollute the real tree
    import tempfile
    _tmp = tempfile.mkdtemp()
    _orig_dir = KTREE_DIR
    import knowledge_tree as _self
    _self.KTREE_DIR = Path(_tmp)

    print("\n🌱 Knowledge Tree self-test\n")

    ktree = KnowledgeTree()
    ktree.print_shape()

    # ── Routing tests ──────────────────────────────────────────────────────────
    test_queries = [
        ("What is the derivative of x squared?",  "math"),
        ("How does photosynthesis work?",          "science"),
        ("Explain impressionism in painting",      "art"),
        ("What is music theory?",                  "art"),    # also math signal
        ("Tell me about black holes",              "science"),
        ("How do primes relate to cryptography?",  "math"),
        ("What makes a good short story?",         "art"),
        ("How warm is it today?",                  None),     # no domain
    ]

    print("── Routing ──────────────────────────────────────────────")
    for q, expected in test_queries:
        domain, ctx = ktree.context_for(q)
        ok = "✅" if domain == expected else "⚠️ "
        print(f" {ok} '{q[:45]:<45}' → {str(domain):<10}  (expected: {str(expected)})")

    print()

    # ── Grow test ──────────────────────────────────────────────────────────────
    planted = ktree.grow(
        "What is the Pythagorean theorem?",
        "The Pythagorean theorem states that in a right triangle, "
        "a² + b² = c², where c is the hypotenuse.",
        "math"
    )
    print(f"Grow test: planted {planted} leaves")

    # ── Context retrieval after growth ────────────────────────────────────────
    domain, ctx = ktree.context_for("right triangle hypotenuse")
    print(f"\nContext for 'right triangle hypotenuse' (domain={domain}):")
    print("─" * 50)
    print(ctx[:600] if ctx else "(no context)")
    print("─" * 50)

    import json as _json
    print(f"\nStats: {_json.dumps(ktree.stats(), indent=2)}")

    # Cleanup
    shutil.rmtree(_tmp, ignore_errors=True)
    print("\n✅ Knowledge Tree self-test complete.\n")

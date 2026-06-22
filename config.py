# Ground truth for all user-facing configuration.
# Edit this file to change provider, model, or pipeline behavior.
# All values here are read by cli.py and passed into the pipeline.

PROVIDER = "google"  # Options: "anthropic", "openai", "groq", "ollama", "google", "google-fast"

MODELS = {
    "anthropic":   "claude-sonnet-4-20250514",
    "openai":      "gpt-4o",
    "groq":        "llama-3.1-70b-versatile",
    "ollama":      "llama3.1",
    "google":      "gemini-3-flash-preview",     # "gemini-2.5-flash-lite"
    "google-fast": "gemma-4-31b-it",
}

PIPELINE = {
    # ── Generation ──────────────────────────────────────────────────────────────
    "max_slides":           16,      # max slides per deck — Planner cap (4–20)
    "writer_batch_size":    5,       # slides per Writer API call — stays within output token limits
    # ── Extraction ──────────────────────────────────────────────────────────────
    # extractor: which PDF extractor to use.
    #   "pymupdf4llm" — lite path; no model-weight downloads (default)
    #   "mineru"      — quality path; bundles layout/formula/table models
    "extractor":            "mineru",
    # ── Chunking ────────────────────────────────────────────────────────────────
    "chunk_size":           8_000,   # target chunk size in characters (~2k tokens)
    "overlap_size":         1_500,   # sliding window overlap between chunks (~300 words)
    # ── Routing ─────────────────────────────────────────────────────────────────
    # multi_deck_chapter_threshold: number of level-1 chapters above which
    #   multi-deck mode is considered. Must also exceed length threshold.
    # multi_deck_length_threshold: minimum document size in characters.
    #   ~40,000 chars ≈ 20 pages of dense academic text.
    #   Both conditions must be true for multi-deck mode to activate.
    "multi_deck_chapter_threshold": 3,       # minimum level-1 chapters for multi-deck
    "multi_deck_length_threshold":  100_000,  # minimum characters for multi-deck (~20 pages)
    # ── Concurrency ─────────────────────────────────────────────────────────────
    "max_concurrent":       None,    # None = use provider-aware default (5 for cloud, 1 for ollama)
    # ── Retry and format ────────────────────────────────────────────────────────
    "max_format_retries":   3,       # per-agent JSON format retry limit
    "max_review_cycles":    3,       # Critic/Refiner loop kill switch
    # ── Resiliency ──────────────────────────────────────────────────────────────
    "request_timeout":            600,  # seconds before a hung call is cancelled and retried
    "max_rate_limit_retries":     6,    # tenacity stop_after_attempt for 429/5xx errors
    "backoff_wait_min":           4,    # tenacity wait_exponential min seconds
    "backoff_wait_max":           60,   # tenacity wait_exponential max seconds
    "circuit_breaker_threshold":  5,    # consecutive failures before the circuit opens
    "circuit_breaker_cooldown":   60,   # seconds to wait before probing after the circuit opens
    # ── Output ──────────────────────────────────────────────────────────────────
    # duplicate_policy: what to do with previous outputs for the same paper.
    #   "overwrite" — delete older outputs for the same title before writing
    #   "keep_both" — keep every run; timestamped filenames never collide
    "duplicate_policy":     "overwrite",
    # ── Viewer ──────────────────────────────────────────────────────────────────
    "port":                 7654,    # localhost port for the --open HTML server
    # ── Debug ───────────────────────────────────────────────────────────────────
    "debug":                False,   # write intermediate agent outputs to disk
}

# ── Optional settings.json overrides ──────────────────────────────────────────
# The web UI writes settings.json to the project root.  config.py applies those
# overrides here so every entry point (server.py, cli.py) sees the same values.
import json as _json
import sys  as _sys
from pathlib import Path as _Path

_SETTINGS_PATH = _Path(__file__).parent / "settings.json"

if _SETTINGS_PATH.exists():
    try:
        _overrides = _json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except _json.JSONDecodeError as _e:
        _sys.exit(f"settings.json is malformed: {_e}")

    if "PROVIDER" in _overrides:
        PROVIDER = _overrides["PROVIDER"]
    if "MODELS" in _overrides:
        MODELS = _overrides["MODELS"]
    if "PIPELINE" in _overrides:
        _unknown = set(_overrides["PIPELINE"]) - set(PIPELINE)
        if _unknown:
            _sys.exit(
                f"settings.json error: unknown PIPELINE keys {sorted(_unknown)}. "
                f"Valid keys: {sorted(PIPELINE)}"
            )
        PIPELINE.update(_overrides["PIPELINE"])

    if PROVIDER not in MODELS:
        _sys.exit(
            f"settings.json error: PROVIDER={PROVIDER!r} is not a key in MODELS. "
            f"Available keys: {list(MODELS.keys())}"
        )

if PIPELINE["duplicate_policy"] not in ("overwrite", "keep_both"):
    _sys.exit(
        f"config error: duplicate_policy={PIPELINE['duplicate_policy']!r} "
        f"must be 'overwrite' or 'keep_both'."
    )

if PIPELINE["extractor"] not in ("pymupdf4llm", "mineru"):
    _sys.exit(
        f"config error: extractor={PIPELINE['extractor']!r} "
        f"must be 'pymupdf4llm' or 'mineru'."
    )

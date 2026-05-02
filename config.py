# Ground truth for all user-facing configuration.
# Edit this file to change provider, model, or pipeline behavior.
# All values here are read by cli.py and passed into the pipeline.

PROVIDER = "google"  # Options: "anthropic", "openai", "groq", "ollama", "google", "google-fast"

MODELS = {
    "anthropic":   "claude-sonnet-4-20250514",
    "openai":      "gpt-4o",
    "groq":        "llama-3.1-70b-versatile",
    "ollama":      "llama3.1",
    "google":      "gemini-2.5-flash",     # "gemini-2.5-flash-lite"
    "google-fast": "gemma-4-31b-it",
}

PIPELINE = {
    # ── Generation ──────────────────────────────────────────────────────────────
    "max_slides":           16,      # hard cap on slides per deck
    "writer_batch_size":    5,       # slides per Writer API call — stays within output token limits
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
    # ── Viewer ──────────────────────────────────────────────────────────────────
    "port":                 7654,    # localhost port for the --open HTML server
    # ── Debug ───────────────────────────────────────────────────────────────────
    "debug":                False,   # write intermediate agent outputs to disk
}

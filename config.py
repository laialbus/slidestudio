PROVIDER = "google-fast"  # Options: "anthropic", "openai", "groq", "ollama", "google", "google-fast"

MODELS = {
    "anthropic":   "claude-sonnet-4-20250514",
    "openai":      "gpt-4o",
    "groq":        "llama-3.1-70b-versatile",
    "ollama":      "llama3.1",
    "google":      "gemini-3.1-flash-lite-preview",
    "google-fast": "gemini-2.0-flash",
}

PIPELINE = {
    "max_slides":           16,
    "chunk_size":           8_000,
    "overlap_size":         1_500,
    "multi_deck_threshold": 3,
    "max_concurrent":       None,
    "writer_batch_size":    5,
    "max_format_retries":   3,
    "max_review_cycles":    3,
    "port":                 7654,
    "debug":                False,
}

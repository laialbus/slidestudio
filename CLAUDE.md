# SlideStudio

Agentic pipeline that transforms academic PDFs into reel-style teaching slides with scrolling mechanism.
No hosted backend — users run the pipeline locally with their own API key.

## Architecture
See ARCHITECTURE.md for the full design. Key rules:

- Agents communicate in-memory (Python dicts). No file I/O between steps.
- Every agent output is validated by a Pydantic schema before the next agent runs.
- Schemas define shape and validation only — no methods except simple computed properties like all_passed on Critique
- All API calls go through BaseProvider.complete_json() — never call the API directly.
- All filesystem paths go through slugify() before touching disk.
- The semaphore in utils/rate_limiter.py is the single concurrency budget for all providers.
- The pipeline always ends with SlidesFinal or DeckIndex in memory
- Exporters consume pipeline output — they never call agents or providers
- To add a new output format, add a file in exporters/ that inherits
  BaseExporter. NOTE: `BaseExporter` and `exporters/{base,pptx,gui}.py` are
  reserved, empty placeholder stubs — not implemented yet. The only live
  exporter today is the static viewer in `exporters/html/`.

## Entry points
- `python cli.py paper.pdf` — run the full pipeline
- `python cli.py paper.pdf --estimate` — dry-run cost estimate, no API calls
- `python cli.py paper.pdf --debug` — write intermediate agent outputs to disk
- `python cli.py paper.pdf --open` — generate and open viewer in browser

## Provider config
Set PROVIDER and MODEL in config.py. Add your API key to .env (never commit it).

## Do not
- Add methods to schema classes that fetch, transform, or process data
- Import agents, providers, pipeline, or utils from inside schemas/
- Call provider APIs outside of providers/
- Write output paths as raw strings — always use slugify()

## Coding Standards
1. **Configuration vs. Domain Constants**
    - `config.py` is the single source of truth for all user-facing settings. It contains three top-level names only:
        - `PROVIDER` — the active provider string
        - `MODELS` — a dict mapping provider names to model strings
        - `PIPELINE` — a dict of runtime tunables (thresholds, limits, batch sizes, flags)
    - Internal implementation constants (library flags like `BLOCK_TYPE_IMAGE = 1`, HTTP status codes, regex patterns) stay localized at the top of the file that uses them. Do not move them to `config.py`.
    - If unsure whether a value belongs in `config.py` or local to its file, keep it local. Do not promote it without a clear reason.

2. **Dependency Injection (Strict)**
    - Classes must NEVER import `config.py` — not in methods, not at the top of the file, not anywhere.
    - Do not use default arguments for **config-sourced values** in method signatures. Any value that originates from `config.py` must be a strictly required parameter (no default) in `__init__` or `run()`. The caller is 100% responsible for providing it.
    - Optional **behavioural flags** — parameters whose `None` value is itself a meaningful instruction rather than a missing config value — may use `= None` as a default. A parameter qualifies as a behavioural flag if:
        * It changes per-call rather than per-session
        * `None` produces a distinct, well-defined code path
        * No entry in `config.py` could ever supply its value
    - Current example: `Planner.run(scope=None)` — `None` means "full document mode", not "config value was forgotten".

3. **Config Ownership**
    - `cli.py`, `pipeline.py`, and `server.py` are the only files that may import `config.py` directly. All other modules receive values through constructor parameters or function arguments.

## Code Style
@PYTHON_STYLE.md

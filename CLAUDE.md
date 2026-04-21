# SlideStudio

Agentic pipeline that transforms academic PDFs into reel-style teaching slides.
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
- To add a new output format, add a file in exporters/ that inherits BaseExporter

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

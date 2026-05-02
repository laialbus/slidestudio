# SlideStudio

Turn any academic PDF into a polished, scrollable slide deck — locally, with your own API key.

SlideStudio runs a multi-agent pipeline (Analyst → Planner → Writer → Critic/Refiner) that understands your document, designs a slide arc, drafts content, and self-reviews for accuracy. Short papers produce a single deck; long textbooks produce a chapter-per-deck index you can browse like a table of contents.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourname/slidestudio
cd slidestudio
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run on any PDF

```bash
python cli.py run path/to/your/paper.pdf --open
```

Slides are generated and opened in your browser automatically.

---

## CLI Subcommands

SlideStudio exposes four subcommands. Run `python cli.py --help` to list them.

### `run` — generate slides

```bash
python cli.py run paper.pdf
```

| Flag | Description |
|------|-------------|
| `--open` | Serve `outputs/` on localhost and open the viewer in your browser after generation |
| `--fast` | Skip the Critic/Refiner review loop entirely; faster but lower quality |
| `--debug` | Write all intermediate agent outputs (plan, draft, critique) to `outputs/debug/` for inspection |
| `--resume` | Load completed stages from the checkpoint cache and continue from the last completed step |
| `--force` | Ignore the checkpoint cache and run fresh, overwriting any cached stages |
| `--max-concurrent N` | Override the default concurrency limit (default: 5 for cloud providers, 1 for Ollama) |
| `--provider NAME` | Override the provider set in `config.py` (e.g. `anthropic`, `openai`) |

#### Examples

```bash
# Full run with browser viewer
python cli.py run paper.pdf --open

# Fast run without review cycle
python cli.py run paper.pdf --fast --open

# Resume a run that was interrupted
python cli.py run paper.pdf --resume

# Debug intermediate outputs
python cli.py run paper.pdf --debug
```

### `estimate` — dry-run cost estimate

Runs PDF extraction locally and prints the anticipated token usage and cost. No API calls are made.

```bash
python cli.py estimate paper.pdf
```

#### Example output

```
{'mode': 'single-deck', 'decks': 1, 'api_calls': 8, 'input_tokens': 12400, ...}
```

### `serve` — open viewer for an existing output

Opens the browser viewer for a previously generated output without re-running the pipeline. Useful for re-inspecting results or sharing a deck after the fact.

```bash
python cli.py serve paper.pdf
```

If no output exists for the given PDF, prints a clear error and exits:

```
No output found for 'paper.pdf'.
Run `python cli.py run paper.pdf` first to generate slides.
```

### `library-refresh` — rebuild the library index

Scans the `outputs/` directory and regenerates `outputs/library.json` from scratch. Run this if you manually copied, moved, or deleted output files and the in-browser library no longer reflects the current state.

```bash
python cli.py library-refresh
```

Under normal use you never need this — every `run` call automatically upserts the new entry into `library.json`. Use `library-refresh` as a recovery tool.

---

## Output formats

### Single-deck (short documents)

Documents with **3 or fewer chapters** produce a single JSON file:

```
outputs/
├── library.json                    ← auto-maintained library index
└── attention-is-all-you-need.json
```

Open with `--open` to view the scrollable slide reel.

### Multi-deck (long documents)

Documents with **more than 3 chapters** produce one deck per chapter plus an index:

```
outputs/
├── library.json                    ← auto-maintained library index
└── biology-101-textbook/
    ├── index.json              ← table of contents
    ├── 01_introduction.json
    ├── 02_cell_structure.json
    └── 03_cellular_respiration.json
```

The viewer shows a chapter grid on load; clicking a chapter transitions into its slide reel. Use the **Back** button or keyboard arrows (← →) to navigate.

### `library.json`

Every `run` automatically upserts an entry into `outputs/library.json` — a flat JSON array sorted newest-first by `generated_at`. Each entry records title, file path, type (`single_deck` / `multi_deck`), slide count, provider, and model. The in-browser library reads this file to populate the home screen.

---

## Changing the provider or model

Edit `config.py`:

```python
PROVIDER = "anthropic"          # anthropic | openai | groq | ollama

MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai":    "gpt-4o",
    ...
}
```

Add the corresponding key to `.env`:

```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

---

## Library viewer

The `--open` flag (or `serve`) starts a local HTTP server and opens `exporters/html/index.html` in your browser. The viewer is a single-page app with three states:

| State | What you see |
|-------|-------------|
| **Library** | Card grid of every generated deck, sorted newest-first. Shows title, type chip (Single / Multi-deck), date, slide count, and model. Includes a live search bar. |
| **TOC** | Chapter list for a multi-deck paper, or a direct jump to the reel for single-deck. Breadcrumb shows `Library › Paper Title`. |
| **Reel** | Scrollable slide deck. Breadcrumb shows `Library › Paper Title › Chapter`. |

Click any breadcrumb segment to navigate back. The viewer also supports deep-linking via the `?file=` query parameter (used automatically by `serve`).

---

## Troubleshooting

### `AuthenticationError` or `401 Unauthorized`

Your API key is missing or incorrect. Check:

1. `.env` exists in the project root (not `.env.example`)
2. The key matches the active `PROVIDER` in `config.py`
3. The key has not expired or been revoked

```bash
# Verify the file is present and contains a key:
cat .env
```

### `ModuleNotFoundError`

The virtual environment is not activated, or dependencies are not installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Viewer shows a blank page

The `--open` flag serves files on `localhost`. If the page is blank, check the browser console for errors. The most common cause is opening `index.html` directly via `file://` — always use the `--open` flag so files are served over HTTP.

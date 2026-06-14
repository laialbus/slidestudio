# SlideStudio

Turn any academic PDF into a polished, scrollable slide deck — locally, with your own API key.

SlideStudio runs a multi-agent pipeline (Analyst → Planner → Writer → Critic/Refiner) that understands your document, designs a slide arc, drafts content, and self-reviews for accuracy. Short papers produce a single deck; long textbooks produce a chapter-per-deck index you can browse like a table of contents.

> This codebase was written almost entirely by [Claude](https://www.anthropic.com/claude), with a small number of changes contributed by [Gemini 3](https://deepmind.google/technologies/gemini/).

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourname/slidestudio
cd slidestudio
```

### 2. Create a virtual environment

Python 3.12 is required — ML dependencies (Surya, PyTorch) do not yet ship pre-built wheels for Python 3.13+.

```bash
python3.12 -m venv venv
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

Open `.env` and add the key for your chosen provider. The default provider is Google:

```
GOOGLE_API_KEY=AIza...
```

Other supported providers:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### 5. Start the web studio

```bash
python cli.py serve
```

This opens `http://localhost:7654` in your browser. From here you can upload PDFs, track generation progress, browse your library, and configure provider settings — no further command-line interaction needed.

#### CLI alternative

If you prefer the command line:

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

### `serve` — start the web studio

Starts the local FastAPI server and opens the browser. With no argument it opens the library home screen — you can upload PDFs and manage your library from there without using the command line.

```bash
python cli.py serve
```

Pass a PDF path to jump straight to that deck's viewer:

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
└── attention_is_all_you_need_3f9a2b7c_20260609T142755.json
```

Filenames are `{filename}_{hash}_{YYYYMMDDTHHMMSS}.json` — the source PDF's
filename, a short hash of its **content**, and the UTC generation timestamp.
The content hash is the document's identity: it is how the overwrite policy
recognises a re-run of the same paper regardless of how its title was extracted
or whether the file was renamed (see *Regenerating the same paper*). That
identity is stored as a field inside each deck JSON and in `library.json` — the
hash in the filename is just a human/grep convenience and is never parsed back,
so the naming scheme can change without affecting behaviour. The timestamp keeps
successive runs distinct. Open with `--open` to view the scrollable slide reel.

### Multi-deck (long documents)

Documents with **more than 3 chapters** produce one deck per chapter plus an index:

```
outputs/
├── library.json                    ← auto-maintained library index
└── biology_101_textbook_8c1d4e0a_20260609T142755/
    ├── index.json              ← table of contents
    ├── 01_introduction.json
    ├── 02_cell_structure.json
    └── 03_cellular_respiration.json
```

The viewer shows a chapter grid on load; clicking a chapter transitions into its slide reel. Use the **Back** button or keyboard arrows (← →) to navigate.

### `library.json`

Every `run` automatically upserts an entry into `outputs/library.json` — a flat JSON array sorted newest-first by `generated_at`. Each entry records title, file path, type (`single_deck` / `multi_deck`), slide count, provider, and model. The in-browser library reads this file to populate the home screen, so the timestamped filenames never need to be read by a human — the manifest is the title-to-file trace.

### Regenerating the same paper

`PIPELINE["duplicate_policy"]` in `config.py` (also settable from the web UI settings panel) controls what happens when you regenerate a paper whose slides already exist in `outputs/`:

- `"overwrite"` (default) — older outputs for the same PDF are deleted, and their stale `library.json` entries pruned, **after** the new deck is written, so a crash mid-generation never destroys the prior deck without producing a replacement. "Same PDF" is matched by a hash of the file's **content**, not its title or filename, so a re-run overwrites even if the extracted title drifted or the source file was renamed. Archived copies are never touched.
- `"keep_both"` — every generation is kept side by side; the timestamp in the filename keeps them distinct.

If a run fails partway, its completed stages are cached. In the **web UI** a failed generation shows a **Resume** button that continues from the last completed stage (the CLI equivalent is `--resume`). A resumed run that succeeds overwrites the prior deck just like a fresh one — cleanup always runs after a successful write, never before. A successful run clears its own stage cache, so a later regeneration of the same PDF always starts fresh.

---

## Provider and model

The easiest way is the **Settings panel** — click the gear icon (top-right of the browser UI), pick a provider, and save. The server picks up the change on the next job without restarting.

To configure manually, edit `config.py`:

```python
PROVIDER = "google"    # anthropic | openai | google | google-fast | ollama

MODELS = {
    "anthropic":   "claude-sonnet-4-20250514",
    "openai":      "gpt-4o",
    "google":      "gemini-3-flash-preview",
    "google-fast": "gemma-4-31b-it",
    "ollama":      "llama3.1",
}
```

Alternatively, create a `settings.json` in the project root (the web UI writes this automatically):

```json
{
  "PROVIDER": "anthropic",
  "MODELS": {
    "anthropic": "claude-sonnet-4-20250514"
  }
}
```

`settings.json` overrides `config.py` without modifying it. Delete it to revert to `config.py` defaults. Add the key for the chosen provider to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

---

## Library viewer

The `--open` flag (or `serve`) starts a local FastAPI server and opens `exporters/html/index.html` in your browser. The viewer is a single-page app with four states:

| State | What you see |
|-------|-------------|
| **Library** | Card grid of every generated deck, sorted newest-first. Shows title, type chip (Single / Multi-deck), date, slide count, and model. Includes a live search bar and archive/restore actions. |
| **TOC** | Chapter list for a multi-deck paper, or a direct jump to the reel for single-deck. Breadcrumb shows `Library › Paper Title`. |
| **Reel** | Scrollable slide deck. Breadcrumb shows `Library › Paper Title › Chapter`. |
| **Settings** | Gear icon (top-right) opens a panel to change the active provider and model, and toggle dark/light theme. |

Click any breadcrumb segment to navigate back. The viewer also supports deep-linking via the `?file=` query parameter (used automatically by `serve`).

### Upload via browser

With the server running, use the upload button on the library home screen to submit a PDF. The server processes it in the background and the library updates automatically when the job completes — no CLI needed.

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

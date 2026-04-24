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
python cli.py path/to/your/paper.pdf --open
```

Slides are generated and opened in your browser automatically.

---

## CLI Flags

| Flag | Description |
|------|-------------|
| *(none)* | Generate slides and save to `outputs/` |
| `--estimate` | Dry-run only — print estimated token usage and cost without making any API calls |
| `--fast` | Skip the Critic/Refiner review loop entirely; faster but lower quality |
| `--debug` | Write all intermediate agent outputs (plan, draft, critique) to `outputs/debug/` for inspection |
| `--open` | Serve `outputs/` on localhost and open the viewer in your browser after generation |
| `--max-concurrent N` | Override the default concurrency limit (default: 5 for cloud providers, 1 for Ollama) |

### Examples

```bash
# Estimate cost before spending any money
python cli.py paper.pdf --estimate

# Full run with browser viewer
python cli.py paper.pdf --open

# Fast run without review cycle
python cli.py paper.pdf --fast --open

# Debug intermediate outputs
python cli.py paper.pdf --debug
```

---

## Output formats

### Single-deck (short documents)

Documents with **3 or fewer chapters** produce a single JSON file:

```
outputs/
└── attention-is-all-you-need.json
```

Open with `--open` to view the scrollable slide reel.

### Multi-deck (long documents)

Documents with **more than 3 chapters** produce one deck per chapter plus an index:

```
outputs/
└── biology-101-textbook/
    ├── index.json              ← table of contents
    ├── 01_introduction.json
    ├── 02_cell_structure.json
    └── 03_cellular_respiration.json
```

The viewer shows a chapter grid on load; clicking a chapter transitions into its slide reel. Use the **Back** button or keyboard arrows (← →) to navigate.

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

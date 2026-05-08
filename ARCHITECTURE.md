# SlideStudio — Agentic Pipeline Architecture

## Overview

SlideStudio is a local agentic workflow that transforms academic PDFs — textbooks,
research papers, lecture notes — into structured teaching slides. There is no hosted
backend. Users clone the repository, supply their own AI provider credentials, and
run the pipeline on their own machine. The repository provides the workflow; the
user provides the compute.

The core insight is that slide generation is not a single prompt task. It is a
multi-step reasoning process: understand the document, plan a narrative arc, write
clearly, then verify accuracy. Each step is a dedicated agent with a single
responsibility.

---

## Design Principles

**Single responsibility per agent.**
Each agent does one thing and does it well. No agent reads the document and
writes slides in the same step. Separation of concerns produces better output and
makes individual agents easy to test, improve, or replace.

**Provider agnosticism.**
The pipeline makes no assumption about which AI service is running. Anthropic,
OpenAI, Groq, Ollama, or any OpenAI-compatible endpoint are all valid. Provider
configuration lives in one place and nowhere else.

**Validated outputs, not trusted outputs.**
LLMs do not reliably adhere to output formats. Every agent output is validated
against a Pydantic schema before the next agent runs. Malformed responses are
caught immediately and retried automatically — they never silently propagate.

**Transparent communication.**
Agents pass Python dicts directly in memory. No intermediate file I/O occurs
between agent steps. An optional debug mode writes each agent's output to disk
for inspection. The only required file write is the final viewer output.

**Graceful degradation.**
Every feedback loop has a hard exit condition. The Critic/Refiner cycle runs a
maximum of three iterations. If the pipeline cannot produce a clean output within
that budget, it surfaces the best available result with a warning rather than
looping indefinitely or crashing silently.

**Local first.**
PDFs never leave the user's machine except for the text sent to their chosen AI
provider. No accounts, no telemetry, no hosted infrastructure.

---

## Repository Structure

```
slidestudio/
│
├── pipeline.py               # Orchestrator — runs agents in sequence
├── cli.py                    # CLI entry point — run, estimate, serve, library-refresh
├── server.py                 # FastAPI server — upload, job status, library, settings API
├── config.py                 # Provider, model, and pipeline settings
├── settings.json             # Optional user overrides (written by web UI, gitignored)
│
├── agents/
│   ├── base.py               # Shared Agent base class
│   ├── analyst.py            # Agent 1 — document understanding (two-pass + Map-Reduce)
│   ├── planner.py            # Agent 2 — slide arc design
│   ├── writer.py             # Agent 3 — slide drafting (batched)
│   ├── critic.py             # Agent 4 — accuracy and clarity review
│   └── refiner.py            # Agent 5 — final revision
│
├── extractors/
│   └── pdf.py                # PDF extraction, dynamic header detection, overlap chunking,
│                             #   figure/table placeholders, Surya layout analysis (PyMuPDF)
│
├── providers/
│   ├── base.py               # Shared LLM interface — Semaphore + circuit breaker + tenacity
│   ├── config.py             # ProviderConfig dataclass (injected into provider instances)
│   ├── errors.py             # CircuitOpenError, FatalAPIError
│   ├── anthropic.py          # Anthropic SDK wrapper
│   ├── openai.py             # OpenAI SDK wrapper
│   ├── google.py             # Google Generative AI wrapper
│   └── ollama.py             # Ollama local wrapper
│
├── utils/
│   ├── rate_limiter.py       # asyncio.Semaphore — provider-aware concurrency cap
│   ├── cost_estimator.py     # Token counting and cost table for --estimate flag
│   ├── slugify.py            # Filesystem-safe string sanitiser for filenames and folders
│   ├── checkpoint.py         # Resumable runs — save/restore pipeline stage state
│   └── library.py            # Rebuild and upsert entries in outputs/library.json
│
├── schemas/
│   ├── global_skeleton.py    # Pydantic model — Pass 1 skeleton (headers/TOC)
│   ├── chapter_map.py        # Pydantic model — intermediate merge unit (Map-Reduce)
│   ├── document_map.py       # Pydantic model — Analyst output
│   ├── slide_plan.py         # Pydantic model — Planner output (with chunk_indices)
│   ├── slides_draft.py       # Pydantic model — Writer output
│   ├── critique.py           # Pydantic model — Critic output
│   ├── slides_final.py       # Pydantic model — Refiner output (single deck)
│   └── deck_index.py         # Pydantic model — multi-deck index (viewer TOC)
│
├── outputs/                  # Generated slide sets land here
│   ├── library.json          # Auto-maintained index of all decks
│   ├── archive/              # Archived decks (hidden from main library)
│   ├── paper.json            # Single-deck output (short documents)
│   └── biology-textbook/     # Multi-deck output (long documents)
│       ├── index.json
│       ├── 01_introduction.json
│       └── ...
│
├── tests/                    # pytest suite (~530 tests)
│   ├── test_extractor.py
│   ├── test_analyst.py  test_planner.py  test_writer.py
│   ├── test_critic.py   test_refiner.py  test_pipeline.py
│   ├── test_router.py        # single-deck vs multi-deck routing
│   ├── test_rate_limiter.py  # semaphore cap and backoff behaviour
│   ├── test_cost_estimator.py
│   ├── test_slugify.py  test_filesystem_safety.py
│   ├── test_retry_loop.py  test_resiliency.py  test_stress.py
│   ├── test_checkpoint.py  test_library_manifest.py
│   ├── test_multi_deck.py  test_schemas.py
│   ├── test_cli.py  test_layout.py
│   ├── test_archive_endpoints.py  test_settings_endpoints.py
│   ├── test_config_settings.py
│   ├── test_base_provider.py  test_anthropic_provider.py  test_google_provider.py
│   └── ...
│
├── prompts/                  # Prompt templates (separate from code)
│   ├── analyst_skeleton.txt  analyst_chunk.txt  analyst_merge.txt
│   ├── planner.txt  writer.txt  critic.txt  refiner.txt
│
├── exporters/
│   ├── base.py
│   ├── pptx.py
│   ├── gui.py
│   └── html/
│       └── index.html        # Single-page viewer app (library, reel, TOC, settings)
│
├── .env.example              # API key template — never committed
├── requirements.txt
├── requirements-dev.txt
├── ARCHITECTURE.md
├── CLAUDE.md
└── README.md
```

**Key structural notes:**

- `server.py` is the FastAPI application. It exposes upload, job-status, library, archive/unarchive/delete, and GET/PUT settings endpoints, and serves `outputs/` and `exporters/html/` as static mounts. `cli.py` calls `server.serve_and_open()` for the `--open` and `serve` subcommands.
- `config.py` loads `settings.json` from the project root if present, replacing `PROVIDER` and `MODELS` with the overrides. Malformed JSON or an unknown provider key causes an immediate `sys.exit` with a clear error message.
- `providers/config.py` holds `ProviderConfig` — all provider tunables (timeouts, retry counts, backoff bounds, circuit-breaker thresholds) are injected rather than read from `config.py` inside the provider.
- `providers/errors.py` defines `CircuitOpenError` and `FatalAPIError`, which the pipeline catches separately from transient errors.
- `utils/checkpoint.py` enables `--resume`: each pipeline stage writes its output to `.checkpoints/` keyed by a hash of (PDF content + model + chunk size); a resumed run skips already-completed stages.
- `utils/library.py` maintains `outputs/library.json` — every `run` upserts its entry; archive/unarchive calls trigger a rebuild.
- `schemas/slide_plan.py` — `PlannedSlide.chunk_indices` is bounded at `max_length=3` to prevent unbounded context injection into the Writer.
- `extractors/pdf.py` uses Surya for layout analysis when available, falling back to PyMuPDF heuristics for header detection.
- `agents/writer.py` generates slides in fixed-size batches to stay within output token limits.

---

## Improvement 1 — Pydantic Schema Validation

### The problem

LLMs do not reliably produce valid JSON on every call. Even the best models will
occasionally hallucinate field names, nest objects incorrectly, or omit required
keys. A malformed output from the Analyst will crash the Planner with an
unhelpful `KeyError`. By the time the error surfaces, the root cause is obscured.

### The solution

Every agent output is defined as a Pydantic model. The `BaseProvider.complete_json()`
method is responsible for parsing, validating, and retrying — not the agents
themselves. Agents only receive clean, validated data.

```python
# schemas/document_map.py

from pydantic import BaseModel, Field
from typing import Literal

class Section(BaseModel):
    heading: str
    importance: Literal["high", "medium", "low"]
    summary: str

class DocumentMap(BaseModel):
    title: str
    document_type: Literal["research_paper", "textbook", "lecture_notes", "other"]
    technical_level: Literal["beginner", "intermediate", "advanced"]
    core_thesis: str
    key_concepts: list[str] = Field(min_length=1)
    sections: list[Section] = Field(min_length=1)
```

```python
# providers/base.py

from pydantic import BaseModel, ValidationError
import json

MAX_FORMAT_RETRIES = 3

class BaseProvider:
    def complete_json(self, prompt: str, schema: type[BaseModel], system: str = "") -> BaseModel:
        """
        Calls the LLM, parses the response as JSON, and validates it against
        the provided Pydantic schema. On failure, sends the validation error
        back to the model and asks it to correct its output. Retries up to
        MAX_FORMAT_RETRIES times before raising.
        """
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, MAX_FORMAT_RETRIES + 1):
            raw = self._call(messages, system)

            try:
                data = json.loads(self._extract_json(raw))
                return schema.model_validate(data)

            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == MAX_FORMAT_RETRIES:
                    raise RuntimeError(
                        f"Model failed to produce valid output after "
                        f"{MAX_FORMAT_RETRIES} attempts.\n"
                        f"Last error: {e}\n"
                        f"Last response: {raw[:300]}"
                    )

                # Tell the model what it got wrong and ask it to fix it
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was invalid. Error: {e}\n\n"
                        f"Please return ONLY a valid JSON object that matches "
                        f"the required schema. No explanation, no markdown."
                    )
                })

    def _extract_json(self, text: str) -> str:
        """Strips markdown fences and extracts the JSON object."""
        clean = text.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", clean, 0)
        return clean[start:end]

    def _call(self, messages: list, system: str) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
```

Each agent calls `provider.complete_json(prompt, SchemaClass)` and receives a
validated Pydantic model instance. Field access is then type-safe and explicit
rather than fragile dict lookups.

---

## Improvement 2 — Robust Chunking and Global Context

### The problem

Fixed-size chunking introduces two distinct failure modes that compound each other.

**The boundary problem.** Even with paragraph-snap boundaries, a chunk that ends
with "This led to the development of a new architecture" and a next chunk that begins
with "The architecture, which we call the Transformer..." leaves the LLM with
unresolvable pronouns. Hard edges fracture continuous reasoning.

**The global context problem.** Even if boundaries are clean, Chunk 10 has no
knowledge of the core thesis established in Chunk 1. The Analyst risks
misidentifying early concepts because it cannot see where the document is heading.

### The solution: three strategies in combination

The three strategies below each solve a different layer of the problem. Together
they produce an extraction pipeline that handles everything from a 5-page paper
to a 500-page textbook without quality degradation.

---

#### Strategy 1 — Chunk overlap (the sliding window)

Instead of hard chunk edges, each chunk includes the final `OVERLAP_SIZE`
characters of the previous chunk. This gives the LLM the immediately preceding
context — resolving pronouns and continuing arguments — without significantly
increasing token usage.

```python
# extractors/pdf.py

import pymupdf

CHUNK_SIZE    = 8_000   # target chunk size in characters (~2k tokens)
OVERLAP_SIZE  = 1_500   # trailing overlap from previous chunk (~300 words)

class PDFExtractor:
    def extract(self, file_path: str) -> dict:
        """
        Returns a dict with two keys:
          "headers" — list of section headings extracted from font metadata
          "chunks"  — list of overlapping text chunks
        Both are consumed by the Analyst's two-pass logic.
        """
        doc = pymupdf.open(file_path)
        full_text = "\n".join(page.get_text() for page in doc)
        headers   = self._extract_headers(doc)
        chunks    = self._chunk(full_text)
        return {"headers": headers, "chunks": chunks}

    def _extract_headers(self, doc) -> list[str]:
        """
        Scans each page for text whose font size is significantly larger
        than the body font size. Returns these as the document's section
        headings — used by Pass 1 of the Analyst to build the GlobalSkeleton.
        """
        headers = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["size"] >= 14:   # tune threshold per document type
                            text = span["text"].strip()
                            if text:
                                headers.append(text)
        return headers

    def _chunk(self, text: str) -> list[str]:
        """
        Splits text into overlapping chunks of at most CHUNK_SIZE characters.
        Each boundary snaps to the nearest paragraph break. The next chunk
        begins OVERLAP_SIZE characters before the previous chunk's end,
        ensuring continuous thoughts are never severed cold.
        """
        if len(text) <= CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            if end < len(text):
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1:
                    end = paragraph_break

            chunks.append(text[start:end].strip())

            # Move start back by OVERLAP_SIZE to create the sliding window.
            # The max() guard prevents an infinite loop if no paragraph break
            # was found and the window cannot advance.
            start = max(end - OVERLAP_SIZE, start + 1)

        return [c for c in chunks if c]
```

---

#### Strategy 2 — Two-pass skeleton architecture

Before the Analyst reads any raw text, a fast and cheap Pass 1 scans the
document's headers — extracted by `_extract_headers()` above — and asks the LLM
to generate a `GlobalSkeleton`: the document's overarching structure in a few
hundred tokens. This skeleton is then injected into the system prompt of every
subsequent chunk analysis call, giving every chunk access to the full document
map without re-reading any raw text.

**Why this enables parallelisation.** The rolling-state approach (passing Chunk 1's
output into Chunk 2's prompt) forces strict sequential processing. The skeleton
approach decouples the chunks — each one already has global context from the
skeleton, so all chunks can be analysed concurrently in Pass 2, then merged
in a single final call.

```
Pass 1 — Global Skeleton (1 API call, headers only)
─────────────────────────────────────────────────────
  Input:  list of section headings extracted by PyMuPDF
  Prompt: "Given these section headers, produce a GlobalSkeleton:
           document type, core thesis, and the section outline."
  Output: GlobalSkeleton (validated by Pydantic)


Pass 2 — Chunk Analysis (1 API call per chunk, parallelisable)
─────────────────────────────────────────────────────
  System prompt for EVERY chunk:
    "Here is the global outline of the entire document:
     {global_skeleton_json}
     You are currently reading Chunk {i}, which falls under
     '{section_heading}'. Extract key concepts from this chunk,
     aligning your extraction with the global context above."

  Each call produces a partial DocumentMap.
  All calls run concurrently via asyncio.gather().


Merge call (1 API call)
─────────────────────────────────────────────────────
  Input:  GlobalSkeleton + all partial DocumentMaps
  Output: unified DocumentMap
```

```python
# agents/analyst.py (illustrative — full implementation in code)

import asyncio

class AnalystAgent(BaseAgent):
    name = "analyst"
    output_schema = DocumentMap

    async def run(self, extraction: dict) -> DocumentMap:
        headers = extraction["headers"]
        chunks  = extraction["chunks"]

        # Pass 1 — cheap skeleton from headers alone
        skeleton = self._build_skeleton(headers)

        # Pass 2 — all chunks analysed concurrently with skeleton context
        tasks = [
            self._analyse_chunk(chunk, skeleton, index=i)
            for i, chunk in enumerate(chunks)
        ]
        partial_maps = await asyncio.gather(*tasks)

        # Merge all partial maps into one unified DocumentMap
        if len(partial_maps) == 1:
            return partial_maps[0]
        return self._merge(skeleton, partial_maps)

    def _build_skeleton(self, headers: list[str]) -> GlobalSkeleton:
        prompt = self._load_prompt("analyst_skeleton").format(
            headers="\n".join(headers)
        )
        return self.provider.complete_json(prompt, GlobalSkeleton)

    async def _analyse_chunk(
        self, chunk: str, skeleton: GlobalSkeleton, index: int
    ) -> DocumentMap:
        prompt = self._load_prompt("analyst_chunk").format(
            global_skeleton=skeleton.model_dump_json(indent=2),
            chunk_index=index + 1,
            chunk_text=chunk
        )
        return self.provider.complete_json(prompt, DocumentMap)
```

---

#### Strategy 3 — Semantic chunk boundaries via header detection

The header list produced by `_extract_headers()` does double duty: it feeds the
skeleton pass (Strategy 2) and guides the chunk boundary logic. When a chunk
boundary coincides with a detected section heading, the chunker prefers that
boundary over a plain paragraph break. This means chunks align with semantic
units — chapters and sections — rather than arbitrary character counts.

For well-structured academic documents (which is the primary use case), this
produces chunks that correspond exactly to the author's intended thought units.
For poorly structured PDFs where no headings are detected, the paragraph-snap
fallback ensures the overlap chunker still produces clean boundaries.

---

### Combined flow

```
PDF File
    │
    ▼
PDFExtractor.extract()
    ├── _extract_headers()  →  headers: list[str]
    └── _chunk()            →  chunks: list[str]  (overlapping, boundary-aware)
    │
    │  { "headers": [...], "chunks": [...] }
    ▼
Analyst — Pass 1
    │  Headers only → 1 cheap API call
    │  Output → GlobalSkeleton (validated by Pydantic)
    │
    ▼
Analyst — Pass 2 (concurrent)
    │  GlobalSkeleton injected into every chunk prompt
    │  All chunks processed in parallel via asyncio.gather()
    │  Output → [partial_map_1, partial_map_2, ...N]
    │
    ▼
Analyst — Merge
    │  GlobalSkeleton + all partial maps → 1 API call
    │  Output → DocumentMap (validated by Pydantic)
    │
    ▼
Planner, Writer, Critic/Refiner loop  (unchanged)
```

The Planner, Writer, Critic, and Refiner receive the same `DocumentMap`
interface as before. None of them are aware of chunking, skeletons, or
parallelisation. The complexity is contained entirely within the Analyst.

---

## Improvement 3 — FastAPI Server and Web Studio

### The problem

Opening `viewer/index.html` directly in a browser uses the `file://` protocol.
Modern browsers enforce strict CORS policies on `file://` — a JavaScript `fetch()`
call to load `slides_final.json` from the same directory will be blocked entirely.
The viewer will silently fail to load any slides.

### The solution

`server.py` is a FastAPI application that serves the viewer over HTTP and exposes
a full REST API. The `--open` flag and `serve` subcommand in the CLI call
`server.serve_and_open()`, which starts uvicorn on `localhost:{PIPELINE["port"]}`
and opens the browser.

**API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Accept a PDF, enqueue a background pipeline job |
| `GET` | `/status/{job_id}` | Poll job progress and retrieve the output URL |
| `GET` | `/library` | Return `outputs/library.json` |
| `POST` | `/archive/{slug}` | Move a deck to `outputs/archive/` |
| `POST` | `/unarchive/{slug}` | Restore a deck from the archive |
| `DELETE` | `/archive/{slug}` | Permanently delete an archived deck |
| `GET` | `/settings` | Return active `PROVIDER` and `MODELS` from `config` |
| `PUT` | `/settings` | Write `settings.json` and hot-reload `config` |

**Static mounts:**

- `/outputs` → `outputs/` directory (slide JSON files)
- `/exporters/html` → `exporters/html/` (viewer SPA, with HTML fallback)

The server never imports or modifies `config.py` directly — the PUT `/settings`
endpoint writes `settings.json` and calls `importlib.reload(config)`, so the next
pipeline job picks up the new provider without a restart. The port is read from
`config.PIPELINE["port"]` and passed in by the caller; `server.py` itself has no
hard-coded configuration.

---

## Improvement 4 — Critic/Refiner Kill Switch

### The problem

A linear Critic → Refiner flow is fragile. If the Refiner's fix introduces a new
error — which is plausible, since rewriting one part of a slide can affect another
— a naive cyclic routing sends the output back to the Critic, which flags the new
error, which triggers another Refiner call, and so on indefinitely. Each iteration
costs time and API tokens. Without a hard exit, the pipeline can loop until it
hits a rate limit or the user kills it.

### The solution

The Critic/Refiner cycle is wrapped in a loop with a strict `MAX_REVIEW_CYCLES`
counter. After each Refiner pass, the Critic reviews again. If all slides pass,
the loop exits cleanly. If the maximum is reached, the pipeline exits with the
best available output and logs a warning listing which slides still have issues.

```
                    ┌─────────────────────────┐
                    │  slides_draft           │
                    │  (from Writer)          │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
              ┌────▶│  Agent 4 — Critic       │
              │     │  Reviews all slides     │
              │     └────────────┬────────────┘
              │                  │
              │         ┌────────▼────────┐
              │         │ All slides pass?│
              │         └────────┬────────┘
              │           yes    │    no
              │     ┌────────────┘    └──────────────┐
              │     │                                 │
              │     ▼                                 ▼
              │  EXIT LOOP                ┌───────────────────────┐
              │  → slides_final           │  cycle += 1           │
              │                           │  cycle > MAX? → EXIT  │
              │                           │  with warning         │
              │                           └───────────┬───────────┘
              │                                       │
              │                           ┌───────────▼───────────┐
              └───────────────────────────│  Agent 5 — Refiner    │
                                          │  Fixes flagged slides  │
                                          └───────────────────────┘
```

```python
# pipeline.py

MAX_REVIEW_CYCLES = 3

def run_review_loop(draft, doc_map, critic, refiner):
    """
    Runs the Critic/Refiner cycle with a hard exit at MAX_REVIEW_CYCLES.
    Returns the best available slides and a list of any unresolved issues.
    """
    current = draft
    unresolved = []

    for cycle in range(1, MAX_REVIEW_CYCLES + 1):
        critique = critic.run(doc_map=doc_map, slides=current)

        failed = [s for s in critique.slides if not s.passed]

        if not failed:
            return current, []   # clean exit — all slides passed

        if cycle == MAX_REVIEW_CYCLES:
            # Hard exit — return best effort with warning
            unresolved = [
                f"Slide {s.index}: {s.issues[0].detail}"
                for s in failed
            ]
            break

        # Apply fixes and loop back to Critic
        current = refiner.run(
            doc_map=doc_map,
            slides=current,
            critique=critique
        )

    return current, unresolved
```

The CLI surfaces unresolved issues clearly so the user knows the output may be
imperfect, rather than presenting a silently degraded result as final:

```
  Agent 4  Critic .............. ⚠  2 issues unresolved after 3 cycles
           Slide 3: Positional encoding description remains ambiguous
           Slide 7: Attention weight formula not fully explained

  Output saved with warnings. Review flagged slides before use.
```

---

## Agent Pipeline (Updated)

```
PDF File
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PDFExtractor (extractors/pdf.py)                       │
│  Reads PDF via PyMuPDF.                                 │
│  _extract_headers() → list of section headings          │
│  _chunk() → overlapping chunks, boundary-aware          │
│  _inject_placeholders() → replaces image/table blocks   │
│    with [FIGURE EXCLUDED: caption] and                  │
│    [TABLE EXCLUDED: caption] markers in chunk text      │
│  Returns: { "headers": [...], "chunks": [...] }         │
└─────────────────────────────────────────────────────────┘
    │  { headers, chunks }
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 1 — Analyst (two-pass + hierarchical merge)      │
│  Pass 1: headers → 1 API call → GlobalSkeleton          │
│  Pass 2: all chunks processed concurrently,             │
│          GlobalSkeleton injected into every prompt      │
│  Merge (Map-Reduce):                                    │
│    Step A — chunks within each chapter → ChapterMap     │
│    Step B — all ChapterMaps → final DocumentMap         │
│  All outputs validated via Pydantic.                    │
│  Output → DocumentMap + raw chunks (kept in memory)     │
└─────────────────────────────────────────────────────────┘
    │  DocumentMap + chunks[]
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 2 — Planner                                      │
│  Designs slide arc from DocumentMap.                    │
│  Each PlannedSlide carries chunk_indices — the specific │
│  chunk IDs whose raw text the Writer must receive.      │
│  Output validated against SlidePlan (Pydantic).         │
│  Output → SlidePlan                                     │
└─────────────────────────────────────────────────────────┘
    │  SlidePlan + DocumentMap + chunks[]
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 3 — Writer                                       │
│  Receives SlidePlan, DocumentMap, and only the specific │
│  raw text chunks referenced by each slide's             │
│  chunk_indices. Generates slides in fixed-size batches  │
│  (default: 5 slides per call) to stay within output    │
│  token limits. Batches are concatenated in Python.      │
│  Output validated against SlidesDraft (Pydantic).       │
│  Output → SlidesDraft                                   │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│  Critic / Refiner review loop                           │
│  Max 3 cycles. Exits early if all slides pass.          │
│  Exits with warning if max cycles reached.              │
│  All outputs validated via Pydantic at each cycle.      │
│  Output → SlidesFinal                                   │
└─────────────────────────────────────────────────────────┘
    │  SlidesFinal (JSON written to outputs/)
    ▼
┌─────────────────────────────────────────────────────────┐
│  Local FastAPI Server + Viewer                          │
│  server.py serves outputs/ and exporters/html/ on       │
│  localhost. index.html loads JSON via http:// —         │
│  no CORS issues.                                        │
└─────────────────────────────────────────────────────────┘
```

---

## Improvement 5 — Multi-Deck Generation for Large Documents

### The problem

A fixed `max_slides` cap of 16 works well for a 15-page research paper but
ruthlessly compresses a 500-page textbook into a shallow summary. Every chapter
gets one or two slides. The output is technically correct but useless for studying
because the resolution is far too low to convey the actual content.

### The solution: a router in `pipeline.py`

No new agents are needed. The fix is purely orchestration. After the Analyst
produces the `GlobalSkeleton`, the pipeline inspects the number of `level: 1`
headers (chapters). If the count exceeds `MULTI_DECK_THRESHOLD`, it switches to
multi-deck mode — spawning one full Planner → Writer → Critic/Refiner loop per
chapter, each scoped to that chapter's content. If the count is below the
threshold, it proceeds with the existing single-deck flow unchanged.

```
After Analyst completes
         │
         ▼
┌────────────────────────────────────┐
│  Router (pipeline.py)              │
│  chapters = skeleton.level_1_count │
│  chapters > MULTI_DECK_THRESHOLD?  │
└────────────┬───────────────────────┘
             │
      no     │     yes
             │
 ┌───────────▼──────┐    ┌──────────────────────────────────────────┐
 │  Single-Deck     │    │  Multi-Deck                              │
 │  Planner         │    │  For each chapter in skeleton (parallel):│
 │  Writer          │    │    Planner  (scoped to chapter)          │
 │  Critic/Refiner  │    │    Writer                                │
 │                  │    │    Critic / Refiner loop                 │
 │  → paper.json    │    │  → outputs/textbook-title/               │
 └──────────────────┘    │      index.json                          │
                         │      01_chapter_one.json                 │
                         │      02_chapter_two.json  ...            │
                         └──────────────────────────────────────────┘
```

---

#### The router logic

```python
# pipeline.py

MULTI_DECK_THRESHOLD = 3   # documents with more than this many chapters
                            # get the multi-deck treatment

def route(skeleton: GlobalSkeleton, doc_map: DocumentMap, extraction: dict):
    chapters = [s for s in skeleton.sections if s.level == 1]

    if len(chapters) <= MULTI_DECK_THRESHOLD:
        return run_single_deck(doc_map, skeleton)
    else:
        return run_multi_deck(doc_map, skeleton, chapters, extraction)


async def run_multi_deck(doc_map, skeleton, chapters, extraction):
    """
    Spawns one scoped Planner→Writer→Critic/Refiner loop per chapter,
    all running concurrently. Produces N SlidesFinal objects and writes
    a DeckIndex alongside them.
    """
    tasks = [
        run_single_deck(doc_map, skeleton, scope=chapter)
        for chapter in chapters
    ]
    decks = await asyncio.gather(*tasks)

    return build_deck_index(skeleton.title, chapters, decks)
```

---

#### The scoped Planner

The Planner receives an optional `scope` parameter. When present, the prompt
instructs it to design slides only for that chapter, while using the
`GlobalSkeleton` for alignment — so definitions and terminology remain consistent
with the rest of the book even when each chapter is generated independently.

```
# prompts/planner.txt (scope block, injected when scope is set)

[System]
You are designing a slide arc based on the provided Document Map.

{% if scope %}
Generate slides ONLY for: "{{ scope.heading }}".
Do not write slides for other chapters.
Use the Global Skeleton to ensure your definitions and terminology
align with the rest of the document.
{% endif %}
```

The Planner's `run()` signature gains one optional argument:

```python
def run(self, doc_map: DocumentMap, skeleton: GlobalSkeleton,
        scope: SectionEntry | None = None) -> SlidePlan:
    prompt = self._build_prompt(doc_map, skeleton, scope)
    return self._call(prompt, schema=SlidePlan)
```

Writer, Critic, and Refiner are unaffected — they receive a `SlidePlan` as
before and have no awareness of whether they are in single-deck or multi-deck
mode.

---

#### DeckIndex schema

```python
# schemas/deck_index.py

from pydantic import BaseModel

class DeckEntry(BaseModel):
    chapter_title: str
    file:          str    # relative filename, e.g. "01_introduction.json"

class DeckIndex(BaseModel):
    title:         str
    type:          str = "multi_deck"
    generated_at:  str
    provider:      str
    model:         str
    decks:         list[DeckEntry]
```

Example `index.json` written to disk:

```json
{
  "title": "Biology 101 Textbook",
  "type": "multi_deck",
  "generated_at": "2025-04-18T14:32:00Z",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "decks": [
    { "chapter_title": "Chapter 1: Introduction to Biology",
      "file": "01_introduction.json" },
    { "chapter_title": "Chapter 2: Cell Structure",
      "file": "02_cell_structure.json" },
    { "chapter_title": "Chapter 3: Cellular Respiration",
      "file": "03_respiration.json" }
  ]
}
```

---

#### Viewer impact

The viewer loads whatever JSON the server points it at. On load, it checks the
`type` field:

- `type` is absent or `"single_deck"` → standard slide reel, as before.
- `type` is `"multi_deck"` → render a Table of Contents page listing each
  chapter. Clicking a chapter fetches that chapter's `SlidesFinal` JSON and
  transitions into the standard slide reel. The back button returns to the TOC.

This is a conditional on the existing load function — no new rendering engine,
no framework changes. The slide reel itself is completely reused.

```javascript
// viewer/index.html (simplified load logic)

async function load(url) {
    const data = await fetch(url).then(r => r.json());

    if (data.type === "multi_deck") {
        renderTOC(data);       // chapter list → each item calls load(deck.file)
    } else {
        renderSlides(data);    // existing slide reel — unchanged
    }
}
```

---

#### Updated pipeline diagram (full, with router)

```
PDF File
    │
    ▼
PDFExtractor.extract()
    ├── _extract_headers()  →  headers: list[str]
    └── _chunk()            →  chunks: list[str]  (overlapping, boundary-aware)
    │
    │  { "headers": [...], "chunks": [...] }
    ▼
Analyst — Pass 1  →  GlobalSkeleton
Analyst — Pass 2  →  partial maps (concurrent, trimmed skeleton per chunk)
Analyst — Merge (Map-Reduce)
    Step A: group partial maps by chapter → ChapterMaps (one merge call per chapter)
    Step B: merge all ChapterMaps → DocumentMap  (one final merge call)
    │
    │  GlobalSkeleton + DocumentMap + chunks[] (kept in memory)
    ▼
┌──────────────────────────────────────────┐
│  Router (pipeline.py)                    │
│  level-1 chapter count > threshold?      │
└──────────┬───────────────────────────────┘
           │
     no    │    yes
           │
┌──────────▼────────┐   ┌─────────────────────────────────────────┐
│  Single-Deck Flow │   │  Multi-Deck Flow (per chapter, parallel)│
│  Planner          │   │  Planner  (scope = chapter)             │
│  Writer           │   │  Writer                                 │
│  Critic/Refiner   │   │  Critic / Refiner loop                  │
│                   │   │  → SlidesFinal per chapter              │
│  → paper.json     │   │  → DeckIndex (index.json)               │
└───────────────────┘   └─────────────────────────────────────────┘
    │                        │
    └──────────┬─────────────┘
               ▼
    FastAPI Server + Viewer (server.py)
    Loads single JSON or index.json → TOC → chapter JSON
```

---

## Improvement 6 — Production Hardening

### Fix 1 — API rate limiting: Semaphore + tenacity

**Why both.** `asyncio.Semaphore` is proactive — it prevents more than N calls
from being in-flight at once. `tenacity` is reactive — it recovers gracefully
when a 429 lands despite the semaphore, which can still happen when the API's
per-minute *token* budget is exhausted rather than the per-request limit. Neither
alone is sufficient; together they form a complete defence.

```python
# utils/rate_limiter.py

import asyncio

# A single semaphore shared across all providers and agents.
# All concurrent API calls draw from this one budget.
# Value is read from config so users can tune it per-provider.
_semaphore: asyncio.Semaphore | None = None

def get_semaphore(max_concurrent: int = 5) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max_concurrent)
    return _semaphore
```

```python
# providers/base.py

import asyncio
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel, ValidationError
from utils.rate_limiter import get_semaphore

class RateLimitError(Exception):
    """Raised when the API returns HTTP 429."""

class BaseProvider:
    async def complete_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str = ""
    ) -> BaseModel:
        """
        Every API call goes through here.
        1. Acquires the shared semaphore before the request.
        2. tenacity retries on RateLimitError with exponential backoff.
        3. On JSON/validation failure, feeds the error back to the model
           and retries up to MAX_FORMAT_RETRIES times.
        """
        semaphore = get_semaphore(CONFIG.PIPELINE["max_concurrent"])
        async with semaphore:
            return await self._call_with_backoff(prompt, schema, system)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(6),
    )
    async def _call_with_backoff(self, prompt, schema, system):
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, CONFIG.PIPELINE["max_format_retries"] + 1):
            raw = await self._call(messages, system)

            try:
                data = json.loads(self._extract_json(raw))
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == CONFIG.PIPELINE["max_format_retries"]:
                    raise RuntimeError(f"Format retries exhausted: {e}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": f"Invalid output. Error: {e}. Return valid JSON only."
                })

    async def _call(self, messages: list, system: str) -> str:
        """Provider-specific HTTP call. Must raise RateLimitError on 429."""
        raise NotImplementedError
```

The `tenacity` decorator handles the backoff schedule: 4s → 8s → 16s → 32s →
60s, up to 6 attempts. If all 6 fail, the exception propagates to `pipeline.py`,
which logs the failure and — in multi-deck mode — marks that chapter as failed
rather than crashing the entire run.

---

### Fix 2 — Dynamic header detection

The hardcoded `>= 14` threshold breaks silently on academic papers that use 10pt
body text and 12pt bold headers. The fix calculates the statistical mode of all
font sizes on each page, which reliably identifies body text, then defines a
header as any span whose size is strictly greater than the mode, or equal to the
mode but bold.

```python
# extractors/pdf.py  (_extract_headers, revised)

from statistics import mode, StatisticsError

def _extract_headers(self, doc) -> list[str]:
    """
    Dynamically detects headers by comparing each span's font size to
    the statistical mode of all font sizes on the page (the body text).
    A span is a header if its size exceeds the mode, OR if its size
    equals the mode but the font is bold — catching documents that use
    weight rather than size to mark headings.
    """
    headers = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        all_sizes = [
            span["size"]
            for block in blocks
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]

        if not all_sizes:
            continue

        try:
            body_size = mode(all_sizes)
        except StatisticsError:
            body_size = min(all_sizes)   # fallback: smallest size = body

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text    = span.get("text", "").strip()
                    size    = span["size"]
                    is_bold = "bold" in span.get("font", "").lower()
                    if not text:
                        continue
                    if size > body_size or (size == body_size and is_bold):
                        headers.append(text)

    return headers
```

---

### Fix 3 — Skeleton trimming

For a 600-page textbook with 142 headers, the serialised `GlobalSkeleton` can
exceed 5,000 tokens. Injecting this into every chunk prompt is redundant —
Chapter 10 does not need the granular subsections of Chapter 2. The trim method
keeps all `level: 1` chapter headings globally but retains `level: 2` and
`level: 3` detail only for the chapter surrounding the current chunk.

```python
# agents/analyst.py  (_trim_skeleton)

def _trim_skeleton(
    self,
    skeleton: GlobalSkeleton,
    current_chunk_index: int
) -> GlobalSkeleton:
    """
    Returns a leaner skeleton: all level-1 headings are kept globally,
    but level-2/3 detail is retained only for the current chapter.
    """
    current_chapter_pos = max(
        (s.position for s in skeleton.sections
         if s.level == 1 and s.position <= current_chunk_index),
        default=0
    )

    trimmed = [
        s for s in skeleton.sections
        if s.level == 1 or s.position >= current_chapter_pos
    ]

    return skeleton.model_copy(update={"sections": trimmed})
```

The trimmed skeleton is passed into `_analyse_chunk()` rather than the full one,
keeping prompt size bounded regardless of document length.

---

### Fix 4 — Dry-run cost estimation (`--estimate`)

Before any API calls are made, `--estimate` runs the extractor and skeleton
pass locally, then calculates the anticipated token usage and cost.

```python
# utils/cost_estimator.py

# Cost per million tokens (input / output), USD.
# Single location to update when provider pricing changes.
PRICING = {
    "anthropic": {
        "claude-sonnet-4-20250514": {"input": 3.00,  "output": 15.00},
        "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},
    },
    "openai": {
        "gpt-4o":      {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    },
    "groq": {
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    },
    "ollama": {
        "*": {"input": 0.00, "output": 0.00},   # local — no cost
    },
}

CHARS_PER_TOKEN = 4   # conservative approximation

def estimate(extraction: dict, skeleton: GlobalSkeleton,
             provider: str, model: str) -> dict:
    chunks   = extraction["chunks"]
    chapters = [s for s in skeleton.sections if s.level == 1]
    is_multi = len(chapters) > CONFIG.PIPELINE["multi_deck_threshold"]
    deck_count = len(chapters) if is_multi else 1

    analyst_calls = 1 + len(chunks) + 1     # skeleton + chunks + merge
    deck_calls    = deck_count * 4           # planner + writer + critic + refiner (avg)
    total_calls   = analyst_calls + deck_calls

    avg_chunk_tokens = sum(len(c) for c in chunks) // len(chunks) // CHARS_PER_TOKEN
    total_input_tok  = total_calls * avg_chunk_tokens
    total_output_tok = total_calls * 500     # average output per call

    prices = (PRICING.get(provider, {}).get(model)
              or PRICING.get(provider, {}).get("*")
              or {"input": 0, "output": 0})

    cost = (total_input_tok  / 1_000_000 * prices["input"]
          + total_output_tok / 1_000_000 * prices["output"])

    return {
        "mode":          "multi-deck" if is_multi else "single-deck",
        "decks":         deck_count,
        "api_calls":     total_calls,
        "input_tokens":  total_input_tok,
        "output_tokens": total_output_tok,
        "est_cost_usd":  round(cost, 4),
        "est_time_sec":  total_calls * 3,    # 3s per call, conservative
    }
```

CLI output for `--estimate`:

```
python cli.py biology-textbook.pdf --estimate

SlideStudio — Cost Estimate  (no API calls made)
──────────────────────────────────────────────────────
  Document     biology-101-textbook.pdf  48.3 MB · 612 pages
  Provider     Anthropic / claude-sonnet-4-20250514
  Mode         Multi-deck  (18 chapters)

  API calls    ~114   (11 analyst + 72 generation + 31 review)
  Input        ~2,280,000 tokens
  Output       ~57,000 tokens
  Est. cost    ~$7.71 USD
  Est. time    ~6 min

  Estimates are conservative — real costs are typically 20–40% lower.
  Run without --estimate to generate slides.
──────────────────────────────────────────────────────
```

---

## Improvement 7 — Logic Gap Hardening

### Fix 1 — Writer factual grounding via chunk_indices

**The problem.** The Writer previously received only the `SlidePlan` and the
`DocumentMap`. The `DocumentMap` contains 3-sentence summaries per section — not
the author's actual prose. When the Planner schedules a slide on "Equation 4:
Attention Weights," the Writer has no source text to draw from and will
hallucinate the body content.

**The fix.** `PlannedSlide` now carries a `chunk_indices` field — a list of
chunk IDs whose raw text is required to write that slide accurately. The Planner
is responsible for identifying which chunks contain the relevant content. The
pipeline then fetches those specific chunks and passes them to the Writer
alongside the plan. The Writer never fabricates content it has not read.

```python
# agents/writer.py  (illustrative — grounding logic)

def run(
    self,
    slide_plan: SlidePlan,
    doc_map:    DocumentMap,
    chunks:     list[str],     # all raw chunks, kept in memory from extraction
) -> SlidesDraft:
    """
    Generates slides in batches of WRITER_BATCH_SIZE.
    Each batch receives only the raw chunks referenced by those slides'
    chunk_indices — not the full chunk list.
    """
    all_slides = []

    for batch in self._batch(slide_plan.slides):
        # Collect only the chunks this batch of slides actually needs
        needed_chunk_ids = {i for slide in batch for i in slide.chunk_indices}
        source_text = "\n\n---\n\n".join(
            f"[Chunk {i}]\n{chunks[i]}"
            for i in sorted(needed_chunk_ids)
            if i < len(chunks)
        )
        draft_batch = self._write_batch(batch, doc_map, source_text)
        all_slides.extend(draft_batch.slides)

    return SlidesDraft(title=slide_plan.title, slides=all_slides)
```

The prompt for each batch explicitly instructs the Writer to use only the
provided source text for body content — never to summarise from memory.

---

### Fix 2 — Hierarchical Map-Reduce merge

**The problem.** Merging 94 partial `DocumentMap` objects in a single prompt
produces 37,600+ tokens of dense, repetitive JSON. At this scale, attention
degrades: the LLM truncates output, drops chapters, or fails to synthesise a
coherent thesis. The problem worsens linearly with document length.

**The fix.** A two-step Map-Reduce. In Step A, partial maps within the same
chapter are merged into a `ChapterMap` — one small merge call per chapter. In
Step B, the resulting `ChapterMap` objects (typically 8–20, each compact) are
merged into the final `DocumentMap`. The maximum input size of any single merge
call is now bounded by the chapter count, not the chunk count.

```
Pass 2 output: 94 partial maps
    │
    ▼  Step A — chapter-level merge (one call per chapter, parallelisable)
    ├── Ch01 chunks  1– 6  →  ChapterMap 01   (1 merge call)
    ├── Ch02 chunks  7–12  →  ChapterMap 02   (1 merge call)
    ├── ...
    └── Ch18 chunks 89–94  →  ChapterMap 18   (1 merge call)
    │
    ▼  Step B — document-level merge (1 call, 18 ChapterMaps as input)
    DocumentMap  ← well within attention budget
```

```python
# agents/analyst.py  (merge logic)

async def _merge(
    self,
    skeleton:     GlobalSkeleton,
    partial_maps: list[DocumentMap],   # one per chunk
    sections:     list[SectionEntry],  # chapter boundaries from skeleton
) -> DocumentMap:

    # Step A — group partial maps by chapter and merge concurrently
    chapter_groups = self._group_by_chapter(partial_maps, sections)
    chapter_tasks  = [
        self._merge_chapter(skeleton, group, chapter)
        for chapter, group in chapter_groups.items()
    ]
    chapter_maps: list[ChapterMap] = await asyncio.gather(*chapter_tasks)

    # Step B — merge all ChapterMaps into the final DocumentMap
    return await self._merge_document(skeleton, chapter_maps)

def _group_by_chapter(
    self,
    partial_maps: list[DocumentMap],
    sections:     list[SectionEntry],
) -> dict[str, list[DocumentMap]]:
    """
    Assigns each partial map to its chapter using the chunk position
    metadata recorded in the GlobalSkeleton's SectionEntry objects.
    """
    groups: dict[str, list[DocumentMap]] = {}
    chapter_boundaries = [s for s in sections if s.level == 1]

    for i, pmap in enumerate(partial_maps):
        chapter = next(
            (s.heading for s in reversed(chapter_boundaries)
             if s.position <= i),
            chapter_boundaries[0].heading if chapter_boundaries else "main"
        )
        groups.setdefault(chapter, []).append(pmap)

    return groups
```

---

### Fix 3 — Batched slide generation

**The problem.** Requesting all 16 slides in a single Writer call can generate
3,500+ tokens of JSON. Models with a 4,096 output token limit will truncate
mid-slide, producing broken JSON that Pydantic catches as a `JSONDecodeError`.
The retry loop then re-runs the identical oversized prompt and hits the same
ceiling — exhausting `max_format_retries` and crashing.

**The fix.** The Writer generates slides in fixed-size batches. Each batch
produces a small, safe number of slides that fits well within the output token
budget. The validated lists are concatenated in Python — no LLM is involved in
the join.

```python
# agents/writer.py  (_batch helper)

WRITER_BATCH_SIZE = 5   # slides per API call — safe for all supported models

def _batch(self, slides: list[PlannedSlide]) -> list[list[PlannedSlide]]:
    """Splits a slide list into fixed-size batches."""
    return [
        slides[i : i + WRITER_BATCH_SIZE]
        for i in range(0, len(slides), WRITER_BATCH_SIZE)
    ]
```

For a 16-slide deck this produces 4 calls of 4, 4, 4, and 4 slides — each
generating roughly 800–1,000 output tokens, well below any provider's ceiling.
The batch size is a named constant so it can be tuned without touching logic.

---

### Fix 4 — Provider-aware concurrency defaults

**The problem.** A semaphore of 5 is correct for cloud APIs (Anthropic, OpenAI,
Groq) which are built for concurrent requests. It is destructive for Ollama,
which runs a single model instance on the user's GPU. Sending 5 simultaneous
generation requests to a local Llama instance will saturate VRAM, trigger OOM
errors, or crash the server — none of which return a clean HTTP 429 that
`tenacity` can catch and retry.

**The fix.** The semaphore default is provider-aware. Ollama always defaults to
`max_concurrent = 1` unless the user explicitly overrides it with
`--max-concurrent`. Cloud providers default to 5.

```python
# utils/rate_limiter.py  (updated)

import asyncio

# Provider-aware defaults.
# Ollama runs locally — serialise all requests to protect GPU memory.
# Cloud providers are designed for concurrency.
PROVIDER_DEFAULTS = {
    "anthropic": 5,
    "openai":    5,
    "groq":      5,
    "ollama":    1,   # sequential — local GPU cannot handle concurrent inference
}

_semaphore: asyncio.Semaphore | None = None

def get_semaphore(provider: str, user_override: int | None = None) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        limit = user_override if user_override is not None \
                else PROVIDER_DEFAULTS.get(provider, 5)
        _semaphore = asyncio.Semaphore(limit)
    return _semaphore

def reset_semaphore() -> None:
    """Call between test runs to avoid state leakage."""
    global _semaphore
    _semaphore = None
```

The `BaseProvider.__init__` passes `self.provider_name` when acquiring the
semaphore, so the default resolves correctly without any caller-side logic.
The Ollama provider also skips the `tenacity` backoff decorator entirely —
replacing it with a simple timeout handler, since local servers crash rather
than returning 429.

---

## Improvement 8 — Real-World Filesystem and Data Integrity

### Fix 1 — Figure and table placeholders

**The problem.** `PDFExtractor` only reads `span["text"]`. Images, charts, and
complex tables are silently dropped. When the extracted text references "Figure 4"
or "as shown in Table 2", the Analyst has no knowledge of what those visuals
contained. It either produces a slide that reads "See Figure 4" — useless
without the figure — or hallucinates the data that should have been there.

**The fix.** `pdf.py` detects image and table blocks via PyMuPDF's block type
enumeration and injects structured text placeholders into the chunk at the
exact position the visual occupied. If a caption is found in an adjacent span,
it is included. This costs nothing extra and prevents the downstream agents
from treating a missing visual as a logical gap in their own reasoning.

```python
# extractors/pdf.py  (_read_pdf, updated)

import pymupdf

BLOCK_TYPE_IMAGE = 1   # PyMuPDF block type constant for images

def _read_pdf(self, file_path: str) -> str:
    doc = pymupdf.open(file_path)
    pages_text = []

    for page in doc:
        blocks    = page.get_text("dict")["blocks"]
        page_text = []

        for i, block in enumerate(blocks):
            if block["type"] == BLOCK_TYPE_IMAGE:
                # Look for a caption in the immediately following text block
                caption = ""
                if i + 1 < len(blocks) and blocks[i + 1]["type"] == 0:
                    next_spans = [
                        span["text"].strip()
                        for line in blocks[i + 1].get("lines", [])
                        for span in line.get("spans", [])
                    ]
                    candidate = " ".join(next_spans)
                    # Accept as caption only if it starts with a figure marker
                    if candidate.lower().startswith(("fig", "figure", "chart")):
                        caption = candidate

                placeholder = (
                    f'[FIGURE EXCLUDED: "{caption}"]'
                    if caption else
                    "[FIGURE EXCLUDED: no caption detected]"
                )
                page_text.append(placeholder)

            elif block["type"] == 0:   # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            page_text.append(text)

        pages_text.append(" ".join(page_text))

    return "\n".join(pages_text)
```

Table detection follows the same pattern. PyMuPDF 1.23+ exposes
`page.find_tables()` which returns bounding boxes and cell content. A table
placeholder injects `[TABLE EXCLUDED: N rows × M cols]` plus the first row as
a header hint, giving the Analyst enough to know a data table existed and what
its columns were.

The Analyst and Planner prompts are updated to instruct the model to treat
placeholders as content signals — never to write a slide that says only
"See Figure 4", but instead to note that a visual was present and describe
what the surrounding text says it demonstrates.

---

### Fix 2 — Bounded chunk_indices on PlannedSlide

**The problem.** Without a cap on `chunk_indices`, the Planner can assign
`[1, 2, 3, 4, 5, 6, 7, 8]` to a summary slide, injecting 16,000+ tokens of
raw text into a single Writer prompt. This hits the input token limit, stalls
generation, or produces an invoice for a single slide that costs more than
the rest of the deck combined.

**The fix.** `PlannedSlide.chunk_indices` is bounded at `max_length=3` in the
Pydantic schema. Three chunks is approximately 6,000 characters — enough to
ground a specific claim or example factually, but not enough to reproduce an
entire chapter. For broad summary slides, the Planner prompt explicitly instructs
the model to rely on the `DocumentMap` summary fields rather than raw chunk text.
This is architecturally correct — the `DocumentMap` exists precisely to provide
that synthesised view.

The Pydantic constraint enforces this at the schema level, meaning a misbehaving
model output is caught and retried before it ever reaches the Writer — not
discovered mid-generation when the token bill is already accruing.

---

### Fix 3 — Filesystem-safe slugify

**The problem.** In multi-deck mode, folder and filenames are generated from
PDF titles and chapter headings extracted by PyMuPDF. Academic headings
routinely contain characters that are strictly forbidden on Windows
(`< > : " / \ | ? *`) and cause path separator collisions on macOS and Linux
(`/`, `:`). A heading like "Chapter 1: The Cell / Is it Alive?" will crash the
pipeline with a fatal `FileNotFoundError` at the very last step — after minutes
of generation and dollars of API spend.

**The fix.** A `slugify()` helper in `utils/slugify.py` sanitises every string
before it touches the disk. It is called once per output path and never inline
in business logic.

```python
# utils/slugify.py

import re
import unicodedata

def slugify(text: str, max_length: int = 80) -> str:
    """
    Converts an arbitrary string into a filesystem-safe slug.

    Steps:
      1. Normalise unicode (é → e, ñ → n, etc.)
      2. Lowercase
      3. Strip all characters that are not alphanumeric, spaces, or hyphens
      4. Replace runs of spaces/hyphens with a single underscore
      5. Strip leading and trailing underscores
      6. Truncate to max_length to avoid PATH_MAX issues on Windows (260 chars)

    Examples:
      "Chapter 1: The Cell / Is it Alive?"  →  "chapter_1_the_cell_is_it_alive"
      "§3.2 ATP Synthesis & the Mitochondria" →  "32_atp_synthesis_the_mitochondria"
      "  leading / trailing  "               →  "leading_trailing"
    """
    # Step 1 — normalise unicode to ASCII equivalents where possible
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Step 2 — lowercase
    text = text.lower()

    # Step 3 — remove forbidden characters (keep alphanumerics, spaces, hyphens)
    text = re.sub(r"[^\w\s-]", "", text)

    # Step 4 — collapse whitespace and hyphens into a single underscore
    text = re.sub(r"[-\s]+", "_", text)

    # Step 5 — strip leading/trailing underscores
    text = text.strip("_")

    # Step 6 — truncate
    return text[:max_length]
```

`slugify()` is called in two places in `pipeline.py`:

```python
# pipeline.py  (output path construction)

from utils.slugify import slugify

def build_output_dir(pdf_title: str) -> Path:
    """Single-deck: outputs/<slug>.json
       Multi-deck:  outputs/<slug>/"""
    return Path("outputs") / slugify(pdf_title)

def build_deck_filename(chapter_index: int, chapter_heading: str) -> str:
    """e.g. 03_cellular_respiration.json"""
    return f"{chapter_index:02d}_{slugify(chapter_heading)}.json"
```

These are the only two locations that construct output paths. No other part of
the codebase produces filenames — making the sanitisation surface minimal and
easy to audit.

---

## Pydantic Schemas

```python
# schemas/global_skeleton.py

from pydantic import BaseModel, Field

class SectionEntry(BaseModel):
    heading: str
    level: int       # 1 = chapter, 2 = section, 3 = subsection
    position: int    # ordinal — which chunk this heading falls in

class GlobalSkeleton(BaseModel):
    title:         str = Field(max_length=120)
    document_type: str
    core_thesis:   str = Field(max_length=400)
    sections:      list[SectionEntry]

    def as_context(self) -> str:
        """Serialises the skeleton for injection into chunk prompts."""
        return self.model_dump_json(indent=2)
```

```python
# schemas/chapter_map.py

from pydantic import BaseModel, Field

class ChapterMap(BaseModel):
    """
    Intermediate merge unit in the hierarchical Map-Reduce.
    Produced by merging all partial DocumentMaps for one chapter.
    Multiple ChapterMaps are then merged into the final DocumentMap.
    """
    chapter_heading: str
    key_concepts:    list[str] = Field(min_length=1)
    summary:         str = Field(max_length=500)
    chunk_range:     tuple[int, int]   # (first_chunk_index, last_chunk_index)
```

```python
# schemas/slide_plan.py

from pydantic import BaseModel, Field
from typing import Literal

VALID_TAGS = Literal[
    "Introduction", "Key Concept", "Definition",
    "Example", "Insight", "Data Point", "Takeaway", "Summary"
]

class PlannedSlide(BaseModel):
    index:          int
    tag:            VALID_TAGS
    source_section: str
    intention:      str       = Field(max_length=200)
    emphasis:       str       = Field(max_length=200)
    chunk_indices:  list[int] = Field(min_length=1, max_length=3)
    # chunk_indices: the specific raw text chunk IDs the Writer must
    # receive to ground this slide factually. Hard cap of 3 chunks
    # (~6,000 characters) prevents unbounded context injection — if
    # a slide needs a broader view, the Planner must rely on the
    # DocumentMap summary rather than injecting the full raw text.

class SlidePlan(BaseModel):
    title:        str = Field(max_length=60)
    total_slides: int = Field(ge=4, le=20)
    slides:       list[PlannedSlide]
```

```python
# schemas/critique.py

from pydantic import BaseModel
from typing import Literal

class Issue(BaseModel):
    type: Literal["inaccuracy", "clarity", "gap", "density", "heading_mismatch"]
    detail: str

class SlideReview(BaseModel):
    index: int
    passed: bool
    issues: list[Issue] = []

class Critique(BaseModel):
    slides: list[SlideReview]

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.slides)

    @property
    def failed_slides(self) -> list[SlideReview]:
        return [s for s in self.slides if not s.passed]
```

---

## Provider Interface (Updated)

```python
# providers/base.py

class BaseProvider:
    def complete_json(self, prompt: str, schema: type[BaseModel], system: str = "") -> BaseModel:
        """
        Calls the LLM, validates output against the Pydantic schema,
        and retries with error feedback on failure.
        Up to MAX_FORMAT_RETRIES attempts before raising.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
```

```python
# config.py

PROVIDER = "google"    # anthropic | openai | google | google-fast | ollama

MODELS = {
    "anthropic":   "claude-sonnet-4-20250514",
    "openai":      "gpt-4o",
    "google":      "gemini-3-flash-preview",
    "google-fast": "gemma-4-31b-it",
    "ollama":      "llama3.1",
}

PIPELINE = {
    "max_slides":                    16,
    "writer_batch_size":             5,
    "chunk_size":                    8_000,
    "overlap_size":                  1_500,
    "multi_deck_chapter_threshold":  3,
    "multi_deck_length_threshold":   100_000,
    "max_concurrent":                None,
    "max_format_retries":            3,
    "max_review_cycles":             3,
    "request_timeout":               600,
    "max_rate_limit_retries":        6,
    "backoff_wait_min":              4,
    "backoff_wait_max":              60,
    "circuit_breaker_threshold":     5,
    "circuit_breaker_cooldown":      60,
    "port":                          7654,
    "debug":                         False,
}

# settings.json (project root, gitignored) is loaded here if present.
# The web UI writes it; config.py applies it so all entry points stay in sync.
# Malformed JSON or an unknown PROVIDER key causes sys.exit with a clear message.
```

---

## Agent Base Class (Updated)

```python
# agents/base.py

from pathlib import Path
from pydantic import BaseModel

class BaseAgent:
    name: str = "base"
    output_schema: type[BaseModel] = None   # set by each subclass

    def __init__(self, provider):
        self.provider = provider
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        return Path(f"prompts/{self.name}.txt").read_text()

    def _call(self, prompt: str, system: str = "") -> BaseModel:
        """
        Calls the provider with schema validation built in.
        Retries with error feedback are handled inside the provider.
        """
        return self.provider.complete_json(
            prompt=prompt,
            schema=self.output_schema,
            system=system
        )

    def run(self, **context) -> BaseModel:
        raise NotImplementedError
```

---

## CLI Design (Updated)

```bash
# Basic usage
python cli.py paper.pdf

# Specify provider
python cli.py paper.pdf --provider openai

# Specify model
python cli.py paper.pdf --provider anthropic --model claude-opus-4-20250514

# Dry-run — estimate tokens, cost, and time before any API calls
python cli.py paper.pdf --estimate

# Skip the critic/refiner loop (faster, lower quality)
python cli.py paper.pdf --fast

# Save intermediate agent outputs for inspection
python cli.py paper.pdf --debug

# Open viewer automatically when done (spins up local server)
python cli.py paper.pdf --open

# Tune concurrency (default: 5, lower for stricter rate limits)
python cli.py paper.pdf --max-concurrent 3

# Run a single agent in isolation (for development)
python cli.py paper.pdf --agent writer --debug
```

### Terminal output — single-deck (short paper)

```
SlideStudio — Agentic Slide Generator
──────────────────────────────────────────────
  Document   attention-is-all-you-need.pdf
             0.4 MB · 15 pages · 2 chunks
  Provider   Anthropic / claude-sonnet-4-20250514
  Mode       Single-deck

  Extracting text .............. ✓  15 pages · 2 chunks · 8 headers
  Agent 1  Analyst  Pass 1 ...... ✓  GlobalSkeleton built
           Analyst  Pass 2 ...... ✓  2 chunks analysed in parallel
           Analyst  Merge ....... ✓  11 concepts · research_paper
  Agent 2  Planner ............. ✓  13 slides planned
  Agent 3  Writer .............. ✓  13 slides drafted
  Review   Cycle 1/3 ........... ✓  All slides passed

──────────────────────────────────────────────
  Output     outputs/attention-is-all-you-need.json
  Serving    http://localhost:7654
  Viewer     Opening in browser...
  Total      28.1s · 5 API calls
──────────────────────────────────────────────
```

### Terminal output — multi-deck (large textbook)

```
SlideStudio — Agentic Slide Generator
──────────────────────────────────────────────
  Document   biology-101-textbook.pdf
             48.3 MB · 612 pages · 94 chunks
  Provider   Anthropic / claude-sonnet-4-20250514
  Mode       Multi-deck  (18 chapters detected)

  Extracting text .............. ✓  612 pages · 94 chunks · 142 headers
  Agent 1  Analyst  Pass 1 ...... ✓  GlobalSkeleton built (18 chapters)
           Analyst  Pass 2 ...... ✓  94 chunks analysed in parallel
           Analyst  Merge ....... ✓  DocumentMap complete
  Router  ...................... ✓  18 chapters → multi-deck mode

  Generating 18 chapter decks in parallel...
  Ch 01  Introduction ........... ✓  14 slides · 2 review cycles
  Ch 02  Cell Structure ......... ✓  16 slides · 1 review cycle
  Ch 03  Cellular Respiration ... ✓  15 slides · 1 review cycle
  ...
  Ch 18  Ecology ................ ✓  13 slides · 2 review cycles

──────────────────────────────────────────────
  Output     outputs/biology-101-textbook/
             index.json + 18 chapter decks
  Serving    http://localhost:7654
  Viewer     Opening in browser... (TOC view)
  Total      4m 12s · 114 API calls
──────────────────────────────────────────────
```

---

## Setup for New Users

```bash
# 1. Clone the repository
git clone https://github.com/yourname/slidestudio
cd slidestudio

# 2. Create a virtual environment (Python 3.12 required)
python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env and add the key for your provider (default: Google):
# GOOGLE_API_KEY=AIza...

# 5. Run on any PDF
python cli.py run path/to/your/paper.pdf --open
# → Slides generated and opened in browser automatically

# Or start the web studio for browser-based uploads:
python cli.py serve
```

---

## requirements.txt

```
anthropic>=0.25.0
fastapi>=0.100.0
google-genai>=1.0.0
pymupdf>=1.24.0
pymupdf4llm>=1.27.0
pydantic>=2.0.0
python-dotenv>=1.0.0
python-multipart>=0.0.7
rich>=13.0.0
surya-ocr>=0.17.0
tenacity>=8.0.0
typer>=0.12.0
uvicorn[standard]>=0.20.0
```

- `anthropic` / `google-genai` — provider SDKs (OpenAI and Ollama use their own SDKs)
- `fastapi` / `uvicorn` / `python-multipart` — web server and file upload
- `pymupdf` / `pymupdf4llm` — PDF extraction with dynamic font-size header detection
- `surya-ocr` — layout analysis for accurate figure and table detection
- `pydantic` — schema validation and automatic retry on malformed LLM output
- `tenacity` — exponential backoff retry on HTTP 429 rate limit responses
- `python-dotenv` — loads `.env` file
- `rich` — formatted terminal output
- `typer` — CLI argument parsing

---

## Incremental Build Plan

**Milestone 1 — Extractor works**
`pdf.py` extracts text, detects headers via dynamic font-size mode, produces
overlapping chunks with paragraph-snap boundaries, and injects `[FIGURE EXCLUDED]`
and `[TABLE EXCLUDED]` placeholders for image and table blocks. Test on three PDF
types: a paper with inline figures (confirm placeholders appear at correct
positions), a textbook with data tables (confirm table placeholder includes
column count), and a poorly-formatted PDF where bold weight is the only heading
signal (confirm non-empty headers list). Zero AI dependency at this stage.

**Milestone 2 — Pydantic schemas, rate limiter, and slugify defined**
All eight schemas written and tested. `PlannedSlide.chunk_indices` validated at
`max_length=3` — confirm a Pydantic `ValidationError` is raised when the model
returns 4 or more indices. `slugify()` tested against a fixture of hostile strings:
Windows-forbidden characters, leading/trailing spaces, unicode ligatures, empty
string, and a 200-character heading (confirm truncation at 80). Semaphore
provider-aware defaults verified. No API calls yet.

**Milestone 3 — Analyst two-pass + hierarchical merge works end to end**
Pass 1 (skeleton), Pass 2 (concurrent chunk analysis with trimmed skeleton), and
the two-step Map-Reduce merge all producing valid Pydantic-validated output.
Confirm `_group_by_chapter()` assigns chunks to the correct chapters. Confirm
Step B merge input stays well below 10,000 tokens even for a 100-chapter document.
Confirm figure placeholder text in chunks flows through to the `DocumentMap`
section summaries without being stripped.

**Milestone 4 — Planner chunk_indices and Writer batching verified**
Planner produces `PlannedSlide` objects with `chunk_indices` of length 1–3 only
— any attempt to exceed 3 is caught by Pydantic before the Writer runs. Writer
receives and uses only the referenced raw chunks. Batches of 5 slides each stay
under 1,200 output tokens. Confirm a 16-slide deck produces exactly 4 Writer
API calls concatenated correctly.

**Milestone 5 — Full single-deck pipeline runs**
All five agents run in sequence. Router identifies single-deck. Review loop runs
with `MAX_REVIEW_CYCLES = 3`. Debug mode writes all intermediates to disk.
Output file written to `outputs/<slugified_title>.json` — confirm no path errors
on both macOS and Windows.

**Milestone 6 — Kill switch and chunking verified**
Force the review loop to hit the cycle limit using a deliberately bad prompt.
Verify overlap chunking on a 200-page textbook. Run `--estimate` and confirm
accuracy within 30% of actual cost from a real run.

**Milestone 7 — Multi-deck routing and filesystem safety verified**
Feed a textbook with more than `MULTI_DECK_THRESHOLD` chapters, including at
least one chapter heading containing a colon, slash, and unicode character.
Confirm `slugify()` produces valid filenames for all chapters on both macOS and
Windows. Semaphore caps concurrent calls at the provider-aware default. A
simulated 429 mid-run confirms tenacity retries without losing the chapter.
All chapter decks land in `outputs/<slug>/` with a valid `index.json`. Viewer
renders the TOC page correctly.

**Milestone 8 — CLI and viewer polished**
`--estimate`, `--open`, `--fast`, `--debug`, and `--max-concurrent` all work.
FastAPI server confirmed in Chrome, Firefox, and Safari for both single JSON and
index.json modes. Web studio upload flow works end to end. A non-developer can
follow the README successfully.

**Milestone 9 — Multi-provider support**
OpenAI and Ollama providers implemented. Provider swap confirmed with one config
line change. Pydantic retry, provider-aware semaphore, and tenacity backoff all
confirmed working. Ollama confirmed to run sequentially at `max_concurrent = 1`
with no crashes under a 200-page textbook. `PRICING` table verified accurate.

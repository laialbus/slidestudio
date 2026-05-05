"""
FastAPI server for SlideStudio.

This module is an entry point alongside cli.py and is permitted to import
config.py directly.  All other modules receive config values via injection.

Endpoints:
  POST /upload          — accept a PDF, save to pdfs/, enqueue generation
  GET  /status/{job_id} — poll job progress and retrieve the output URL
  GET  /library         — proxy outputs/library.json
  Static mounts:
    /outputs            — generated slide JSON files
    /exporters/html     — viewer HTML/JS/CSS (html=True for index fallback)
"""

import asyncio
import importlib
import inspect
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.refiner import RefinerAgent
from agents.writer import WriterAgent
from pipeline import run as pipeline_run
from providers.base import BaseProvider
from providers.config import ProviderConfig
from providers.errors import CircuitOpenError, FatalAPIError
from utils.checkpoint import Checkpoint
from utils.slugify import slugify

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.resolve()
_PDFS_DIR     = _PROJECT_ROOT / "pdfs"
_OUTPUTS_DIR  = _PROJECT_ROOT / "outputs"


# ── Job state ─────────────────────────────────────────────────────────────────

@dataclass
class JobState:
    status: str       # "queued" | "running" | "done" | "error"
    pdf_path: Path
    progress: dict = field(default_factory=dict)
    output_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


_jobs: dict[str, JobState] = {}


# ── Provider helpers (mirrors cli.py pattern) ─────────────────────────────────

def _discover_provider_class(provider_key: str) -> type[BaseProvider] | None:
    base = provider_key.split("-")[0]
    try:
        module = importlib.import_module(f"providers.{base}")
    except ImportError:
        return None
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if (
            cls.__module__ == module.__name__
            and issubclass(cls, BaseProvider)
            and cls is not BaseProvider
        ):
            return cls
    return None


_PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    key: cls
    for key in config.MODELS
    if (cls := _discover_provider_class(key)) is not None
}


def _resolve_api_key(provider_key: str) -> str:
    base = provider_key.split("-")[0].upper()
    return os.environ.get(f"{base}_API_KEY", "")


def _build_agents(provider_key: str) -> dict:
    provider_cls = _PROVIDER_REGISTRY.get(provider_key)
    if provider_cls is None:
        raise ValueError(f"Provider {provider_key!r} is not available")
    model_name = config.MODELS[provider_key]
    provider_cfg = ProviderConfig(
        model=model_name,
        max_concurrent=config.PIPELINE["max_concurrent"],
        max_format_retries=config.PIPELINE["max_format_retries"],
        max_rate_limit_retries=config.PIPELINE["max_rate_limit_retries"],
        request_timeout=config.PIPELINE["request_timeout"],
        circuit_breaker_threshold=config.PIPELINE["circuit_breaker_threshold"],
        circuit_breaker_cooldown=config.PIPELINE["circuit_breaker_cooldown"],
        backoff_wait_min=config.PIPELINE["backoff_wait_min"],
        backoff_wait_max=config.PIPELINE["backoff_wait_max"],
    )
    provider_instance = provider_cls(
        config=provider_cfg,
        api_key=_resolve_api_key(provider_key),
    )
    return {
        "analyst": AnalystAgent(provider_instance),
        "planner": PlannerAgent(provider_instance),
        "writer":  WriterAgent(
            provider_instance,
            writer_batch_size=config.PIPELINE["writer_batch_size"],
        ),
        "critic":  CriticAgent(provider_instance),
        "refiner": RefinerAgent(provider_instance),
    }


def _build_checkpoint(pdf_path: Path, model_name: str) -> Checkpoint:
    try:
        run_key = Checkpoint.compute_key(
            pdf_path, model_name, config.PIPELINE["chunk_size"]
        )
    except OSError:
        import hashlib
        run_key = hashlib.sha256(str(pdf_path).encode()).hexdigest()[:16]
    return Checkpoint(
        base_dir=_PROJECT_ROOT / ".checkpoints",
        run_key=run_key,
        resume=False,
    )


# ── Background pipeline task ──────────────────────────────────────────────────

async def _run_pipeline_job(job_id: str, pdf_path: Path) -> None:
    job = _jobs[job_id]
    job.status = "running"

    provider_key = config.PROVIDER
    model_name = config.MODELS[provider_key]

    def progress_cb(stage: str, completed: int, total: int) -> None:
        job.progress[stage] = {"completed": completed, "total": total}

    try:
        agents = _build_agents(provider_key)
        ck = _build_checkpoint(pdf_path, model_name)

        _, _, output_path = await pipeline_run(
            file_path=pdf_path,
            agents=agents,
            output_dir=_OUTPUTS_DIR,
            chunk_size=config.PIPELINE["chunk_size"],
            overlap_size=config.PIPELINE["overlap_size"],
            multi_deck_chapter_threshold=config.PIPELINE["multi_deck_chapter_threshold"],
            multi_deck_length_threshold=config.PIPELINE["multi_deck_length_threshold"],
            max_review_cycles=config.PIPELINE["max_review_cycles"],
            debug=config.PIPELINE["debug"],
            checkpoint=ck,
            on_progress=progress_cb,
        )

        job.status = "done"
        if output_path is not None:
            job.output_url = "/" + output_path.relative_to(_PROJECT_ROOT).as_posix()

    except (CircuitOpenError, FatalAPIError) as exc:
        job.status = "error"
        job.error = f"API error: {exc}"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)


# ── FastAPI app factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Build and return the FastAPI application.

    Directories are created here so a fresh clone works without manual setup.
    API routes are registered before StaticFiles mounts so the router
    resolves /upload and /status before the static catch-all.
    """
    _PDFS_DIR.mkdir(exist_ok=True)
    _OUTPUTS_DIR.mkdir(exist_ok=True)

    app = FastAPI(title="SlideStudio", docs_url=None, redoc_url=None)

    # ── API routes ─────────────────────────────────────────────────────────

    @app.post("/upload")
    async def upload_pdf(file: UploadFile = File(...)):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        slug = slugify(Path(file.filename).stem) or "upload"
        pdf_path = _PDFS_DIR / f"{slug}.pdf"

        content = await file.read()
        pdf_path.write_bytes(content)

        job_id = str(uuid.uuid4())
        _jobs[job_id] = JobState(status="queued", pdf_path=pdf_path)
        asyncio.create_task(_run_pipeline_job(job_id, pdf_path))

        return {"job_id": job_id}

    @app.get("/status/{job_id}")
    async def job_status(job_id: str):
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {
            "status":     job.status,
            "progress":   job.progress,
            "output_url": job.output_url,
            "error":      job.error,
            "created_at": job.created_at,
        }

    @app.get("/library")
    async def get_library():
        manifest = _OUTPUTS_DIR / "library.json"
        if not manifest.exists():
            return JSONResponse([])
        return JSONResponse(json.loads(manifest.read_text(encoding="utf-8")))

    # ── Static file mounts — registered after API routes ──────────────────

    app.mount(
        "/outputs",
        StaticFiles(directory=str(_OUTPUTS_DIR)),
        name="outputs",
    )
    app.mount(
        "/exporters/html",
        StaticFiles(
            directory=str(_PROJECT_ROOT / "exporters" / "html"),
            html=True,
        ),
        name="viewer",
    )

    return app


# ── Public entry point (replaces exporters/html_server.py) ───────────────────

def serve_and_open(output_path: Path | None, port: int) -> None:
    """
    Start the FastAPI server on localhost:{port} and open the viewer.
    Blocks until Ctrl+C.
    """
    import threading
    import webbrowser

    import uvicorn
    from rich import print as rprint
    from rich.panel import Panel

    if output_path is not None:
        try:
            relative = output_path.relative_to(Path.cwd())
        except ValueError:
            relative = output_path
        viewer_url = (
            f"http://localhost:{port}/exporters/html/index.html"
            f"?file=/{relative}"
        )
    else:
        viewer_url = f"http://localhost:{port}/exporters/html/index.html"

    threading.Timer(0.5, webbrowser.open, args=[viewer_url]).start()

    rprint(Panel(
        f"[bold green]Server running on http://localhost:{port}[/bold green]\n"
        f"[cyan]Viewer:[/cyan]  {viewer_url}\n\n"
        f"[dim]Press [bold]Ctrl+C[/bold] to stop the server.[/dim]",
        title="[bold white]SlideStudio Viewer[/bold white]",
        border_style="bright_blue",
        expand=False,
    ))

    uvicorn.run(create_app(), host="localhost", port=port, log_level="warning")

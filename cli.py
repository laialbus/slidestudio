#!/usr/bin/env python3

import asyncio
import importlib
import inspect
import os
from pathlib import Path
from typing import Callable, Optional

import typer
from dotenv import load_dotenv
from rich import print as rprint
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

import config
from providers.config import ProviderConfig
from providers.errors import CircuitOpenError, FatalAPIError
from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.refiner import RefinerAgent
from agents.writer import WriterAgent
from pipeline import estimate as pipeline_estimate
from pipeline import run as pipeline_run
from utils.library import rebuild_library_manifest
from providers.base import BaseProvider
from utils.checkpoint import Checkpoint, resolve_output_path

load_dotenv()

app = typer.Typer()


def _make_progress_callback(
    progress: Progress,
) -> tuple[Callable[[str, int, int], None], Callable[[], None]]:
    task_ids: dict[str, int] = {}

    _stage_labels: dict[str, str] = {
        "extract": "Extracting text",
        "analyst": "Analyst",
        "planner": "Planner",
        "writer":  "Writer",
        "review":  "Review",
        "chapter": "Generating chapters",
    }

    def callback(stage: str, completed: int, total: int) -> None:
        if stage not in task_ids:
            label = _stage_labels.get(stage, stage)
            task_ids[stage] = progress.add_task(label, total=total)
        progress.update(task_ids[stage], completed=completed, total=total)

    def finish_all() -> None:
        for tid in task_ids.values():
            progress.stop_task(tid)

    return callback, finish_all


# ──────────────────────────────────────────────────────────────
# Provider discovery
#
# config.MODELS is the single source of truth for which providers exist.
# For each key we:
#   1. Strip any "-variant" suffix (e.g. "google-fast" → "google") to get the
#      module name, since variants share an implementation and differ only in
#      model string.
#   2. Import providers.<base>.
#   3. Pick the lone BaseProvider subclass DEFINED in that module.
#
# A key is silently skipped if its SDK is missing or the module is a stub with
# no subclass yet. Adding a provider = add to config.MODELS + create the module.
# ──────────────────────────────────────────────────────────────

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
    # Variants share an env var with their base ("google-fast" uses GOOGLE_API_KEY).
    base = provider_key.split("-")[0].upper()
    return os.environ.get(f"{base}_API_KEY", "")


def _resolve_provider_class(provider_key: str) -> type[BaseProvider]:
    try:
        return _PROVIDER_REGISTRY[provider_key]
    except KeyError as e:
        available = ", ".join(sorted(_PROVIDER_REGISTRY)) or "<none>"
        rprint(
            f"[red]Provider {provider_key!r} is not available.[/red]\n"
            f"Registered providers: {available}\n"
            f"If you expected {provider_key!r} to work, ensure its SDK is installed "
            f"and providers/{provider_key.split('-')[0]}.py defines a BaseProvider subclass."
        )
        raise typer.Exit(code=1) from e


@app.command()
def run(
    pdf_path: str,
    open_browser: bool = typer.Option(False, "--open"),
    fast: bool = False,
    debug: bool = False,
    max_concurrent: Optional[int] = typer.Option(None, "--max-concurrent"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Override config.PROVIDER"),
    resume: bool = typer.Option(False, "--resume", help="Load completed stages from cache"),
    force: bool = typer.Option(False, "--force", help="Ignore cache, run fresh, overwrite cache"),
):
    provider_key = provider or config.PROVIDER
    model_name   = config.MODELS[provider_key]

    _max_concurrent = max_concurrent if max_concurrent is not None else config.PIPELINE["max_concurrent"]
    max_cycles = 0 if fast else config.PIPELINE["max_review_cycles"]
    effective_debug = debug or config.PIPELINE["debug"]

    checkpoint = _build_checkpoint(
        pdf_path=Path(pdf_path).resolve(),
        model_name=model_name,
        chunk_size=config.PIPELINE["chunk_size"],
        resume=resume and not force,
    )

    provider_cls = _resolve_provider_class(provider_key)
    provider_config = ProviderConfig(
        model=model_name,
        max_concurrent=_max_concurrent,
        max_format_retries=config.PIPELINE["max_format_retries"],
        max_rate_limit_retries=config.PIPELINE["max_rate_limit_retries"],
        request_timeout=config.PIPELINE["request_timeout"],
        circuit_breaker_threshold=config.PIPELINE["circuit_breaker_threshold"],
        circuit_breaker_cooldown=config.PIPELINE["circuit_breaker_cooldown"],
        backoff_wait_min=config.PIPELINE["backoff_wait_min"],
        backoff_wait_max=config.PIPELINE["backoff_wait_max"],
    )
    provider_instance = provider_cls(
        config=provider_config,
        api_key=_resolve_api_key(provider_key),
    )
    agents = {
        "analyst": AnalystAgent(provider_instance),
        "planner": PlannerAgent(provider_instance),
        "writer":  WriterAgent(provider_instance, writer_batch_size=config.PIPELINE["writer_batch_size"]),
        "critic":  CriticAgent(provider_instance),
        "refiner": RefinerAgent(provider_instance),
    }

    pipeline_error: Exception | None = None
    result = issues = output_path = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description:<30}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        on_progress_cb, finish_all = _make_progress_callback(progress)
        try:
            result, issues, output_path = asyncio.run(
                _run_pipeline(
                    pdf_path, agents,
                    max_cycles, effective_debug, checkpoint,
                    on_progress=on_progress_cb,
                )
            )
        except (CircuitOpenError, FatalAPIError) as exc:
            pipeline_error = exc
        finally:
            finish_all()

    if pipeline_error is not None:
        if isinstance(pipeline_error, CircuitOpenError):
            rprint(f"[red]Pipeline stopped: {pipeline_error}[/red]")
        else:
            rprint(f"[red]API error (non-retryable): {pipeline_error}[/red]")
        raise typer.Exit(code=1)

    if output_path is not None:
        # Run completed — drop the resumable stage cache, then re-pin the output
        # pointer so `serve` can still find the deck without re-running.
        checkpoint.clear()
        checkpoint.save_output_path(output_path)

    if issues:
        rprint("[yellow]Completed with warnings:[/yellow]")
        for issue in issues:
            rprint(f"  {issue}")
    else:
        rprint(f"[green]Done — {output_path}[/green]")

    if open_browser:
        from server import serve_and_open
        serve_and_open(output_path, config.PIPELINE["port"])


@app.command()
def estimate(pdf_path: str):
    """Dry-run cost estimate — prints token usage and cost without making API calls."""
    provider_key = config.PROVIDER
    model_name   = config.MODELS[provider_key]
    rprint(pipeline_estimate(
        file_path=Path(pdf_path),
        chunk_size=config.PIPELINE["chunk_size"],
        overlap_size=config.PIPELINE["overlap_size"],
        provider_key=provider_key,
        model_name=model_name,
    ))


@app.command("library-refresh")
def library_refresh():
    """Rebuild the library manifest by scanning the outputs/ directory."""
    outputs_dir = Path("outputs").resolve()
    if not outputs_dir.is_dir():
        rprint("[yellow]No outputs/ directory found. Run a pipeline first.[/yellow]")
        raise typer.Exit(code=1)
    entries = rebuild_library_manifest(outputs_dir)
    n = len(entries)
    rprint(f"[green]Rebuilt library.json — {n} paper{'s' if n != 1 else ''} indexed.[/green]")


@app.command()
def serve(pdf_path: Optional[str] = typer.Argument(None)):
    """Open the viewer. Omit pdf_path to open the library home."""
    from server import serve_and_open
    if pdf_path is None:
        serve_and_open(None, config.PIPELINE["port"])
        return
    provider_key = config.PROVIDER
    model_name   = config.MODELS[provider_key]
    output_path  = resolve_output_path(
        pdf_path=str(Path(pdf_path).resolve()),
        model=model_name,
        chunk_size=config.PIPELINE["chunk_size"],
    )
    if output_path is None:
        rprint(
            f"[red]No output found for '{pdf_path}'.[/red]\n"
            f"Run [bold]python cli.py run {pdf_path}[/bold] first to generate slides."
        )
        raise typer.Exit(code=1)
    serve_and_open(output_path, config.PIPELINE["port"])


def _build_checkpoint(
    pdf_path: Path,
    model_name: str,
    chunk_size: int,
    resume: bool,
) -> Checkpoint:
    try:
        run_key = Checkpoint.compute_key(pdf_path, model_name, chunk_size)
    except OSError:
        # PDF doesn't exist yet — the pipeline will fail at extraction.
        # Use a path-based fallback key so the Checkpoint object is still valid.
        import hashlib
        run_key = hashlib.sha256(str(pdf_path).encode()).hexdigest()[:16]
    cache_dir = Path(".checkpoints").resolve()
    return Checkpoint(base_dir=cache_dir, run_key=run_key, resume=resume)


async def _run_pipeline(
    pdf_path: str,
    agents: dict,
    max_review_cycles: int,
    debug: bool,
    checkpoint: Checkpoint,
    on_progress: Callable[[str, int, int], None] | None = None,
):
    return await pipeline_run(
        file_path=Path(pdf_path).resolve(),
        agents=agents,
        output_dir=Path("outputs").resolve(),
        chunk_size=config.PIPELINE["chunk_size"],
        overlap_size=config.PIPELINE["overlap_size"],
        multi_deck_chapter_threshold=config.PIPELINE["multi_deck_chapter_threshold"],
        multi_deck_length_threshold=config.PIPELINE["multi_deck_length_threshold"],
        max_review_cycles=max_review_cycles,
        debug=debug,
        duplicate_policy=config.PIPELINE["duplicate_policy"],
        checkpoint=checkpoint,
        on_progress=on_progress,
    )


if __name__ == "__main__":
    app()

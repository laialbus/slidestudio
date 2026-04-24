#!/usr/bin/env python3

import asyncio
import importlib
import inspect
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import print as rprint

import config
from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.refiner import RefinerAgent
from agents.writer import WriterAgent
from extractors.pdf import PDFExtractor
from pipeline import route
from providers.base import BaseProvider
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from utils.cost_estimator import analyze_pdf_cost

load_dotenv()

app = typer.Typer()


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
def main(
    pdf_path: str,
    estimate: bool = False,
    open_browser: bool = typer.Option(False, "--open"),
    fast: bool = False,
    debug: bool = False,
    max_concurrent: Optional[int] = typer.Option(None, "--max-concurrent"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Override config.PROVIDER"),
):
    provider_key = provider or config.PROVIDER
    model_name   = config.MODELS[provider_key]

    if estimate:
        extractor = PDFExtractor(
            chunk_size=config.PIPELINE["chunk_size"],
            overlap_size=config.PIPELINE["overlap_size"],
        )
        extraction = extractor.extract(pdf_path)
        rprint(analyze_pdf_cost(extraction, provider_key, model_name))
        raise typer.Exit()

    _max_concurrent = max_concurrent if max_concurrent is not None else config.PIPELINE["max_concurrent"]
    max_cycles = 0 if fast else config.PIPELINE["max_review_cycles"]
    effective_debug = debug or config.PIPELINE["debug"]

    provider_cls = _resolve_provider_class(provider_key)
    provider_instance = provider_cls(
        provider_name=provider_key,
        model=model_name,
        api_key=_resolve_api_key(provider_key),
        max_concurrent=_max_concurrent,
        max_format_retries=config.PIPELINE["max_format_retries"],
        max_rate_limit_retries=6,
    )
    agents = {
        "planner": PlannerAgent(provider_instance),
        "writer":  WriterAgent(provider_instance, writer_batch_size=config.PIPELINE["writer_batch_size"]),
        "critic":  CriticAgent(provider_instance),
        "refiner": RefinerAgent(provider_instance),
    }

    result, issues, output_path = asyncio.run(
        _run_pipeline(pdf_path, provider_instance, agents, max_cycles, effective_debug)
    )

    if issues:
        rprint("[yellow]Completed with warnings:[/yellow]")
        for issue in issues:
            rprint(f"  {issue}")
    else:
        rprint(f"[green]Done — {output_path}[/green]")

    if open_browser:
        from exporters.html_server import serve_and_open
        serve_and_open(output_path, config.PIPELINE["port"])


async def _run_pipeline(
    pdf_path: str,
    provider,
    agents: dict,
    max_review_cycles: int,
    debug: bool,
):
    extractor = PDFExtractor(
        chunk_size=config.PIPELINE["chunk_size"],
        overlap_size=config.PIPELINE["overlap_size"],
    )
    extraction = extractor.extract(pdf_path)
    doc_map = await AnalystAgent(provider).run(extraction)

    skeleton = GlobalSkeleton(
        title=Path(pdf_path).stem,
        document_type=doc_map.document_type,
        core_thesis=doc_map.core_thesis[:400],
        sections=[
            SectionEntry(heading=h, level=1, position=i)
            for i, h in enumerate(extraction["headers"])
        ],
    )

    return await route(
        title=Path(pdf_path).stem,
        skeleton=skeleton,
        doc_map=doc_map,
        chunks=extraction["chunks"],
        agents=agents,
        multi_deck_threshold=config.PIPELINE["multi_deck_threshold"],
        max_review_cycles=max_review_cycles,
        debug=debug,
        output_dir=Path("outputs"),
    )


if __name__ == "__main__":
    app()

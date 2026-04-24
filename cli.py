#!/usr/bin/env python3

import asyncio
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
from providers.anthropic import AnthropicProvider
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from utils.cost_estimator import analyze_pdf_cost

load_dotenv()

app = typer.Typer()

# need a modular way to register providers without hardcoding.
_PROVIDER_REGISTRY: dict[str, type] = {
    "anthropic": AnthropicProvider,
}

try:
    from providers.google import GoogleProvider
    _PROVIDER_REGISTRY["google"] = GoogleProvider
    _PROVIDER_REGISTRY["google-fast"] = GoogleProvider
except ImportError:
    pass


@app.command()
def main(
    pdf_path: str,
    estimate: bool = False,
    open_browser: bool = typer.Option(False, "--open"),
    fast: bool = False,
    debug: bool = False,
    max_concurrent: Optional[int] = typer.Option(None, "--max-concurrent"),
):
    if estimate:
        extractor = PDFExtractor(
            chunk_size=config.PIPELINE["chunk_size"],
            overlap_size=config.PIPELINE["overlap_size"],
        )
        extraction = extractor.extract(pdf_path)
        result = analyze_pdf_cost(
            extraction,
            config.PROVIDER,
            config.MODELS[config.PROVIDER],
        )
        rprint(result)
        raise typer.Exit()

    _max_concurrent = max_concurrent if max_concurrent is not None else config.PIPELINE["max_concurrent"]
    max_cycles = 0 if fast else config.PIPELINE["max_review_cycles"]
    effective_debug = debug or config.PIPELINE["debug"]

    provider_cls = _PROVIDER_REGISTRY[config.PROVIDER]
    provider = provider_cls(
        provider_name=config.PROVIDER,
        model=config.MODELS[config.PROVIDER],
        api_key=os.environ.get(f"{config.PROVIDER.upper()}_API_KEY", ""),
        max_concurrent=_max_concurrent,
        max_format_retries=config.PIPELINE["max_format_retries"],
        max_rate_limit_retries=6,
    )
    agents = {
        "planner": PlannerAgent(provider),
        "writer":  WriterAgent(provider, writer_batch_size=config.PIPELINE["writer_batch_size"]),
        "critic":  CriticAgent(provider),
        "refiner": RefinerAgent(provider),
    }

    result, issues, output_path = asyncio.run(
        _run_pipeline(pdf_path, provider, agents, max_cycles, effective_debug)
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

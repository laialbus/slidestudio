import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agents.analyst import AnalystAgent, AnalystResult
from extractors.pdf import PDFExtractor
from schemas.deck_index import DeckEntry, DeckIndex
from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import SlidePlan
from schemas.slides_draft import SlidesDraft
from schemas.slides_final import FinalSlide, SlidesFinal
from utils.checkpoint import Checkpoint
from utils.cost_estimator import analyze_pdf_cost
from utils.slugify import slugify


ProgressCallback = Callable[[str, int, int], None] | None


def _notify(cb: ProgressCallback, stage: str, completed: int, total: int) -> None:
    if cb is not None:
        cb(stage, completed, total)


def _write_debug(title: str, output_dir: Path | str, intermediates: dict) -> None:
    """Write debug intermediates to output_dir/debug/<slug>/."""
    if not intermediates:
        return
    slug = slugify(title)
    debug_dir = Path(output_dir) / "debug" / slug
    debug_dir.mkdir(parents=True, exist_ok=True)
    for fname, key in [
        ("01_slide_plan.json", "slide_plan"),
        ("02_slides_draft.json", "slides_draft"),
        ("03_critique.json", "critique"),
    ]:
        if key in intermediates:
            (debug_dir / fname).write_text(
                intermediates[key].model_dump_json(indent=2), encoding="utf-8"
            )


def write_output(
    slides_final: SlidesFinal,
    title: str,
    debug: bool,
    output_dir: Path | str,
    intermediates: dict,
) -> Path:
    slug = slugify(title)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{slug}.json"
    output_path.write_text(slides_final.model_dump_json(indent=2), encoding="utf-8")
    if debug:
        _write_debug(title, out_dir, intermediates)
    return output_path


async def run_single_deck(
    title: str,
    doc_map,
    skeleton,
    chunks: list[str],
    agents: dict,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
    scope: SectionEntry | None = None,
    checkpoint: Checkpoint | None = None,
    _write: bool = True,
    on_progress: ProgressCallback = None,
) -> tuple[SlidesFinal, list[str], Path | None]:
    """
    Runs the planner→writer→critic/refiner loop for one deck.

    _write=False skips the main JSON write (used when the caller, e.g.
    run_multi_deck, delegates all disk I/O to write_deck_index). Debug
    intermediates are still written when debug=True.
    """
    ck = checkpoint

    # If the final output is already checkpointed, skip all agents.
    if ck:
        slides_final_cp = ck.load("slides_final", SlidesFinal)
        if slides_final_cp is not None:
            _notify(on_progress, "planner", 1, 1)
            _notify(on_progress, "writer", 1, 1)
            if max_review_cycles > 0:
                _notify(on_progress, "review", max_review_cycles, max_review_cycles)
            output_path = write_output(slides_final_cp, title, debug, output_dir, {}) if _write else None
            return slides_final_cp, [], output_path

    # Planner — load from checkpoint or run
    slide_plan = ck.load("slide_plan", SlidePlan) if ck else None
    if slide_plan is None:
        slide_plan = await agents["planner"].run(
            doc_map=doc_map, skeleton=skeleton, scope=scope
        )
        if ck:
            ck.save("slide_plan", slide_plan)
    _notify(on_progress, "planner", 1, 1)

    draft = await agents["writer"].run(
        slide_plan=slide_plan, doc_map=doc_map, chunks=chunks
    )
    _notify(on_progress, "writer", 1, 1)

    best_draft, unresolved = await run_review_loop(
        draft, doc_map, agents["critic"], agents["refiner"], max_review_cycles,
        on_progress=on_progress,
    )
    final_critique = await agents["critic"].run(doc_map=doc_map, slides=best_draft)

    slides_final = SlidesFinal(
        title=best_draft.title,
        slides=[FinalSlide(**s.model_dump()) for s in best_draft.slides],
    )

    if ck:
        ck.save("slides_final", slides_final)

    intermediates = {
        "slide_plan": slide_plan,
        "slides_draft": best_draft,
        "critique": final_critique,
    }

    if _write:
        output_path = write_output(slides_final, title, debug, output_dir, intermediates)
    else:
        if debug:
            _write_debug(title, output_dir, intermediates)
        output_path = None

    return slides_final, unresolved, output_path


def write_deck_index(
    title: str,
    decks_data: list[tuple],
    agents: dict,
    output_dir: Path | str,
) -> tuple[DeckIndex, Path]:
    slug = slugify(title)
    out_dir = Path(output_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = agents["planner"].provider.name
    model = agents["planner"].provider.model

    decks = []
    for i, (section, slides_final) in enumerate(decks_data, start=1):
        filename = f"{i:02d}_{slugify(section.heading)}.json"
        (out_dir / filename).write_text(
            slides_final.model_dump_json(indent=2), encoding="utf-8"
        )
        decks.append(DeckEntry(chapter_title=section.heading, file=filename))

    deck_index = DeckIndex(
        title=title,
        type="multi_deck",
        generated_at=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        model=model,
        decks=decks,
    )
    index_path = out_dir / "index.json"
    index_path.write_text(deck_index.model_dump_json(indent=2), encoding="utf-8")
    return deck_index, index_path


async def run_multi_deck(
    title: str,
    doc_map,
    skeleton,
    chunks: list[str],
    agents: dict,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
    checkpoint: Checkpoint | None = None,
    on_progress: ProgressCallback = None,
) -> tuple[DeckIndex, list[str], Path]:
    chapters = [s for s in skeleton.sections if s.level == 1]
    total = len(chapters)
    completed_count = [0]

    async def _run_chapter(chapter):
        result = await run_single_deck(
            title=chapter.heading,
            doc_map=doc_map,
            skeleton=skeleton,
            chunks=chunks,
            agents=agents,
            max_review_cycles=max_review_cycles,
            debug=debug,
            output_dir=output_dir,
            scope=chapter,
            checkpoint=checkpoint.scoped(chapter.heading) if checkpoint else None,
            _write=False,
            on_progress=None,
        )
        completed_count[0] += 1
        _notify(on_progress, "chapter", completed_count[0], total)
        return result

    tasks = [_run_chapter(chapter) for chapter in chapters]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    decks_data = [
        (chapter, result[0])
        for chapter, result in zip(chapters, results)
        if not isinstance(result, Exception)
    ]

    deck_index, index_path = write_deck_index(title, decks_data, agents, output_dir)
    return deck_index, [], index_path


def _cleanup_stale_output(output_dir: Path | str, title: str, is_multi: bool) -> None:
    """Remove the previous output for this document so a fresh run starts clean."""
    out = Path(output_dir)
    slug = slugify(title)
    if is_multi:
        stale = out / slug
        if stale.is_dir():
            shutil.rmtree(stale)
    else:
        stale = out / f"{slug}.json"
        if stale.exists():
            stale.unlink()


async def route(
    title: str,
    skeleton,
    doc_map,
    chunks: list[str],
    agents: dict,
    multi_deck_chapter_threshold: int,
    multi_deck_length_threshold: int,
    total_chars: int,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
    checkpoint: Checkpoint | None = None,
    on_progress: ProgressCallback = None,
):
    chapter_count = sum(1 for s in skeleton.sections if s.level == 1)
    is_multi = (chapter_count > multi_deck_chapter_threshold) and (total_chars > multi_deck_length_threshold)

    # On a fresh run (not resuming from checkpoint), remove stale output so the
    # directory structure is always an exact reflection of the current run.
    is_resuming = checkpoint is not None and checkpoint._resume
    if not is_resuming:
        _cleanup_stale_output(output_dir, title, is_multi)

    if is_multi:
        return await run_multi_deck(
            title, doc_map, skeleton, chunks, agents,
            max_review_cycles, debug, output_dir,
            checkpoint=checkpoint,
            on_progress=on_progress,
        )
    return await run_single_deck(
        title, doc_map, skeleton, chunks, agents,
        max_review_cycles, debug, output_dir,
        checkpoint=checkpoint,
        on_progress=on_progress,
    )


def estimate(
    file_path: Path,
    chunk_size: int,
    overlap_size: int,
    provider_key: str,
    model_name: str,
) -> dict:
    extractor = PDFExtractor(chunk_size=chunk_size, overlap_size=overlap_size)
    extraction = extractor.extract(str(file_path))
    return analyze_pdf_cost(extraction, provider_key, model_name)


async def run(
    file_path: Path,
    provider,
    agents: dict,
    output_dir: Path,
    chunk_size: int,
    overlap_size: int,
    multi_deck_chapter_threshold: int,
    multi_deck_length_threshold: int,
    max_review_cycles: int,
    debug: bool,
    checkpoint: Checkpoint | None = None,
    on_progress: ProgressCallback = None,
):
    extractor = PDFExtractor(chunk_size=chunk_size, overlap_size=overlap_size)
    extraction = extractor.extract(str(file_path))
    _notify(on_progress, "extract", 1, 1)

    ck = checkpoint

    # Analyst — load skeleton + doc_map from checkpoint or run fresh
    skeleton_cp = ck.load("skeleton", GlobalSkeleton) if ck else None
    doc_map_cp  = ck.load("doc_map", DocumentMap) if ck else None

    if skeleton_cp is not None and doc_map_cp is not None:
        analyst_result = AnalystResult(skeleton=skeleton_cp, doc_map=doc_map_cp)
    else:
        analyst_result = await AnalystAgent(provider).run(extraction)
        if ck:
            ck.save("skeleton", analyst_result.skeleton)
            ck.save("doc_map", analyst_result.doc_map)
    _notify(on_progress, "analyst", 1, 1)

    skeleton = analyst_result.skeleton
    doc_map  = analyst_result.doc_map

    chunks = extraction["chunks"]
    total_chars = sum(len(c) for c in chunks)

    if debug:
        level_counts: dict[int, int] = {}
        for s in skeleton.sections:
            level_counts[s.level] = level_counts.get(s.level, 0) + 1
        dist = ", ".join(f"{v} level-{k}" for k, v in sorted(level_counts.items()))
        chapter_count = level_counts.get(1, 0)
        is_multi = (chapter_count > multi_deck_chapter_threshold) and (total_chars > multi_deck_length_threshold)
        mode = "multi-deck" if is_multi else "single-deck"
        print(f"Skeleton: {len(skeleton.sections)} sections — {dist}")
        print(
            f"Router: {chapter_count} chapters, {total_chars:,} chars → {mode} mode"
            f" (chapter threshold: {multi_deck_chapter_threshold},"
            f" length threshold: {multi_deck_length_threshold:,})"
        )

    return await route(
        title=skeleton.title,
        skeleton=skeleton,
        doc_map=doc_map,
        chunks=chunks,
        agents=agents,
        multi_deck_chapter_threshold=multi_deck_chapter_threshold,
        multi_deck_length_threshold=multi_deck_length_threshold,
        total_chars=total_chars,
        max_review_cycles=max_review_cycles,
        debug=debug,
        output_dir=output_dir,
        checkpoint=ck,
        on_progress=on_progress,
    )


async def run_review_loop(
    draft: SlidesDraft,
    doc_map,
    critic,
    refiner,
    max_review_cycles: int,
    on_progress: ProgressCallback = None,
) -> tuple[SlidesDraft, list[str]]:
    current = draft
    unresolved: list[str] = []

    for cycle in range(1, max_review_cycles + 1):
        critique = await critic.run(doc_map=doc_map, slides=current)
        _notify(on_progress, "review", cycle, max_review_cycles)
        failed = critique.failed_slides

        if not failed:
            return current, []

        if cycle == max_review_cycles:
            unresolved = [
                f"Slide {s.index}: {s.issues[0].detail}"
                for s in failed
            ]
            break

        current = await refiner.run(
            doc_map=doc_map,
            slides=current,
            critique=critique,
        )

    return current, unresolved

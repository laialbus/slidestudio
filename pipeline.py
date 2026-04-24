import asyncio
from datetime import datetime, timezone
from pathlib import Path

from schemas.deck_index import DeckEntry, DeckIndex
from schemas.global_skeleton import SectionEntry
from schemas.slides_draft import SlidesDraft
from schemas.slides_final import FinalSlide, SlidesFinal
from utils.slugify import slugify


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
        debug_dir = out_dir / "debug" / slug
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "01_slide_plan.json").write_text(
            intermediates["slide_plan"].model_dump_json(indent=2), encoding="utf-8"
        )
        (debug_dir / "02_slides_draft.json").write_text(
            intermediates["slides_draft"].model_dump_json(indent=2), encoding="utf-8"
        )
        (debug_dir / "03_critique.json").write_text(
            intermediates["critique"].model_dump_json(indent=2), encoding="utf-8"
        )
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
) -> tuple[SlidesFinal, list[str]]:
    slide_plan = await agents["planner"].run(doc_map=doc_map, skeleton=skeleton, scope=scope)
    draft = await agents["writer"].run(
        slide_plan=slide_plan, doc_map=doc_map, chunks=chunks
    )
    best_draft, unresolved = await run_review_loop(
        draft, doc_map, agents["critic"], agents["refiner"], max_review_cycles
    )
    final_critique = await agents["critic"].run(doc_map=doc_map, slides=best_draft)

    slides_final = SlidesFinal(
        title=best_draft.title,
        slides=[FinalSlide(**s.model_dump()) for s in best_draft.slides],
    )
    intermediates = {
        "slide_plan": slide_plan,
        "slides_draft": best_draft,
        "critique": final_critique,
    }
    output_path = write_output(slides_final, title, debug, output_dir, intermediates)
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
    multi_deck_threshold: int,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
) -> tuple[DeckIndex, list[str], Path]:
    chapters = [s for s in skeleton.sections if s.level == 1]
    tasks = [
        run_single_deck(
            title=chapter.heading,
            doc_map=doc_map,
            skeleton=skeleton,
            chunks=chunks,
            agents=agents,
            max_review_cycles=max_review_cycles,
            debug=debug,
            output_dir=output_dir,
            scope=chapter,
        )
        for chapter in chapters
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    decks_data = [
        (chapter, result[0])
        for chapter, result in zip(chapters, results)
        if not isinstance(result, Exception)
    ]

    deck_index, index_path = write_deck_index(title, decks_data, agents, output_dir)
    return deck_index, [], index_path


async def route(
    title: str,
    skeleton,
    doc_map,
    chunks: list[str],
    agents: dict,
    multi_deck_threshold: int,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
):
    chapter_count = sum(1 for s in skeleton.sections if s.level == 1)
    if chapter_count > multi_deck_threshold:
        return await run_multi_deck(
            title, doc_map, skeleton, chunks, agents,
            multi_deck_threshold, max_review_cycles, debug, output_dir,
        )
    return await run_single_deck(
        title, doc_map, skeleton, chunks, agents,
        max_review_cycles, debug, output_dir,
    )


async def run_review_loop(
    draft: SlidesDraft,
    doc_map,
    critic,
    refiner,
    max_review_cycles: int,
) -> tuple[SlidesDraft, list[str]]:
    current = draft
    unresolved: list[str] = []

    for cycle in range(1, max_review_cycles + 1):
        critique = await critic.run(doc_map=doc_map, slides=current)
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

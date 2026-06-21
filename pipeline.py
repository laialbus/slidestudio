import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agents.analyst import AnalystResult
from extractors.base import ExtractionResult
from extractors.factory import make_extractor
from schemas.deck_index import DeckEntry, DeckIndex
from schemas.deck_output import DeckOutput, ImageEntry
from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import SlidePlan
from schemas.slides_draft import SlidesDraft
from schemas.slides_final import FinalSlide, SlidesFinal
from utils.checkpoint import Checkpoint
from utils.cost_estimator import analyze_pdf_cost
from utils.library import remove_library_entries_for_hash, upsert_library_manifest
from utils.pdf_hash import pdf_content_hash
from utils.slugify import slugify


ProgressCallback = Callable[[str, int, int], None] | None


def _notify(cb: ProgressCallback, stage: str, completed: int, total: int) -> None:
    if cb is not None:
        cb(stage, completed, total)


def build_figure_catalog(
    images: list[dict],
    chunk_images: list[list[int]],
    figure_purposes: list[dict],
) -> list[dict]:
    """
    Build the compact figure catalog the Planner chooses from: one entry per
    referenceable figure, each {figure_id, caption, purpose, source_chunk}.

    A figure is referenceable only if its caption matched a [FIGURE_ID] marker
    and landed in a chunk — i.e. it appears in chunk_images. Figures with
    empty or unmatched captions never appear there, so they are excluded from
    the catalog entirely (they would otherwise be unreferenceable dead weight).

    source_chunk is the first chunk the figure's marker fell in, used downstream
    only as a soft ranking signal. purpose is the Analyst's conceptual/evidential
    hint, or "unspecified" when the Analyst did not classify it.
    """
    caption_by_id = {img["index"]: img.get("caption", "") for img in images}

    source_chunk: dict[int, int] = {}
    for chunk_idx, fig_ids in enumerate(chunk_images):
        for fig_id in fig_ids:
            source_chunk.setdefault(fig_id, chunk_idx)

    catalog: list[dict] = []
    for fig_id in sorted(source_chunk):
        caption = caption_by_id.get(fig_id, "").strip()
        if not caption:
            continue  # defensive — marker-matched figures always have a caption
        purpose = next(
            (fp[fig_id] for fp in figure_purposes if fig_id in fp),
            "unspecified",
        )
        catalog.append({
            "figure_id":    fig_id,
            "caption":      caption,
            "purpose":      purpose,
            "source_chunk": source_chunk[fig_id],
        })
    return catalog


def _output_slug(title: str) -> str:
    """Slug for output paths. slugify() returns "" for empty or all-symbol
    titles, which would produce a hidden file named ".json" — fall back to
    the same name used for untitled extractions."""
    return slugify(title) or "untitled_document"


def _unique_output_path(out_dir: Path, slug: str, stamp: str, suffix: str = "") -> Path:
    """
    Return out_dir/{slug}_{stamp}{suffix}, appending a counter if two runs
    land in the same second (only possible with duplicate_policy=keep_both).
    """
    candidate = out_dir / f"{slug}_{stamp}{suffix}"
    n = 2
    while candidate.exists():
        candidate = out_dir / f"{slug}_{stamp}_{n}{suffix}"
        n += 1
    return candidate


def _write_debug(title: str, output_dir: Path | str, intermediates: dict) -> None:
    """Write debug intermediates to output_dir/debug/<slug>/."""
    if not intermediates:
        return
    slug = _output_slug(title)
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


def _build_deck_output(
    slides_final: SlidesFinal,
    all_images: list[dict],
    deck_type: str = "single_deck",
    *,
    generated_at: str,
    provider: str,
    model: str,
    doc_hash: str = "",
) -> DeckOutput:
    """Build a DeckOutput, filtering images to only those referenced by the slides."""
    referenced_ids = {
        ref for s in slides_final.slides for ref in s.image_refs
    }
    deck_images = [
        ImageEntry(
            index=img["index"],
            caption=img.get("caption", ""),
            data_uri=img["data_uri"],
            page=img.get("page", 0),
        )
        for img in all_images
        if img["index"] in referenced_ids
    ]
    return DeckOutput(
        title=slides_final.title,
        type=deck_type,
        generated_at=generated_at,
        provider=provider,
        model=model,
        doc_hash=doc_hash,
        slides=slides_final.slides,
        images=deck_images,
    )


def write_output(
    slides_final: SlidesFinal,
    all_images: list[dict],
    title: str,
    debug: bool,
    output_dir: Path | str,
    intermediates: dict,
    provider: str,
    model: str,
    output_stem: str | None = None,
    doc_hash: str = "",
) -> Path:
    # output_stem is the content-keyed file stem ({filename}_{hash}) computed by
    # run(); None falls back to the display title (used by direct unit tests).
    # The hash in the stem is cosmetic — identity is the stored doc_hash field.
    generated = datetime.now(timezone.utc)
    now = generated.isoformat()
    stem = output_stem or _output_slug(title)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = _unique_output_path(
        out_dir, stem, generated.strftime("%Y%m%dT%H%M%S"), suffix=".json"
    )
    deck_output = _build_deck_output(
        slides_final, all_images, generated_at=now,
        provider=provider, model=model, doc_hash=doc_hash,
    )
    output_path.write_text(deck_output.model_dump_json(indent=2), encoding="utf-8")
    upsert_library_manifest(out_dir, {
        "title":        deck_output.title,
        "file":         "/" + output_path.relative_to(out_dir.parent).as_posix(),
        "type":         "single_deck",
        "generated_at": now,
        "provider":     provider,
        "model":        model,
        "doc_hash":     doc_hash,
        "slide_count":  len(deck_output.slides),
        "deck_count":   1,
    })
    if debug:
        _write_debug(title, out_dir, intermediates)
    return output_path


async def run_single_deck(
    title: str,
    doc_map,
    skeleton,
    chunks: list[str],
    images: list[dict],
    agents: dict,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
    figure_catalog: list[dict] = [],
    scope: SectionEntry | None = None,
    checkpoint: Checkpoint | None = None,
    _write: bool = True,
    on_progress: ProgressCallback = None,
    output_stem: str | None = None,
    doc_hash: str = "",
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
            output_path = write_output(
                slides_final_cp, images, title, debug, output_dir, {},
                provider=agents["planner"].provider.name,
                model=agents["planner"].provider.model,
                output_stem=output_stem, doc_hash=doc_hash,
            ) if _write else None
            return slides_final_cp, [], output_path

    # Planner — load from checkpoint or run
    slide_plan = ck.load("slide_plan", SlidePlan) if ck else None
    if slide_plan is None:
        slide_plan = await agents["planner"].run(
            doc_map=doc_map, skeleton=skeleton,
            figure_catalog=figure_catalog, scope=scope,
        )
        if ck:
            ck.save("slide_plan", slide_plan)
    _notify(on_progress, "planner", 1, 1)

    draft = await agents["writer"].run(
        slide_plan=slide_plan, doc_map=doc_map, chunks=chunks
    )
    _notify(on_progress, "writer", 1, 1)

    # Slide index → source chunk indices, so the Critic/Refiner can read the
    # raw material behind each slide and judge depth, not just coherence.
    slide_chunks = {s.index: s.chunk_indices for s in slide_plan.slides}

    # Strip the Summary/Takeaway slide before the review loop — it is generated
    # from the completed deck afterward, not reviewed by the Critic.
    summary_planned = next(
        (s for s in slide_plan.slides if s.tag in ("Summary", "Takeaway")),
        None,
    )
    if summary_planned:
        content_draft = SlidesDraft(
            title=draft.title,
            slides=[s for s in draft.slides if s.index != summary_planned.index],
        )
    else:
        print("Warning: no Summary or Takeaway slide in plan — summary generation skipped.")
        content_draft = draft

    best_draft, unresolved = await run_review_loop(
        content_draft, doc_map, agents["critic"], agents["refiner"], max_review_cycles,
        chunks=chunks, slide_chunks=slide_chunks,
        on_progress=on_progress,
    )

    # Generate the summary from the finished, reviewed content slides.
    if summary_planned:
        summary_draft = await agents["writer"].write_summary(
            completed_slides=best_draft,
            summary_index=summary_planned.index,
        )
        summary_slide = summary_draft.slides[0] if summary_draft.slides else None
    else:
        summary_slide = None

    final_critique = await agents["critic"].run(
        doc_map=doc_map, slides=best_draft,
        chunks=chunks, slide_chunks=slide_chunks,
    )

    final_slides = [FinalSlide(**s.model_dump()) for s in best_draft.slides]
    if summary_slide:
        final_slides.append(FinalSlide(**summary_slide.model_dump()))

    slides_final = SlidesFinal(
        title=best_draft.title,
        slides=final_slides,
    )

    if ck:
        ck.save("slides_final", slides_final)

    intermediates = {
        "slide_plan": slide_plan,
        "slides_draft": best_draft,
        "critique": final_critique,
    }

    if _write:
        output_path = write_output(
            slides_final, images, title, debug, output_dir, intermediates,
            provider=agents["planner"].provider.name,
            model=agents["planner"].provider.model,
            output_stem=output_stem, doc_hash=doc_hash,
        )
    else:
        if debug:
            _write_debug(title, output_dir, intermediates)
        output_path = None

    return slides_final, unresolved, output_path


def write_deck_index(
    title: str,
    decks_data: list[tuple],
    images: list[dict],
    agents: dict,
    output_dir: Path | str,
    output_stem: str | None = None,
    doc_hash: str = "",
) -> tuple[DeckIndex, Path]:
    generated = datetime.now(timezone.utc)
    now = generated.isoformat()
    stem = output_stem or _output_slug(title)
    out_dir = _unique_output_path(
        Path(output_dir), stem, generated.strftime("%Y%m%dT%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = agents["planner"].provider.name
    model = agents["planner"].provider.model

    decks = []
    for i, (section, slides_final) in enumerate(decks_data, start=1):
        filename = f"{i:02d}_{slugify(section.heading)}.json"
        deck_output = _build_deck_output(
            slides_final, images, generated_at=now,
            provider=provider, model=model, doc_hash=doc_hash,
        )
        (out_dir / filename).write_text(
            deck_output.model_dump_json(indent=2), encoding="utf-8"
        )
        decks.append(DeckEntry(chapter_title=section.heading, file=filename))

    deck_index = DeckIndex(
        title=title,
        type="multi_deck",
        generated_at=now,
        provider=provider,
        model=model,
        doc_hash=doc_hash,
        decks=decks,
    )
    index_path = out_dir / "index.json"
    index_path.write_text(deck_index.model_dump_json(indent=2), encoding="utf-8")
    upsert_library_manifest(Path(output_dir), {
        "title":        deck_index.title,
        "file":         "/" + index_path.relative_to(Path(output_dir).parent).as_posix(),
        "type":         "multi_deck",
        "generated_at": now,
        "provider":     provider,
        "model":        model,
        "doc_hash":     doc_hash,
        "slide_count":  sum(len(sf.slides) for _, sf in decks_data),
        "deck_count":   len(decks),
    })
    return deck_index, index_path


async def run_multi_deck(
    title: str,
    doc_map,
    skeleton,
    chunks: list[str],
    images: list[dict],
    agents: dict,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
    figure_catalog: list[dict] = [],
    checkpoint: Checkpoint | None = None,
    on_progress: ProgressCallback = None,
    output_stem: str | None = None,
    doc_hash: str = "",
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
            images=images,
            agents=agents,
            max_review_cycles=max_review_cycles,
            debug=debug,
            output_dir=output_dir,
            figure_catalog=figure_catalog,
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

    deck_index, index_path = write_deck_index(
        title, decks_data, images, agents, output_dir,
        output_stem=output_stem, doc_hash=doc_hash,
    )
    return deck_index, [], index_path


def _cleanup_stale_output(
    output_dir: Path | str, doc_hash: str, keep: Path | None = None
) -> None:
    """
    Remove previous outputs for this document (duplicate_policy="overwrite").

    Identity is the stored `doc_hash` field in library.json — never parsed from
    the filename — so the output naming scheme can change freely without breaking
    cleanup. The manifest is the source of truth: every entry whose doc_hash
    matches is deleted from disk (single-deck file, or the whole directory for a
    multi-deck index) and pruned from the manifest. Archived entries live under
    archive/ and are never matched (the prune preserves them).

    `keep` is the output just written by the current run; it (and its manifest
    entry) is preserved while its older siblings are deleted. This lets the
    caller clean up *after* the new deck is on disk, so a crash mid-generation
    never destroys the prior deck without producing a replacement. A None/empty
    hash is a no-op (unit-test routing paths that don't supply one).
    """
    out = Path(output_dir)
    if not out.is_dir() or not doc_hash:
        return
    manifest_path = out / "library.json"
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []
    keep_file_key = None
    if keep is not None:
        keep_file_key = "/" + Path(keep).relative_to(out.parent).as_posix()
    for e in entries:
        if (
            e.get("archived")
            or e.get("doc_hash") != doc_hash
            or e.get("file") == keep_file_key
        ):
            continue
        target = out.parent / e.get("file", "").lstrip("/")
        # A multi-deck entry points at .../index.json — remove the whole deck dir.
        to_remove = target.parent if e.get("type") == "multi_deck" else target
        if to_remove.is_dir():
            shutil.rmtree(to_remove, ignore_errors=True)
        elif to_remove.exists():
            to_remove.unlink()

    # Prune the now-deleted entries from the manifest (same field-based match).
    remove_library_entries_for_hash(out, doc_hash, keep_file=keep_file_key)


async def route(
    title: str,
    skeleton,
    doc_map,
    chunks: list[str],
    images: list[dict],
    agents: dict,
    multi_deck_chapter_threshold: int,
    multi_deck_length_threshold: int,
    total_chars: int,
    max_review_cycles: int,
    debug: bool,
    output_dir: Path | str,
    duplicate_policy: str,
    figure_catalog: list[dict] = [],
    checkpoint: Checkpoint | None = None,
    on_progress: ProgressCallback = None,
    output_stem: str | None = None,
    doc_hash: str = "",
):
    chapter_count = sum(1 for s in skeleton.sections if s.level == 1)
    is_multi = (chapter_count > multi_deck_chapter_threshold) and (total_chars > multi_deck_length_threshold)

    if is_multi:
        result = await run_multi_deck(
            title, doc_map, skeleton, chunks, images, agents,
            max_review_cycles, debug, output_dir,
            figure_catalog=figure_catalog,
            checkpoint=checkpoint,
            on_progress=on_progress,
            output_stem=output_stem,
            doc_hash=doc_hash,
        )
    else:
        result = await run_single_deck(
            title, doc_map, skeleton, chunks, images, agents,
            max_review_cycles, debug, output_dir,
            figure_catalog=figure_catalog,
            checkpoint=checkpoint,
            on_progress=on_progress,
            output_stem=output_stem,
            doc_hash=doc_hash,
        )

    # Crash-safe overwrite: delete the document's prior outputs (and prune their
    # manifest entries) only *after* the new deck is on disk, excluding the one
    # just written — a crash mid-generation never destroys the prior deck without
    # producing a replacement. This runs on every successful overwrite, resumed
    # or not: a resume only ever continues a *failed* run (success clears the
    # checkpoint), which never wrote a final deck, so the only outputs on disk are
    # prior generations that overwrite should replace. keep_both skips cleanup —
    # its timestamps keep every run distinct.
    new_path = result[2]
    if duplicate_policy == "overwrite" and new_path is not None:
        _cleanup_stale_output(output_dir, doc_hash, keep=new_path)

    return result


def estimate(
    file_path: Path,
    chunk_size: int,
    overlap_size: int,
    extractor: str,
    provider_key: str,
    model_name: str,
) -> dict:
    extraction = make_extractor(extractor, chunk_size, overlap_size).extract(
        str(file_path)
    )
    return analyze_pdf_cost(extraction, provider_key, model_name)


async def run(
    file_path: Path,
    agents: dict,
    output_dir: Path,
    chunk_size: int,
    overlap_size: int,
    extractor: str,
    multi_deck_chapter_threshold: int,
    multi_deck_length_threshold: int,
    max_review_cycles: int,
    debug: bool,
    duplicate_policy: str,
    checkpoint: Checkpoint | None = None,
    on_progress: ProgressCallback = None,
):
    # Output identity is the PDF's content hash, with a readable stem from the
    # source filename: {filename}_{hash}. This decouples disk naming + the
    # overwrite policy from the LLM-extracted title, which can drift between runs
    # (an empty title once fell back to "untitled_document", so a re-run's stale
    # output was never matched and cleaned up — two library entries resulted).
    doc_hash    = pdf_content_hash(file_path)
    name_slug   = slugify(Path(file_path).stem) or "untitled_document"
    output_stem = f"{name_slug}_{doc_hash}"

    extraction: ExtractionResult = make_extractor(
        extractor, chunk_size, overlap_size
    ).extract(str(file_path))
    _notify(on_progress, "extract", 1, 1)

    # Convert images to plain dicts for in-memory passing (agents never import
    # extractor models per CLAUDE.md constraints).
    images = [img.model_dump() for img in extraction.images]

    # Build the extraction dict the Analyst understands.
    # toc_items allows _build_skeleton to skip the LLM call for structured PDFs.
    analyst_input = {
        "toc_items": [item.model_dump() for item in extraction.toc_items],
        "headers":   [item.heading for item in extraction.toc_items],
        "chunks":    extraction.chunks,
        "pdf_title": extraction.pdf_title,
    }

    ck = checkpoint

    # Analyst — load skeleton + doc_map from checkpoint or run fresh
    skeleton_cp = ck.load("skeleton", GlobalSkeleton) if ck else None
    doc_map_cp  = ck.load("doc_map", DocumentMap) if ck else None

    if skeleton_cp is not None and doc_map_cp is not None:
        analyst_result = AnalystResult(skeleton=skeleton_cp, doc_map=doc_map_cp, figure_purposes=[])
    else:
        analyst_result = await agents["analyst"].run(analyst_input)
        if ck:
            ck.save("skeleton", analyst_result.skeleton)
            ck.save("doc_map", analyst_result.doc_map)
    _notify(on_progress, "analyst", 1, 1)

    skeleton        = analyst_result.skeleton
    doc_map         = analyst_result.doc_map
    figure_purposes = analyst_result.figure_purposes

    chunks = extraction.chunks
    chunk_images = extraction.chunk_images
    total_chars = extraction.char_count

    # Compact figure catalog the Planner picks from (built once, shared across decks).
    figure_catalog = build_figure_catalog(images, chunk_images, figure_purposes)

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
        images=images,
        agents=agents,
        multi_deck_chapter_threshold=multi_deck_chapter_threshold,
        multi_deck_length_threshold=multi_deck_length_threshold,
        total_chars=total_chars,
        max_review_cycles=max_review_cycles,
        debug=debug,
        output_dir=output_dir,
        duplicate_policy=duplicate_policy,
        figure_catalog=figure_catalog,
        checkpoint=ck,
        on_progress=on_progress,
        output_stem=output_stem,
        doc_hash=doc_hash,
    )


async def run_review_loop(
    draft: SlidesDraft,
    doc_map,
    critic,
    refiner,
    max_review_cycles: int,
    chunks: list[str] | None = None,
    slide_chunks: dict[int, list[int]] | None = None,
    on_progress: ProgressCallback = None,
) -> tuple[SlidesDraft, list[str]]:
    current = draft
    unresolved: list[str] = []

    for cycle in range(1, max_review_cycles + 1):
        critique = await critic.run(
            doc_map=doc_map, slides=current,
            chunks=chunks, slide_chunks=slide_chunks,
        )
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
            deck_feedback=critique.deck_feedback,
            chunks=chunks,
            slide_chunks=slide_chunks,
        )

    return current, unresolved

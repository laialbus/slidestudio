import json
from pathlib import Path


def upsert_library_manifest(outputs_dir: Path, entry: dict) -> None:
    """Insert or replace an entry in library.json, keeping it sorted newest-first."""
    manifest_path = outputs_dir / "library.json"
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []
    entry = {**entry, "archived": entry.get("archived", False)}
    entries = [e for e in entries if e.get("file") != entry["file"]]
    entries.append(entry)
    entries.sort(key=lambda e: e.get("generated_at", ""), reverse=True)
    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(manifest_path)


def rebuild_library_manifest(outputs_dir: Path) -> list[dict]:
    """Scan outputs_dir and rebuild library.json from scratch."""
    outputs_dir = Path(outputs_dir)
    archive_dir = outputs_dir / "archive"
    entries = []

    for json_path in sorted(outputs_dir.rglob("*.json")):
        if json_path.name == "library.json":
            continue

        parent = json_path.parent
        if parent == outputs_dir:
            archived = False
        elif parent == archive_dir:
            archived = True
        elif parent.parent == outputs_dir and json_path.name == "index.json":
            archived = False
        elif parent.parent == archive_dir and json_path.name == "index.json":
            archived = True
        else:
            # Chapter file, debug output, or deeper nesting — skip
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        deck_type = data.get("type")
        if deck_type not in ("single_deck", "multi_deck"):
            # Infer type for files that predate the type field
            if "slides" in data:
                deck_type = "single_deck"
            elif "decks" in data:
                deck_type = "multi_deck"
            else:
                continue

        file_key = "/" + json_path.relative_to(outputs_dir.parent).as_posix()

        if deck_type == "single_deck":
            slide_count = len(data.get("slides", []))
            deck_count = 1
        else:
            deck_count = len(data.get("decks", []))
            slide_count = 0
            for deck_entry in data.get("decks", []):
                chapter_path = json_path.parent / deck_entry["file"]
                try:
                    chapter_data = json.loads(chapter_path.read_text(encoding="utf-8"))
                    slide_count += len(chapter_data.get("slides", []))
                except (json.JSONDecodeError, OSError):
                    pass

        entries.append({
            "title":        data.get("title", ""),
            "file":         file_key,
            "type":         deck_type,
            "generated_at": data.get("generated_at", ""),
            "provider":     data.get("provider", ""),
            "model":        data.get("model", ""),
            "slide_count":  slide_count,
            "deck_count":   deck_count,
            "archived":     archived,
        })

    entries.sort(key=lambda e: e.get("generated_at", ""), reverse=True)
    manifest_path = outputs_dir / "library.json"
    manifest_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return entries

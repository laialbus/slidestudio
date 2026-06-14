import hashlib
from pathlib import Path

# Streaming read size — hash large PDFs without loading them fully into memory.
_READ_CHUNK = 65536  # 64 KiB

# Hex chars of the digest embedded in output names. 8 hex = 32 bits, ample to
# tell apart the handful of papers in a personal library while staying short
# enough to keep filenames readable.
DOC_HASH_LENGTH = 8


def pdf_content_hash(file_path: Path | str, length: int = DOC_HASH_LENGTH) -> str:
    """
    Return the first `length` hex chars of the SHA-256 digest of a file's raw
    bytes — the document's *content identity*.

    Two byte-identical PDFs hash the same regardless of filename, path, or
    mtime, so the overwrite policy can recognise a re-run of the same paper even
    when the LLM-extracted title drifts (the bug this fixes: an empty title fell
    back to "untitled_document" on one run, so the stale output was never
    matched and cleaned up). Distinct from Checkpoint.compute_key, which hashes
    name+size+mtime as a *cache* key, not content.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()[:length]

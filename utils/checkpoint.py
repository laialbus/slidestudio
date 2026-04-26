import hashlib
from pathlib import Path

from pydantic import BaseModel

from utils.slugify import slugify


class Checkpoint:
    """
    Thin persistence layer for agent outputs.

    Each run is identified by a `run_key` — a SHA-256 digest of the PDF's
    size + mtime + model + chunk_size. Changing any of these produces a
    different key, so stale cache is never reused silently.

    On every run (regardless of --resume / --force) completed stages are
    written to disk. On a --resume run, `load()` will return the cached
    model and the agent is skipped. Without --resume, `load()` always
    returns None so every stage runs fresh.

    Writes are atomic: the model is serialised to a `.tmp` file, then
    promoted with `Path.replace()` (which is atomic on all platforms).
    An incomplete write leaves a `.tmp` file that is not a valid checkpoint
    and is silently overwritten on the next run.
    """

    def __init__(self, base_dir: Path, run_key: str, resume: bool) -> None:
        self._base   = base_dir / run_key
        self._resume = resume

    # ──────────────────────────────────────────────────────────────
    # Core read / write interface
    # ──────────────────────────────────────────────────────────────

    def save(self, stage: str, model: BaseModel) -> None:
        """Atomically write a validated Pydantic model to the checkpoint store."""
        self._base.mkdir(parents=True, exist_ok=True)
        path = self._base / _safe_filename(stage)
        tmp  = path.with_suffix(".tmp")
        tmp.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def load(self, stage: str, schema: type[BaseModel]) -> BaseModel | None:
        """
        Return the checkpointed model if resume mode is active and the file
        exists, otherwise return None so the caller runs the agent normally.
        """
        if not self._resume:
            return None
        path = self._base / _safe_filename(stage)
        if not path.exists():
            return None
        return schema.model_validate_json(path.read_text(encoding="utf-8"))

    # ──────────────────────────────────────────────────────────────
    # Sub-checkpoints for chapters in multi-deck mode
    # ──────────────────────────────────────────────────────────────

    def scoped(self, name: str) -> "Checkpoint":
        """Return a child Checkpoint rooted at `<base>/<slugified-name>/`."""
        child        = Checkpoint.__new__(Checkpoint)
        child._base  = self._base / slugify(name)
        child._resume = self._resume
        return child

    def save_output_path(self, output_path: Path) -> None:
        """Store the final output path so `serve` can locate it without knowing the title."""
        self._base.mkdir(parents=True, exist_ok=True)
        (self._base / "output_path.txt").write_text(str(output_path), encoding="utf-8")

    def load_output_path(self) -> Path | None:
        """Return the stored output path if it still exists on disk, else None."""
        path = self._base / "output_path.txt"
        if not path.exists():
            return None
        candidate = Path(path.read_text(encoding="utf-8").strip())
        return candidate if candidate.exists() else None

    # ──────────────────────────────────────────────────────────────
    # Cache-key computation
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_key(file_path: Path, model: str, chunk_size: int) -> str:
        """
        Return a 16-character hex prefix of a SHA-256 digest that covers:
          - PDF filename (catches a renamed file at the same path)
          - PDF size in bytes
          - PDF mtime (catches in-place overwrites)
          - model string (model upgrade → fresh run)
          - chunk_size (config change → fresh run)

        Uses hashlib.sha256 — deterministic across processes, unlike hash().
        """
        stat = file_path.stat()
        h = hashlib.sha256()
        h.update(file_path.name.encode())
        h.update(str(int(stat.st_size)).encode())
        h.update(f"{stat.st_mtime:.6f}".encode())
        h.update(model.encode())
        h.update(str(chunk_size).encode())
        return h.hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _safe_filename(stage: str) -> str:
    """Convert a stage name to a safe .json filename."""
    return f"{slugify(stage)}.json"


def resolve_output_path(
    pdf_path: str,
    model: str,
    chunk_size: int,
    base_dir: Path | None = None,
) -> Path | None:
    """
    Return the output path for a previously generated PDF, or None.

    Uses the same hash as Checkpoint.compute_key to locate the checkpoint,
    then reads the stored output_path.txt written by save_output_path().
    Called by `cli.py serve` to find an existing output without re-running
    the pipeline.
    """
    if base_dir is None:
        base_dir = Path(".checkpoints").resolve()
    try:
        run_key = Checkpoint.compute_key(Path(pdf_path).resolve(), model, chunk_size)
    except OSError:
        return None
    ck = Checkpoint(base_dir=base_dir, run_key=run_key, resume=False)
    return ck.load_output_path()

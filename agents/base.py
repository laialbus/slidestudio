from pathlib import Path

from pydantic import BaseModel

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def format_source_chunks(
    chunks: list[str],
    slide_chunks: dict[int, list[int]],
    slide_indices: list[int] | None = None,
) -> str:
    """
    Render the raw source chunks backing a set of slides so a reviewer can
    judge depth against the source, not just internal coherence.

    Deduplicates chunks shared across slides and prefixes a slide→chunk map
    so the reviewer knows which chunks ground each slide. When no chunks are
    available (e.g. the caller did not thread them through), returns a clear
    sentinel rather than an empty string so the prompt stays well-formed.
    """
    if slide_indices is None:
        slide_indices = sorted(slide_chunks)

    mapping = {i: slide_chunks.get(i, []) for i in slide_indices}
    needed = sorted({
        c for ids in mapping.values() for c in ids if 0 <= c < len(chunks)
    })
    if not needed:
        return "(no source chunks available)"

    map_line = "; ".join(
        f"slide {i} → chunks {mapping[i]}" for i in slide_indices if mapping[i]
    )
    blocks = "\n\n---\n\n".join(f"[Chunk {c}]\n{chunks[c]}" for c in needed)
    return f"Slide-to-chunk map: {map_line}\n\n{blocks}"


class BaseAgent:
    name: str = "base"
    output_schema: type[BaseModel] | None = None

    def __init__(self, provider):
        self.provider       = provider
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        path = _PROMPTS_DIR / f"{self.name}.txt"
        return path.read_text() if path.exists() else ""

    def _load_named_prompt(self, prompt_name: str) -> str:
        return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text()

    async def _call(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        return await self.provider.complete_json(
            prompt=prompt,
            schema=schema,
            system=system,
        )

    async def run(self, **context) -> BaseModel:
        raise NotImplementedError

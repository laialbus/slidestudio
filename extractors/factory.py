"""
Extractor selection.

make_extractor maps a name to a BaseExtractor implementation, mirroring the
provider registry in cli.py.  This module never imports config.py: the caller
(cli.py / server.py / pipeline.py) reads PIPELINE["extractor"] and passes the
name and tunables in, keeping extractors free of config coupling.
"""

from extractors.base import BaseExtractor
from extractors.mineru import MineruExtractor
from extractors.pdf import PDFExtractor

# name → implementation.  Importing MineruExtractor is cheap: the heavy MinerU
# package is only invoked (lazily, via its CLI) inside extract(), never at import.
_EXTRACTOR_REGISTRY: dict[str, type[BaseExtractor]] = {
    "pymupdf4llm": PDFExtractor,
    "mineru":      MineruExtractor,
}


def make_extractor(
    name: str, chunk_size: int, overlap_size: int
) -> BaseExtractor:
    """Build the extractor selected by name.

    Args:
        name: registry key (e.g. "pymupdf4llm").
        chunk_size: target chunk size in characters.
        overlap_size: sliding-window overlap between sub-chunks.

    Returns:
        A ready-to-use BaseExtractor instance.

    Raises:
        ValueError: if name is not a registered extractor.
    """
    try:
        cls = _EXTRACTOR_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(_EXTRACTOR_REGISTRY))
        raise ValueError(
            f"Unknown extractor {name!r}. Available extractors: {available}."
        ) from e
    return cls(chunk_size=chunk_size, overlap_size=overlap_size)

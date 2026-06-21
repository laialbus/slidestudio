"""
Extraction abstraction.

BaseExtractor is the single contract every extractor implements, mirroring
BaseProvider and BaseExporter: the pipeline depends only on this interface and
never knows which library produced the result.  The output models
(ExtractionResult and its parts) live here so they are shared by every
implementation without importing any concrete extractor.

These models stay in extractors/ and are never imported by schemas/, agents/,
or providers/ (per CLAUDE.md).
"""

from pydantic import BaseModel


class TocItem(BaseModel):
    level:   int
    heading: str
    page:    int


class ImageRecord(BaseModel):
    index:    int
    caption:  str
    data_uri: str
    page:     int


class ExtractionResult(BaseModel):
    markdown:     str
    toc_items:    list[TocItem]
    chunks:       list[str]
    chunk_images: list[list[int]]   # chunk_idx → figure indices in that chunk
    images:       list[ImageRecord]
    page_count:   int
    char_count:   int
    ocr_used:     bool
    pdf_title:    str = ""          # largest-font text on page 1; empty if undetermined


class BaseExtractor:
    """
    Contract for all extractors: extract(path) -> ExtractionResult.

    Concrete implementations (pdf.py, mineru.py) subclass this.  chunk_size and
    overlap_size are stored here so every extractor chunks consistently; the
    caller supplies them.

    Attributes:
        name: short identifier for the implementation (e.g. "pymupdf4llm").
    """

    def __init__(self, chunk_size: int, overlap_size: int):
        self._chunk_size   = chunk_size
        self._overlap_size = overlap_size

    def extract(self, file_path: str) -> ExtractionResult:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

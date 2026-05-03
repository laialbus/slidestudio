from typing import Literal

from schemas.document_map import DocumentMap


class ChunkMap(DocumentMap):
    # Keys are stringified figure IDs; values classify how the figure is
    # referenced in this chunk's source text.
    figure_purposes: dict[str, Literal["conceptual", "evidential"]] = {}

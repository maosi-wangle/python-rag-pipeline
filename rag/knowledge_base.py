from __future__ import annotations

import json
from pathlib import Path

from .schemas import ChunkRecord


class ChunkStore:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.chunks = self._load_chunks()
        self.by_chunk_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.by_legacy_index = {
            chunk.legacy_index: chunk for chunk in self.chunks if chunk.legacy_index is not None
        }

    def _load_chunks(self) -> list[ChunkRecord]:
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, list) or not payload:
            raise ValueError("Knowledge base must be a non-empty list.")

        return [ChunkRecord.from_dict(item, index) for index, item in enumerate(payload)]

    def __len__(self) -> int:
        return len(self.chunks)

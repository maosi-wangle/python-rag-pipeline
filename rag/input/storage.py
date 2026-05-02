from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from .schemas import InputArtifact


class ArtifactRepository:
    def __init__(self, path: str | Path = "data/platform/artifacts.json") -> None:
        self.path = Path(path)

    def all(self) -> list[InputArtifact]:
        return [InputArtifact.from_dict(item) for item in self._read_items()]

    def get(self, artifact_id: str) -> InputArtifact | None:
        for item in self._read_items():
            if str(item.get("artifact_id")) == artifact_id:
                return InputArtifact.from_dict(item)
        return None

    def get_many(self, artifact_ids: list[str]) -> list[InputArtifact]:
        wanted = {str(item) for item in artifact_ids}
        artifacts: list[InputArtifact] = []
        for item in self._read_items():
            if str(item.get("artifact_id")) in wanted:
                artifacts.append(InputArtifact.from_dict(item))
        return artifacts

    def upsert(self, artifact: InputArtifact) -> None:
        payload = artifact.to_dict()
        items = self._read_items()
        for index, existing in enumerate(items):
            if str(existing.get("artifact_id")) == artifact.artifact_id:
                items[index] = payload
                break
        else:
            items.append(payload)
        self._write_items(items)

    def _read_items(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("items", [])
        if not isinstance(payload, list):
            raise ValueError(f"{self.path} must contain a JSON list.")
        return [dict(item) for item in payload]

    def _write_items(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class LocalArtifactStorage:
    def __init__(self, root: str | Path = "data/platform/artifacts") -> None:
        self.root = Path(root)

    def store_file(
        self,
        source_path: str | Path,
        *,
        owner_user_id: str,
        conversation_id: str | None = None,
        artifact_type: str | None = None,
        parser_id: str = "basic_multimodal",
    ) -> InputArtifact:
        source = Path(source_path).expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Input file not found: {source}")

        artifact_id = f"artifact_{uuid4().hex[:12]}"
        extension = source.suffix.lower()
        target_name = f"{artifact_id}{extension}"
        target = self.root / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

        mime_type, _ = mimetypes.guess_type(str(source))
        return InputArtifact(
            artifact_id=artifact_id,
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            type=artifact_type or _artifact_type_from_extension(extension, mime_type),
            filename=source.name,
            mime_type=mime_type,
            extension=extension,
            size_bytes=target.stat().st_size,
            sha256=_sha256(target),
            storage_path=str(target),
            parser_id=parser_id,
            parser_status="stored",
            parsed={},
            metadata={
                "original_path": str(source),
                "stored_filename": target_name,
            },
            created_at=_now(),
        )

    @staticmethod
    def to_parsed_artifact(
        source_path: str | Path,
        *,
        owner_user_id: str = "default",
        conversation_id: str | None = None,
        artifact_type: str | None = None,
        parser_id: str = "basic_multimodal",
    ) -> InputArtifact:
        source = Path(source_path).expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Input file not found: {source}")
        mime_type, _ = mimetypes.guess_type(str(source))
        return InputArtifact(
            artifact_id=f"transient_{uuid4().hex[:12]}",
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            type=artifact_type or _artifact_type_from_extension(source.suffix.lower(), mime_type),
            filename=source.name,
            mime_type=mime_type,
            extension=source.suffix.lower(),
            size_bytes=source.stat().st_size,
            sha256=_sha256(source),
            storage_path=str(source),
            parser_id=parser_id,
            parser_status="stored",
            parsed={},
            metadata={"original_path": str(source)},
            created_at=_now(),
        )

    @staticmethod
    def mark_parsed(
        artifact: InputArtifact,
        *,
        parsed: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        status: str = "parsed",
    ) -> InputArtifact:
        merged_metadata = dict(artifact.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return replace(
            artifact,
            parser_status=status,
            parsed=dict(parsed),
            metadata=merged_metadata,
        )


def _artifact_type_from_extension(extension: str, mime_type: str | None) -> str:
    ext = extension.lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        return "image"
    if ext in {".txt", ".md", ".markdown", ".html", ".htm"}:
        return "text"
    if mime_type and mime_type.startswith("image/"):
        return "image"
    return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

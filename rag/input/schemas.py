from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class InputArtifact:
    artifact_id: str
    owner_user_id: str
    type: str
    filename: str
    storage_path: str
    conversation_id: str | None = None
    mime_type: str | None = None
    extension: str = ""
    size_bytes: int = 0
    sha256: str = ""
    parser_id: str = "none"
    parser_status: str = "stored"
    parsed: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InputArtifact":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            owner_user_id=str(payload.get("owner_user_id") or "default"),
            conversation_id=payload.get("conversation_id"),
            type=str(payload.get("type") or "unknown"),
            filename=str(payload.get("filename") or ""),
            mime_type=payload.get("mime_type"),
            extension=str(payload.get("extension") or ""),
            size_bytes=int(payload.get("size_bytes") or 0),
            sha256=str(payload.get("sha256") or ""),
            storage_path=str(payload.get("storage_path") or ""),
            parser_id=str(payload.get("parser_id") or "none"),
            parser_status=str(payload.get("parser_status") or "stored"),
            parsed=_dict(payload.get("parsed")),
            metadata=_dict(payload.get("metadata")),
            created_at=payload.get("created_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "parser_status": self.parser_status,
            "parsed": self.parsed,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ParsedMessage:
    text: str
    modalities: list[str] = field(default_factory=list)
    artifacts: list[InputArtifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParsedMessage":
        return cls(
            text=str(payload.get("text") or ""),
            modalities=_list_of_str(payload.get("modalities")),
            artifacts=[
                InputArtifact.from_dict(item)
                for item in payload.get("artifacts", [])
                if isinstance(item, dict)
            ],
            warnings=_list_of_str(payload.get("warnings")),
            metadata=_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return payload

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "modalities": list(self.modalities),
            "artifacts": [artifact.to_prompt_dict() for artifact in self.artifacts],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

from __future__ import annotations

from html import unescape
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image

from .schemas import InputArtifact, ParsedMessage
from .storage import LocalArtifactStorage


class MultimodalInputParser:
    def __init__(self) -> None:
        self.storage = LocalArtifactStorage()
        self._ocr_reader: Any | None = None

    def parse_file(
        self,
        file_path: str | Path,
        *,
        owner_user_id: str = "default",
        conversation_id: str | None = None,
        persist: bool = False,
    ) -> ParsedMessage:
        artifact = (
            self.storage.store_file(
                file_path,
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
            )
            if persist
            else self.storage.to_parsed_artifact(
                file_path,
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
            )
        )
        return self.parse_artifact(artifact)

    def parse_artifact(self, artifact: InputArtifact) -> ParsedMessage:
        suffix = artifact.extension.lower()
        source = Path(artifact.storage_path)
        warnings: list[str] = []

        if suffix in {".txt"}:
            parsed = self._parse_text(source)
            modalities = ["text"]
        elif suffix in {".md", ".markdown"}:
            parsed = self._parse_markdown(source)
            modalities = ["text", "markdown"]
        elif suffix in {".html", ".htm"}:
            parsed = self._parse_html(source)
            modalities = ["text", "html"]
        elif suffix in {".png", ".jpg", ".jpeg"}:
            parsed = self._parse_image(source)
            modalities = ["image", "text"]
        else:
            parsed = {
                "text": "",
                "source_type": "unsupported",
                "warnings": [f"Unsupported input format: {suffix or artifact.filename}"],
            }
            modalities = ["unknown"]
            warnings.extend(parsed["warnings"])

        warnings.extend([str(item) for item in parsed.get("warnings", []) if str(item)])
        parsed_artifact = self.storage.mark_parsed(
            artifact,
            parsed=parsed,
            metadata={
                "modalities": modalities,
            },
            status="parsed" if not warnings else "parsed_with_warnings",
        )
        return ParsedMessage(
            text=str(parsed.get("text") or ""),
            modalities=modalities,
            artifacts=[parsed_artifact],
            warnings=warnings,
            metadata={
                "input_kind": parsed.get("source_type"),
            },
        )

    @staticmethod
    def build_query_context(query: str, message: ParsedMessage) -> str:
        return build_query_with_artifacts(query, message.artifacts)

    @staticmethod
    def _parse_text(path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return {
            "text": text,
            "source_type": "text",
            "warnings": [],
        }

    @staticmethod
    def _parse_markdown(path: Path) -> dict[str, Any]:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        plain = _collapse_blank_lines(_strip_markdown(raw))
        return {
            "text": plain.strip(),
            "source_type": "markdown",
            "raw_markdown": raw,
            "warnings": [],
        }

    @staticmethod
    def _parse_html(path: Path) -> dict[str, Any]:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return {
            "text": _collapse_blank_lines(unescape(text)).strip(),
            "source_type": "html",
            "title": soup.title.string.strip() if soup.title and soup.title.string else "",
            "warnings": [],
        }

    def _parse_image(self, path: Path) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        warnings: list[str] = []
        text_parts: list[str] = []
        with Image.open(path) as image:
            width, height = image.size
        try:
            reader = self._get_ocr_reader()
            raw_blocks = reader.readtext(str(path))
            for box, text, score in raw_blocks:
                line = str(text).strip()
                if not line:
                    continue
                text_parts.append(line)
                blocks.append(
                    {
                        "text": line,
                        "score": float(score),
                        "box": [[float(x), float(y)] for x, y in box],
                    }
                )
        except Exception as exc:
            warnings.append(f"Image OCR failed: {exc}")

        text = "\n".join(text_parts).strip()
        return {
            "text": text,
            "source_type": "image",
            "ocr_blocks": blocks,
            "image_kind": _classify_image(blocks, width, height),
            "contains_dense_text": len(blocks) >= 10,
            "contains_table": _looks_like_table(blocks),
            "warnings": warnings,
            "width": width,
            "height": height,
            "ocr_engine": "easyocr",
        }

    def _get_ocr_reader(self) -> Any:
        if self._ocr_reader is None:
            import easyocr

            self._ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        return self._ocr_reader


def build_query_with_artifacts(query: str, artifacts: list[InputArtifact]) -> str:
    blocks = [artifact_prompt_block(artifact) for artifact in artifacts]
    block_text = "\n\n".join(blocks) if blocks else "无"
    return (
        f"用户问题:\n{query.strip()}\n\n"
        f"附件上下文:\n{block_text}"
    )


def artifact_prompt_block(artifact: InputArtifact, *, max_chars: int = 4000) -> str:
    parsed = artifact.parsed or {}
    parsed_text = str(parsed.get("text") or "").strip()
    if len(parsed_text) > max_chars:
        parsed_text = parsed_text[:max_chars].rstrip() + "\n...(truncated)"
    warnings = parsed.get("warnings") or []
    warning_text = "\n".join(f"- {str(item)}" for item in warnings) if warnings else "- 无"
    metadata_modalities = artifact.metadata.get("modalities") if artifact.metadata else None
    if isinstance(metadata_modalities, list) and metadata_modalities:
        modalities = ", ".join(str(item) for item in metadata_modalities)
    else:
        modalities = artifact.type
    return (
        f"[附件]\n"
        f"- artifact_id: {artifact.artifact_id}\n"
        f"- 文件名: {artifact.filename}\n"
        f"- 类型: {artifact.type}\n"
        f"- 模态: {modalities}\n"
        f"- parser_status: {artifact.parser_status}\n"
        f"- 解析文本:\n{parsed_text or '(empty)'}\n"
        f"- warnings:\n{warning_text}"
    )


def _strip_markdown(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.M)
    cleaned = cleaned.replace("|", " ")
    cleaned = cleaned.replace("---", " ")
    cleaned = cleaned.replace("***", " ")
    return cleaned


def _collapse_blank_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _classify_image(blocks: list[dict[str, Any]], width: int, height: int) -> str:
    if not blocks:
        return "general_image"
    if _looks_like_table(blocks):
        return "table_like_image"
    density = len(blocks) / max(width * height, 1)
    if len(blocks) >= 12 or density > 0.00002:
        return "dense_text_image"
    if len(blocks) >= 4:
        return "mixed_document_image"
    return "general_image"


def _looks_like_table(blocks: list[dict[str, Any]]) -> bool:
    if len(blocks) < 4:
        return False
    ys = []
    for block in blocks:
        box = block.get("box") or []
        if not box:
            continue
        avg_y = sum(float(point[1]) for point in box) / len(box)
        ys.append(round(avg_y / 20))
    if not ys:
        return False
    rows = {}
    for bucket in ys:
        rows[bucket] = rows.get(bucket, 0) + 1
    return sum(1 for count in rows.values() if count >= 2) >= 2

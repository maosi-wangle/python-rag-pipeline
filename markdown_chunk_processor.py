from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jieba
import jieba.analyse
import markdown
from bs4 import BeautifulSoup, Tag


class MarkdownKnowledgeBuilder:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.split_pattern = re.compile(r"[\n。！？；;]")
        jieba.initialize()

    def build(
        self,
        *,
        file_path: str,
        document_id: str,
        title: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        raw_text = Path(file_path).read_text(encoding="utf-8")
        html = markdown.markdown(raw_text)
        soup = BeautifulSoup(html, "html.parser")

        title = title or self._infer_title(soup, Path(file_path).stem)
        source = source or file_path

        current_section = title
        current_sentences: list[str] = []
        chunks: list[dict] = []
        chunk_index = 0

        for element in soup.descendants:
            if not isinstance(element, Tag):
                continue
            if element.name in {"h1", "h2", "h3", "h4"}:
                current_section = element.get_text(" ", strip=True) or current_section
                continue
            if element.name not in {"p", "li"}:
                continue

            text = element.get_text(" ", strip=True)
            if not text:
                continue

            for sentence in self._split_sentences(text):
                candidate = "\n".join(current_sentences + [sentence])
                if current_sentences and len(candidate) > self.chunk_size:
                    chunks.append(
                        self._make_chunk(
                            document_id=document_id,
                            title=title,
                            section=current_section,
                            source=source,
                            chunk_index=chunk_index,
                            text="\n".join(current_sentences),
                        )
                    )
                    chunk_index += 1
                    current_sentences = []
                current_sentences.append(sentence)

        if current_sentences:
            chunks.append(
                self._make_chunk(
                    document_id=document_id,
                    title=title,
                    section=current_section,
                    source=source,
                    chunk_index=chunk_index,
                    text="\n".join(current_sentences),
                )
            )

        return chunks

    def _infer_title(self, soup: BeautifulSoup, fallback: str) -> str:
        heading = soup.find(["h1", "h2"])
        if heading:
            return heading.get_text(" ", strip=True)
        return fallback

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [part.strip() for part in self.split_pattern.split(text) if part.strip()]
        return sentences or [text.strip()]

    def _make_chunk(
        self,
        *,
        document_id: str,
        title: str,
        section: str,
        source: str,
        chunk_index: int,
        text: str,
    ) -> dict:
        keywords = jieba.analyse.extract_tags(
            text,
            topK=10,
            allowPOS=("n", "vn", "eng"),
        )
        return {
            "chunk_id": f"{document_id}_chunk_{chunk_index:04d}",
            "document_id": document_id,
            "title": title,
            "section": section,
            "context": text,
            "keywords": keywords,
            "source": source,
            "metadata": {
                "builder": "markdown_chunk_processor",
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a future-ready knowledge base JSON from a markdown document."
    )
    parser.add_argument("input", help="Path to the markdown file.")
    parser.add_argument(
        "--output",
        default="knowledgeBase.new.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--document-id",
        required=True,
        help="Stable document id used for chunk ids.",
    )
    parser.add_argument("--title", help="Optional document title override.")
    parser.add_argument("--source", help="Optional source override.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Approximate max characters per chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = MarkdownKnowledgeBuilder(chunk_size=args.chunk_size)
    chunks = builder.build(
        file_path=args.input,
        document_id=args.document_id,
        title=args.title,
        source=args.source,
    )
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(chunks)} chunks -> {output_path}")


if __name__ == "__main__":
    main()

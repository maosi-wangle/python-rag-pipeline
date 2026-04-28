from __future__ import annotations

import re
from collections import Counter

import jieba


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
_STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "及",
    "与",
    "吗",
    "呢",
    "啊",
    "如何",
    "怎么",
    "什么",
    "哪些",
    "原因",
    "作用",
    "方法",
    "注意事项",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def regex_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text or "")]


def tokenize(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    try:
        raw_tokens = [token.strip().lower() for token in jieba.cut(normalized)]
    except Exception:
        raw_tokens = regex_tokens(normalized)

    tokens = []
    for token in raw_tokens:
        if not token:
            continue
        if token in _STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        cleaned = normalize_whitespace(item)
        if not cleaned or cleaned in seen:
            continue
        unique_items.append(cleaned)
        seen.add(cleaned)
    return unique_items


def lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0

    text_counter = Counter(text_tokens)
    overlap = sum(min(text_counter[token], 1) for token in set(query_tokens))
    return overlap / max(len(set(query_tokens)), 1)


def has_multi_intent_markers(query: str) -> bool:
    markers = ("以及", "并且", "同时", "分别", "对比", "区别", "优缺点", "步骤", "、")
    return any(marker in query for marker in markers)


def extract_recent_terms(history: list[str], limit: int = 4) -> list[str]:
    terms: list[str] = []
    for item in reversed(history):
        terms.extend(tokenize(item))
        if len(terms) >= limit * 2:
            break
    return unique_preserve_order(terms)[:limit]


def truncate_text(text: str, max_chars: int = 240) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."

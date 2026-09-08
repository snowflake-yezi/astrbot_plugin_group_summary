from __future__ import annotations

import json
import re
from typing import Any

SPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(value: Any, limit: int) -> str:
    text = str(value or "")
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text[:limit]


def clean_string_list(value: Any, limit: int, item_limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        cleaned = clean_text(item, item_limit)
        if cleaned and cleaned not in result:
            result.append(cleaned)
        if len(result) >= limit:
            break
    return tuple(result)


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型响应中没有 JSON 对象") from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("模型响应不是合法 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("模型响应的 JSON 顶层不是对象")
    return payload

"""
nanoevals._parser
~~~~~~~~~~~~~~~~~
Robust LLM output → score/comment extractor.

Handles the many ways LLMs format their responses:
  - ```json { "score": 0.8, "comment": "..." } ```
  - { "score": true, "reasoning": "..." }
  - Score: 0.7\\nReasoning: ...
  - bare "true" / "false" / "0.8"

Raises ParseError explicitly rather than silently returning 0.0.
"""

from __future__ import annotations

import json
import re
from typing import Any


class ParseError(ValueError):
    """Raised when the LLM response cannot be parsed into a score."""


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.DOTALL)
_SCORE_LINE_RE = re.compile(
    r'(?:score|result|rating|value)\s*[=:]\s*([0-9]*\.?[0-9]+|true|false)',
    re.IGNORECASE,
)
# synonym mapping for comment field
_COMMENT_ALIASES = ("comment", "reason", "reasoning", "explanation", "justification", "rationale")
_SCORE_ALIASES = ("score", "result", "rating", "value", "grade")


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON dict from *text*, tolerating markdown fences."""
    # 1. strip code fences
    fence_match = _CODE_FENCE_RE.search(text)
    candidate = fence_match.group(1).strip() if fence_match else text.strip()

    # 2. find the first `{…}` block
    start = candidate.find("{")
    end = candidate.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start:end])
        except json.JSONDecodeError:
            pass

    # 3. try the whole candidate directly
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _coerce_score(raw: Any) -> float | bool:
    """Coerce a raw JSON value to float (0-1) or bool."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        lower = raw.strip().lower()
        if lower in ("true", "yes", "pass", "1"):
            return True
        if lower in ("false", "no", "fail", "0"):
            return False
        try:
            return float(lower)
        except ValueError:
            pass
    raise ParseError(f"Cannot coerce {raw!r} to a score.")


def _extract_comment(data: dict[str, Any]) -> str:
    for alias in _COMMENT_ALIASES:
        if alias in data:
            return str(data[alias])
    return ""


def _extract_score(data: dict[str, Any]) -> float | bool:
    for alias in _SCORE_ALIASES:
        if alias in data:
            return _coerce_score(data[alias])
    raise ParseError(f"No score field found in dict: {list(data.keys())}")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def parse_llm_output(
    response: str,
    *,
    continuous: bool = False,
    choices: list[float] | None = None,
) -> tuple[float | bool, str]:
    """Parse *response* into ``(score, comment)``.

    Args:
        response:   Raw string from the LLM.
        continuous: If True, convert bool scores to 0.0 / 1.0.
        choices:    If provided, snap the score to the nearest value.

    Returns:
        A ``(score, comment)`` tuple.

    Raises:
        ParseError: If *response* cannot be parsed.
    """
    data = _extract_json(response)

    if data is not None and isinstance(data, dict):
        score = _extract_score(data)
        comment = _extract_comment(data)
    else:
        # Fallback: scan for a score line
        match = _SCORE_LINE_RE.search(response)
        if match:
            score = _coerce_score(match.group(1))
            comment = ""
        else:
            # Last resort: is the whole response a bare bool / float?
            stripped = response.strip().lower()
            if stripped in ("true", "false"):
                score = stripped == "true"
                comment = ""
            else:
                raise ParseError(
                    f"Cannot parse LLM output as a score. Response was:\n{response!r}"
                )

    # Post-process
    if continuous and isinstance(score, bool):
        score = 1.0 if score else 0.0

    if choices is not None:
        numeric = float(score)
        score = min(choices, key=lambda c: abs(c - numeric))

    return score, comment

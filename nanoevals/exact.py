"""
nanoevals.exact
~~~~~~~~~~~~~~~
Exact-match evaluator factory.

No LLM required.  Useful for:
  - Validating tool-call argument values
  - Checking structured output fields
  - Regression tests with known-good reference outputs

Usage::

    from nanoevals.exact import create_exact_match

    exact = create_exact_match()
    result = exact(outputs="Paris", reference_outputs="paris")
    # → EvalResult(key='exact_match', score=True, comment='Exact match (normalized).')
"""

from __future__ import annotations

import re
from typing import Any, Callable

from nanoevals.types import EvalResult


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def create_exact_match(
    *,
    key: str = "exact_match",
    normalize: bool = True,
    case_sensitive: bool = False,
) -> Callable[..., EvalResult]:
    """Return an evaluator that checks for exact string equality.

    Args:
        key:            The ``EvalResult.key`` value.
        normalize:      Strip whitespace and collapse internal spaces.
        case_sensitive: If True, comparison is case-sensitive.

    Returns:
        ``evaluator(outputs, reference_outputs, **_) -> EvalResult``
    """

    def evaluator(
        outputs: Any,
        reference_outputs: Any,
        **_: Any,
    ) -> EvalResult:
        a = str(outputs)
        b = str(reference_outputs)

        if normalize:
            a = _normalize(a)
            b = _normalize(b)
        elif not case_sensitive:
            a = a.lower()
            b = b.lower()

        matched = a == b
        suffix = " (normalized)" if normalize else ""
        comment = (
            f"Exact match{suffix}." if matched else f"No match{suffix}: {a!r} ≠ {b!r}"
        )

        return EvalResult(key=key, score=matched, comment=comment)

    evaluator.required_keys: set[str] = {"outputs", "reference_outputs"}  # type: ignore[attr-defined]
    evaluator.feedback_key = key  # type: ignore[attr-defined]

    return evaluator

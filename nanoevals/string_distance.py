"""
nanoevals.string_distance
~~~~~~~~~~~~~~~~~~~~~~~~~
Levenshtein (edit-distance) similarity evaluator factory.

Zero dependencies — implements edit distance inline.

Usage::

    from nanoevals.string_distance import create_levenshtein

    lev = create_levenshtein(threshold=0.8)
    result = lev(outputs="Retrieval Augmented Generation",
                 reference_outputs="Retrieval-Augmented Generation")
    # → EvalResult(key='levenshtein', score=True, comment='Similarity: 0.97 ≥ 0.80')
"""

from __future__ import annotations

from typing import Any, Callable

from nanoevals.types import EvalResult


# ------------------------------------------------------------------
# Pure-Python Levenshtein distance
# ------------------------------------------------------------------

def _levenshtein_distance(a: str, b: str) -> int:
    """Wagner-Fischer algorithm — O(|a| * |b|) time, O(min(|a|,|b|)) space."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def _normalized_similarity(a: str, b: str) -> float:
    """Return 1.0 − normalised edit distance."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    dist = _levenshtein_distance(a, b)
    return 1.0 - dist / max_len


# ------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------

def create_levenshtein(
    *,
    key: str = "levenshtein",
    threshold: float = 0.8,
    normalize: bool = True,
) -> Callable[..., EvalResult]:
    """Return an evaluator that scores string similarity via edit distance.

    Args:
        key:       The ``EvalResult.key`` value.
        threshold: Similarity threshold (0–1) above which the result is a pass.
        normalize: Lowercase + strip before comparison.

    Returns:
        ``evaluator(outputs, reference_outputs, **_) -> EvalResult``
        ``EvalResult.score`` is the raw float similarity (0–1).
        ``EvalResult.passed(threshold)`` returns True/False.
    """

    def evaluator(
        outputs: Any,
        reference_outputs: Any,
        **_: Any,
    ) -> EvalResult:
        a = str(outputs)
        b = str(reference_outputs)

        if normalize:
            a = a.strip().lower()
            b = b.strip().lower()

        similarity = _normalized_similarity(a, b)
        passed = similarity >= threshold
        comment = (
            f"Similarity: {similarity:.2f} {'≥' if passed else '<'} {threshold:.2f}"
        )

        return EvalResult(
            key=key,
            score=round(similarity, 4),
            comment=comment,
            metadata={"threshold": threshold, "passed": passed},
        )

    evaluator.required_keys: set[str] = {"outputs", "reference_outputs"}  # type: ignore[attr-defined]
    evaluator.feedback_key = key  # type: ignore[attr-defined]

    return evaluator

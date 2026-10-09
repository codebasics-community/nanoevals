"""
nanoevals.dataset
~~~~~~~~~~~~~~~~~
Dataset validation utilities.

Validation is driven by metadata attached to each evaluator callable
(``evaluator.required_keys``) rather than a hardcoded registry,
so new evaluators never require touching this file.
"""

from __future__ import annotations

from typing import Any, Callable


def validate(
    dataset: list[dict[str, Any]],
    evaluators: list[Callable],
) -> None:
    """Raise ``ValueError`` if any dataset row is missing required keys.

    Required keys are read from ``evaluator.required_keys`` (a ``set[str]``).
    Evaluators without this attribute are skipped.
    """
    required: set[str] = set()
    for ev in evaluators:
        required.update(getattr(ev, "required_keys", set()))

    if not required:
        return

    for i, row in enumerate(dataset):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(
                f"Dataset row {i} is missing required keys: {sorted(missing)}. "
                f"Row keys present: {sorted(row.keys())}"
            )

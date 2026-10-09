"""
nanoevals.runner
~~~~~~~~~~~~~~~~
Batch evaluation runner with async concurrency.

Usage::

    from nanoevals.runner import evaluate, async_evaluate

    results = evaluate(
        dataset=[{"inputs": "...", "outputs": "..."}],
        evaluators=[faithfulness, correctness],
        concurrency=4,
    )
    print(results.summary())
    # → {'faithfulness': 0.92, 'correctness': 0.87}
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from nanoevals.dataset import validate
from nanoevals.types import BatchResult, EvalResult, SampleResult


# ------------------------------------------------------------------
# Internal: call a single evaluator on a single row
# ------------------------------------------------------------------

async def _call_one(
    evaluator: Callable,
    row: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> EvalResult:
    """Call *evaluator* respecting the semaphore for concurrency control."""
    async with semaphore:
        async_call = getattr(evaluator, "async_call", None)
        if async_call is not None:
            # LLM-as-judge or trajectory judge — native async
            return await async_call(**row)
        else:
            # Deterministic evaluator (exact, levenshtein, trajectory match)
            # Run in a thread pool so we don't block the event loop.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: evaluator(**row))


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def async_evaluate(
    dataset: list[dict[str, Any]],
    evaluators: list[Callable],
    *,
    concurrency: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Run all *evaluators* over *dataset* concurrently.

    Args:
        dataset:     List of dicts, one per sample. Keys are passed as kwargs
                     to each evaluator.
        evaluators:  List of evaluator callables (from the factory functions).
        concurrency: Maximum number of parallel evaluator calls. Each LLM call
                     counts as one unit of concurrency.
        on_progress: Optional callback ``(completed: int, total: int) -> None``
                     called after each sample is fully evaluated.

    Returns:
        A :class:`BatchResult` containing all results.
    """
    validate(dataset, evaluators)

    semaphore = asyncio.Semaphore(concurrency)
    batch = BatchResult()
    total = len(dataset)

    for i, row in enumerate(dataset):
        # Evaluate all evaluators for this row concurrently
        tasks = [_call_one(ev, row, semaphore) for ev in evaluators]
        results: list[EvalResult] = await asyncio.gather(*tasks)

        batch.samples.append(SampleResult(sample_index=i, results=list(results)))

        if on_progress:
            on_progress(i + 1, total)

    return batch


def evaluate(
    dataset: list[dict[str, Any]],
    evaluators: list[Callable],
    *,
    concurrency: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Synchronous wrapper around :func:`async_evaluate`.

    Runs the async event loop for you.  If you are already inside an async
    context (e.g. Jupyter), use ``await async_evaluate(...)`` directly.

    Args:
        dataset:     List of dicts, one per sample.
        evaluators:  List of evaluator callables.
        concurrency: Max parallel evaluator calls (default 4).
        on_progress: Optional progress callback.

    Returns:
        A :class:`BatchResult`.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "evaluate() cannot be called from within a running event loop. "
            "Use 'await async_evaluate(...)' instead."
        )

    return asyncio.run(
        async_evaluate(
            dataset,
            evaluators,
            concurrency=concurrency,
            on_progress=on_progress,
        )
    )

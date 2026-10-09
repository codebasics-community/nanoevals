"""
nanoevals.llm
~~~~~~~~~~~~~
LLM-as-judge evaluator factory.

Usage::

    from nanoevals.llm import create_llm_as_judge
    from nanoevals.prompts import FAITHFULNESS_PROMPT

    faithfulness = create_llm_as_judge(
        prompt=FAITHFULNESS_PROMPT,
        llm=my_llm_callable,
        feedback_key="faithfulness",
        continuous=True,
    )

    result = faithfulness(
        inputs="What is RAG?",
        outputs="RAG means Retrieval Augmented Generation.",
        contexts="RAG retrieves documents before generation.",
    )
    # → EvalResult(key='faithfulness', score=0.92, comment='...')

The *llm* callable may be:
  - Callable[[str], str]            — raw string in, raw string out (simple, current nanoagents style)
  - Callable[[list[dict]], str]     — messages list in, raw string out (supports system prompts)
  - An async version of either
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from nanoevals._parser import ParseError, parse_llm_output
from nanoevals.types import EvalResult

# ------------------------------------------------------------------
# Internal: LLM adapters
# ------------------------------------------------------------------

LLMCallable = Callable[[str | list[dict]], str]


async def _call_llm_async(
    llm: LLMCallable,
    messages: list[dict[str, str]],
) -> str:
    """Call *llm* with the right signature, ensuring an async result."""
    # Determine if user passed a raw-string callable or messages callable
    # by inspecting the first param annotation (best-effort heuristic).
    # We always try messages first; if that raises TypeError we fall back.
    user_message = messages[-1]["content"]

    if asyncio.iscoroutinefunction(llm):
        try:
            return await llm(messages)
        except TypeError:
            return await llm(user_message)
    else:
        try:
            raw = llm(messages)
        except TypeError:
            raw = llm(user_message)
        # If llm returned a coroutine (e.g. a sync wrapper around async), await it.
        if inspect.isawaitable(raw):
            return await raw
        return raw


# ------------------------------------------------------------------
# Internal: prompt formatting
# ------------------------------------------------------------------

def _format_prompt(
    template: str,
    kwargs: dict[str, Any],
    few_shot_examples: list[dict] | None,
) -> str:
    """Format *template* with *kwargs*, appending few-shot examples."""
    # Build optional sections that some prompts expose
    ref = kwargs.get("reference_outputs")
    ctx = kwargs.get("contexts")

    extras = dict(kwargs)
    # Provide helper placeholders so prompts can optionally include them
    extras.setdefault(
        "reference_section",
        f"\n<reference_outputs>\n{ref}\n</reference_outputs>\n" if ref else "",
    )
    extras.setdefault(
        "context_section",
        f"\n<context>\n{ctx}\n</context>\n" if ctx else "",
    )
    # Normalise contexts for RAG prompts that use {contexts} directly
    if ctx and isinstance(ctx, list):
        extras["contexts"] = "\n\n".join(ctx)

    try:
        prompt = template.format_map(extras)
    except KeyError as exc:
        raise ValueError(
            f"Prompt requires variable {exc} but it was not passed to the evaluator."
        ) from exc

    if few_shot_examples:
        examples_text = "\n\n<examples>\n"
        for ex in few_shot_examples:
            examples_text += (
                f"Input: {ex.get('inputs', '')}\n"
                f"Output: {ex.get('outputs', '')}\n"
                f"Reasoning: {ex.get('reasoning', '')}\n"
                f"Score: {ex.get('score', '')}\n\n"
            )
        examples_text += "</examples>"
        prompt += examples_text

    return prompt


def _build_messages(
    prompt: str,
    system: str | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


# ------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------

def create_llm_as_judge(
    prompt: str,
    llm: LLMCallable,
    *,
    feedback_key: str = "score",
    continuous: bool = False,
    choices: list[float] | None = None,
    use_reasoning: bool = True,
    few_shot_examples: list[dict] | None = None,
    system: str | None = None,
) -> Callable[..., EvalResult]:
    """Return an evaluator function that uses *llm* as a judge.

    Args:
        prompt:            An f-string prompt template. Variable names are
                           passed as kwargs when calling the evaluator.
        llm:               Your LLM callable (sync or async).
        feedback_key:      The ``EvalResult.key`` value. Defaults to ``"score"``.
        continuous:        If True, coerce bool scores to 0.0 / 1.0.
        choices:           Snap the score to the nearest value in this list.
        use_reasoning:     If False, the comment field is discarded.
        few_shot_examples: Examples appended to the prompt.
        system:            Optional system message for the judge model.

    Returns:
        A callable ``evaluator(**kwargs) -> EvalResult``.
        The callable is also marked with ``.required_keys`` (a set of str)
        so the runner can validate datasets before calling.
    """

    async def _async_evaluator(**kwargs: Any) -> EvalResult:
        formatted = _format_prompt(prompt, kwargs, few_shot_examples)
        messages = _build_messages(formatted, system)

        try:
            response = await _call_llm_async(llm, messages)
        except Exception as exc:
            return EvalResult(
                key=feedback_key,
                score=False,
                comment=f"LLM call failed: {exc}",
                metadata={"error": str(exc)},
            )

        try:
            score, comment = parse_llm_output(
                response, continuous=continuous, choices=choices
            )
        except ParseError as exc:
            return EvalResult(
                key=feedback_key,
                score=False,
                comment=f"Parse error: {exc}",
                metadata={"raw_response": response},
            )

        return EvalResult(
            key=feedback_key,
            score=score,
            comment=comment if use_reasoning else "",
            metadata={"raw_response": response},
        )

    def evaluator(**kwargs: Any) -> EvalResult:
        """Sync wrapper — runs the async evaluator in a new event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Inside an existing event loop (e.g. Jupyter): schedule as task.
            # Callers should use async_evaluate() in that case.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _async_evaluator(**kwargs))
                return future.result()
        else:
            return asyncio.run(_async_evaluator(**kwargs))

    # Expose async variant as an attribute for the runner
    evaluator.async_call = _async_evaluator  # type: ignore[attr-defined]
    evaluator.feedback_key = feedback_key  # type: ignore[attr-defined]
    evaluator.required_keys: set[str] = set()  # type: ignore[attr-defined]

    return evaluator

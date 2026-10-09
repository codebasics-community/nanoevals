"""
nanoevals.agent.trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluators for ReACT agent trajectories, compatible with nanoagents.

A trajectory is a list of steps. Each step has at minimum:
  - action: str      — tool name invoked
  - input:  dict     — arguments passed to the tool
  - thought: str     — agent's reasoning (optional for matching)
  - obs:    str      — tool observation / output (optional for matching)

This module provides two evaluators:

1. ``create_trajectory_match`` — deterministic, no LLM.
   Compares actual vs expected tool-call sequences.

2. ``create_trajectory_judge`` — LLM-as-judge.
   Assesses overall trajectory quality using a language model.

Usage::

    from nanoevals.agent.trajectory import create_trajectory_match

    # Expected steps (minimal dicts or nanoagents Step objects)
    expected = [
        {"action": "add",      "input": {"a": 5,      "b": 3}},
        {"action": "multiply", "input": {"a": 8,      "b": 4}},
    ]

    actual = agent.history   # list[Step] from nanoagents

    match_eval = create_trajectory_match(match_mode="strict")
    result = match_eval(outputs=actual, reference_outputs=expected)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

from nanoevals._parser import ParseError, parse_llm_output
from nanoevals.llm import LLMCallable, _call_llm_async, _build_messages, _format_prompt
from nanoevals.prompts.agent import TRAJECTORY_QUALITY_PROMPT
from nanoevals.types import EvalResult

import asyncio


# ------------------------------------------------------------------
# Step protocol — works with nanoagents.Step or plain dicts
# ------------------------------------------------------------------

def _get(step: Any, key: str, default: Any = None) -> Any:
    """Attribute or dict access, gracefully."""
    if isinstance(step, dict):
        return step.get(key, default)
    return getattr(step, key, default)


def _step_action(step: Any) -> str:
    return str(_get(step, "action") or "")


def _step_input(step: Any) -> dict:
    val = _get(step, "input") or _get(step, "action_input") or {}
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": val}
    return dict(val)


def _format_trajectory(steps: list[Any]) -> str:
    """Render a step list as human-readable text for LLM prompts."""
    lines = []
    for i, step in enumerate(steps, 1):
        thought = _get(step, "thought", "")
        action = _step_action(step)
        inp = _step_input(step)
        obs = _get(step, "obs") or _get(step, "observation", "")
        lines.append(
            f"Step {i}:\n"
            f"  Thought: {thought}\n"
            f"  Action: {action}\n"
            f"  Input: {json.dumps(inp)}\n"
            f"  Observation: {obs}"
        )
    return "\n\n".join(lines)


# ------------------------------------------------------------------
# Tool-args comparison modes
# ------------------------------------------------------------------

def _args_match(
    actual_args: dict,
    expected_args: dict,
    mode: Literal["exact", "ignore", "partial"],
) -> bool:
    if mode == "ignore":
        return True
    if mode == "exact":
        return actual_args == expected_args
    # partial: expected keys must be present with matching values
    return all(actual_args.get(k) == v for k, v in expected_args.items())


# ------------------------------------------------------------------
# 1. Deterministic trajectory match
# ------------------------------------------------------------------

def create_trajectory_match(
    *,
    key: str = "trajectory_match",
    match_mode: Literal["strict", "unordered", "subset"] = "strict",
    tool_args_mode: Literal["exact", "ignore", "partial"] = "exact",
) -> Callable[..., EvalResult]:
    """Return a deterministic trajectory evaluator.

    Args:
        key:           The ``EvalResult.key`` value.
        match_mode:
            ``"strict"``    — actual steps must match expected steps in order.
            ``"unordered"`` — same tool calls, any order.
            ``"subset"``    — every expected step must appear in actual (superset allowed).
        tool_args_mode:
            ``"exact"``   — action input dicts must match exactly.
            ``"ignore"``  — only tool name is compared.
            ``"partial"`` — expected keys must be present in actual with correct values.

    Returns:
        ``evaluator(outputs, reference_outputs, **_) -> EvalResult``
        where *outputs* and *reference_outputs* are lists of steps.
    """

    def _match_step(actual: Any, expected: Any) -> bool:
        if _step_action(actual) != _step_action(expected):
            return False
        return _args_match(_step_input(actual), _step_input(expected), tool_args_mode)

    def evaluator(
        outputs: list[Any],
        reference_outputs: list[Any],
        **_: Any,
    ) -> EvalResult:
        actual = list(outputs)
        expected = list(reference_outputs)

        if match_mode == "strict":
            if len(actual) != len(expected):
                return EvalResult(
                    key=key,
                    score=False,
                    comment=(
                        f"Length mismatch: got {len(actual)} steps, "
                        f"expected {len(expected)}."
                    ),
                )
            mismatches = []
            for i, (a, e) in enumerate(zip(actual, expected)):
                if not _match_step(a, e):
                    mismatches.append(
                        f"Step {i + 1}: expected action={_step_action(e)!r} "
                        f"input={_step_input(e)}, got action={_step_action(a)!r} "
                        f"input={_step_input(a)}"
                    )
            if mismatches:
                return EvalResult(
                    key=key, score=False, comment="; ".join(mismatches)
                )
            return EvalResult(
                key=key,
                score=True,
                comment=f"All {len(expected)} steps matched (strict).",
            )

        if match_mode == "unordered":
            remaining = list(actual)
            unmatched = []
            for exp in expected:
                found = False
                for idx, act in enumerate(remaining):
                    if _match_step(act, exp):
                        remaining.pop(idx)
                        found = True
                        break
                if not found:
                    unmatched.append(
                        f"action={_step_action(exp)!r} input={_step_input(exp)}"
                    )
            if unmatched:
                return EvalResult(
                    key=key,
                    score=False,
                    comment=f"Missing steps: {'; '.join(unmatched)}",
                )
            return EvalResult(
                key=key,
                score=True,
                comment=f"All {len(expected)} expected steps found (unordered).",
            )

        if match_mode == "subset":
            # Every expected step must appear somewhere in actual.
            missing = []
            actual_actions = [_step_action(a) for a in actual]
            for exp in expected:
                exp_action = _step_action(exp)
                matched = any(
                    _match_step(a, exp) for a in actual
                )
                if not matched:
                    missing.append(f"action={exp_action!r}")
            if missing:
                return EvalResult(
                    key=key,
                    score=False,
                    comment=f"Expected steps not found in trajectory: {', '.join(missing)}",
                )
            return EvalResult(
                key=key,
                score=True,
                comment=(
                    f"All {len(expected)} expected steps are a subset of "
                    f"the {len(actual)}-step trajectory."
                ),
            )

        raise ValueError(f"Unknown match_mode: {match_mode!r}")

    evaluator.required_keys: set[str] = {"outputs", "reference_outputs"}  # type: ignore[attr-defined]
    evaluator.feedback_key = key  # type: ignore[attr-defined]

    return evaluator


# ------------------------------------------------------------------
# 2. LLM-as-judge trajectory evaluator
# ------------------------------------------------------------------

def create_trajectory_judge(
    llm: LLMCallable,
    *,
    prompt: str = TRAJECTORY_QUALITY_PROMPT,
    feedback_key: str = "trajectory_quality",
    continuous: bool = True,
    system: str | None = None,
) -> Callable[..., EvalResult]:
    """Return an LLM-as-judge evaluator for overall trajectory quality.

    Args:
        llm:          Your LLM callable (sync or async).
        prompt:       Prompt template. Receives ``{inputs}``, ``{trajectory}``,
                      ``{outputs}``, and optionally ``{reference_outputs}``.
        feedback_key: The ``EvalResult.key`` value.
        continuous:   If True, return float score instead of bool.
        system:       Optional system message.

    The evaluator expects:
        - ``inputs``            — the original task string
        - ``outputs``           — the agent's trajectory (list[Step])
        - ``final_answer``      — optional final answer string
        - ``reference_outputs`` — optional expected answer
    """

    async def _async_evaluator(**kwargs: Any) -> EvalResult:
        steps: list[Any] = kwargs.pop("outputs", [])
        task: str = str(kwargs.get("inputs", ""))
        final_answer: str = str(kwargs.get("final_answer", ""))
        ref: str = str(kwargs.get("reference_outputs", ""))

        trajectory_text = _format_trajectory(steps)

        prompt_kwargs = {
            "inputs": task,
            "outputs": final_answer,
            "trajectory": trajectory_text,
            "reference_outputs": ref,
            **kwargs,
        }
        formatted = _format_prompt(prompt, prompt_kwargs, None)
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
            score, comment = parse_llm_output(response, continuous=continuous)
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
            comment=comment,
            metadata={"raw_response": response},
        )

    def evaluator(**kwargs: Any) -> EvalResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _async_evaluator(**kwargs))
                return future.result()
        return asyncio.run(_async_evaluator(**kwargs))

    evaluator.async_call = _async_evaluator  # type: ignore[attr-defined]
    evaluator.feedback_key = feedback_key  # type: ignore[attr-defined]
    evaluator.required_keys: set[str] = {"inputs", "outputs"}  # type: ignore[attr-defined]

    return evaluator

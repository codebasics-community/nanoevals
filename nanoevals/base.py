"""
nanoevals.base
~~~~~~~~~~~~~~
Abstract base classes for building custom evaluators.

Users extend these classes to create reusable, composable evaluators
that work identically to factory-produced callables in evaluate() and
async_evaluate().

Three entry points:

BaseEvaluator
    Synchronous custom logic — implement score().

AsyncBaseEvaluator
    Asynchronous custom logic — implement score() as async.

LLMEvaluator  (extends AsyncBaseEvaluator)
    LLM-based evaluators — implement build_prompt() only.
    The base class handles calling the LLM, parsing the output,
    and error handling automatically.

TrajectoryEvaluator  (extends BaseEvaluator)
    nanoagents trajectory evaluators — implement evaluate_steps().
    The base class normalises the 'outputs' kwarg into a step list.

All classes are fully compatible with evaluate() and async_evaluate()
without any changes to the runner.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from nanoevals._parser import ParseError, parse_llm_output
from nanoevals.llm import LLMCallable, _build_messages, _call_llm_async
from nanoevals.types import EvalResult


# ─────────────────────────────────────────────────────────────────────────────
# 1. BaseEvaluator — synchronous
# ─────────────────────────────────────────────────────────────────────────────

class BaseEvaluator(ABC):
    """Abstract base for **synchronous** custom evaluators.

    Subclass this when your scoring logic is pure Python (no I/O, no LLM).
    The runner automatically offloads ``__call__`` to a thread pool, so
    it never blocks the async event loop.

    Class-level attributes (override in your subclass):

    .. code-block:: python

        class MyEvaluator(BaseEvaluator):
            feedback_key = "my_metric"
            required_keys = {"outputs", "reference_outputs"}

            def score(self, outputs: str, reference_outputs: str, **_) -> EvalResult:
                ...

    Then use it anywhere a factory evaluator is accepted::

        results = evaluate(dataset, evaluators=[MyEvaluator()])
    """

    feedback_key: ClassVar[str] = "score"
    """The key used in :class:`~nanoevals.types.EvalResult`. Override in subclasses."""

    required_keys: ClassVar[set[str]] = set()
    """Dataset keys this evaluator requires. Used by :func:`~nanoevals.dataset.validate`."""

    @abstractmethod
    def score(self, **kwargs: Any) -> EvalResult:
        """Compute and return an :class:`~nanoevals.types.EvalResult`.

        All dataset row keys are forwarded as keyword arguments.
        Always accept ``**_`` to safely absorb unknown fields.
        """
        ...

    def __call__(self, **kwargs: Any) -> EvalResult:
        """Delegates to :meth:`score`. Do not override."""
        return self.score(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AsyncBaseEvaluator — async custom logic
# ─────────────────────────────────────────────────────────────────────────────

class AsyncBaseEvaluator(ABC):
    """Abstract base for **asynchronous** custom evaluators.

    Subclass this when your scoring logic involves async I/O
    (external APIs, databases, async HTTP clients, etc.).

    The runner calls ``async_call`` directly on the event loop — no
    thread pool overhead.

    Example::

        class MyAsyncEvaluator(AsyncBaseEvaluator):
            feedback_key = "my_async_metric"
            required_keys = {"outputs"}

            async def score(self, outputs: str, **_) -> EvalResult:
                result = await some_async_api(outputs)
                return EvalResult(key=self.feedback_key, score=result.ok)
    """

    feedback_key: ClassVar[str] = "score"
    required_keys: ClassVar[set[str]] = set()

    @abstractmethod
    async def score(self, **kwargs: Any) -> EvalResult:
        """Async scoring logic. Implement this in your subclass."""
        ...

    async def async_call(self, **kwargs: Any) -> EvalResult:
        """Called by the runner on the async path. Delegates to :meth:`score`."""
        return await self.score(**kwargs)

    def __call__(self, **kwargs: Any) -> EvalResult:
        """Sync wrapper for standalone use (e.g. scripts, tests)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.score(**kwargs))
                return future.result()
        return asyncio.run(self.score(**kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLMEvaluator — implement build_prompt(), everything else is handled
# ─────────────────────────────────────────────────────────────────────────────

class LLMEvaluator(AsyncBaseEvaluator, ABC):
    """Abstract base for **LLM-as-judge** custom evaluators.

    Subclass this and implement only :meth:`build_prompt`. The base class
    handles the full pipeline:

    1. Call :meth:`build_prompt` with the dataset row kwargs.
    2. Send the prompt to the LLM (sync or async callable, auto-detected).
    3. Parse the response via the robust ``_parser`` module.
    4. Return a typed :class:`~nanoevals.types.EvalResult`.

    LLM call failures and parse errors are captured into the result's
    ``comment`` field rather than raising — the runner always receives a
    valid :class:`~nanoevals.types.EvalResult`.

    Example::

        MY_PROMPT = \"\"\"
        Evaluate the SQL query for correctness.
        <schema>{schema}</schema>
        <query>{outputs}</query>
        Respond with JSON: {{"score": <true|false>, "comment": "..."}}
        \"\"\"

        class SQLCorrectnessEvaluator(LLMEvaluator):
            feedback_key = "sql_correctness"
            required_keys = {"outputs", "schema"}

            def build_prompt(self, outputs: str, schema: str, **_) -> str:
                return MY_PROMPT.format(outputs=outputs, schema=schema)

        ev = SQLCorrectnessEvaluator(
            llm=my_llm,
            continuous=False,
        )
        result = ev(outputs="SELECT * FROM users", schema="users(id, name)")

    To customise the full LLM pipeline (e.g. multi-turn, extra parsing),
    override :meth:`score` instead of :meth:`build_prompt`.
    """

    feedback_key: ClassVar[str] = "score"
    required_keys: ClassVar[set[str]] = set()

    def __init__(
        self,
        llm: LLMCallable,
        *,
        continuous: bool = False,
        choices: list[float] | None = None,
        use_reasoning: bool = True,
        system: str | None = None,
        few_shot_examples: list[dict] | None = None,
    ) -> None:
        """
        Args:
            llm:                Your LLM callable.  Accepts both:
                                - ``Callable[[str], str]`` (raw-string contract)
                                - ``Callable[[list[dict]], str]`` (messages contract)
                                Sync and async variants are both supported.
            continuous:         If ``True``, coerce boolean scores to ``0.0`` / ``1.0``.
            choices:            Snap the score to the nearest value in this list.
            use_reasoning:      If ``False``, the comment field is discarded.
            system:             Optional system prompt prepended to every call.
            few_shot_examples:  List of example dicts appended to the prompt.
                                Each dict should have keys: ``inputs``, ``outputs``,
                                ``reasoning``, ``score``.
        """
        self.llm = llm
        self.continuous = continuous
        self.choices = choices
        self.use_reasoning = use_reasoning
        self.system = system
        self.few_shot_examples = few_shot_examples or []

    @abstractmethod
    def build_prompt(self, **kwargs: Any) -> str:
        """Return the formatted prompt string for this evaluation.

        All dataset row keys are forwarded as keyword arguments.
        Raise ``ValueError`` for missing required fields rather than
        silently producing a broken prompt.
        """
        ...

    def _append_few_shot(self, prompt: str) -> str:
        """Append few-shot examples to *prompt* if any were configured."""
        if not self.few_shot_examples:
            return prompt
        block = "\n\n<examples>\n"
        for ex in self.few_shot_examples:
            block += (
                f"Input: {ex.get('inputs', '')}\n"
                f"Output: {ex.get('outputs', '')}\n"
                f"Reasoning: {ex.get('reasoning', '')}\n"
                f"Score: {ex.get('score', '')}\n\n"
            )
        return prompt + block + "</examples>"

    async def score(self, **kwargs: Any) -> EvalResult:
        """Full LLM pipeline — override only if you need extra control."""
        try:
            raw_prompt = self.build_prompt(**kwargs)
        except Exception as exc:
            return EvalResult(
                key=self.feedback_key,
                score=False,
                comment=f"build_prompt() failed: {exc}",
                metadata={"error": str(exc)},
            )

        prompt = self._append_few_shot(raw_prompt)
        messages = _build_messages(prompt, self.system)

        try:
            response = await _call_llm_async(self.llm, messages)
        except Exception as exc:
            return EvalResult(
                key=self.feedback_key,
                score=False,
                comment=f"LLM call failed: {exc}",
                metadata={"error": str(exc)},
            )

        try:
            result_score, comment = parse_llm_output(
                response,
                continuous=self.continuous,
                choices=self.choices,
            )
        except ParseError as exc:
            return EvalResult(
                key=self.feedback_key,
                score=False,
                comment=f"Parse error: {exc}",
                metadata={"raw_response": response},
            )

        return EvalResult(
            key=self.feedback_key,
            score=result_score,
            comment=comment if self.use_reasoning else "",
            metadata={"raw_response": response},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TrajectoryEvaluator — nanoagents trajectory helpers
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryEvaluator(BaseEvaluator, ABC):
    """Abstract base for **nanoagents trajectory** evaluators.

    Implement :meth:`evaluate_steps` and receive a normalised list of
    step-like objects. Works with both nanoagents ``Step`` objects and
    plain dicts (any object with ``action``, ``input``, ``thought``,
    ``obs`` attributes or keys).

    Example::

        class MaxStepsEvaluator(TrajectoryEvaluator):
            feedback_key = "within_step_budget"
            required_keys = {"outputs"}

            def __init__(self, max_steps: int = 5):
                self.max_steps = max_steps

            def evaluate_steps(self, steps: list, **_) -> EvalResult:
                n = len(steps)
                passed = n <= self.max_steps
                return EvalResult(
                    key=self.feedback_key,
                    score=passed,
                    comment=f"{n} steps used (budget: {self.max_steps})",
                )

        ev = MaxStepsEvaluator(max_steps=3)
        results = evaluate(dataset, evaluators=[ev])
    """

    required_keys: ClassVar[set[str]] = {"outputs"}

    @abstractmethod
    def evaluate_steps(self, steps: list[Any], **kwargs: Any) -> EvalResult:
        """Evaluate the *steps* trajectory.

        Args:
            steps:    Normalised list of step-like objects.  Each element
                      has ``action``, ``input``, ``thought``, and ``obs``
                      accessible via attribute or dict key.
            **kwargs: Remaining dataset row fields (e.g. ``inputs``, ``reference_outputs``).
        """
        ...

    def score(self, outputs: list[Any] | Any = None, **kwargs: Any) -> EvalResult:
        """Normalise *outputs* into a list and delegate to :meth:`evaluate_steps`."""
        if outputs is None:
            steps: list[Any] = []
        elif isinstance(outputs, list):
            steps = outputs
        else:
            # Single step passed directly
            steps = [outputs]
        return self.evaluate_steps(steps, **kwargs)

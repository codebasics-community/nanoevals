"""
tests/test_nanoevals.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for nanoevals.

Unit tests (no API key needed):
  - _parser.py
  - create_exact_match
  - create_levenshtein
  - create_trajectory_match (all modes)
  - BatchResult aggregation
  - dataset.validate()

Integration tests (require GROQ_API_KEY, marked with @pytest.mark.integration):
  - create_llm_as_judge with real LLM
  - create_trajectory_judge with real LLM
  - evaluate() batch runner end-to-end

Run unit tests only:
    pytest tests/ -v -m "not integration"

Run all (with API key):
    GROQ_API_KEY=... pytest tests/ -v
"""

from __future__ import annotations

import os
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_step(action: str, input: dict, thought: str = "", obs: str = "") -> dict:
    """Construct a plain-dict step (compatible with trajectory evaluators)."""
    return {"action": action, "input": input, "thought": thought, "obs": obs}


def _groq_llm(prompt: str) -> str:
    """Real LLM callable for integration tests (raw-string contract)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set — skipping integration test")
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ──────────────────────────────────────────────────────────────────────────────
# 1. Parser tests
# ──────────────────────────────────────────────────────────────────────────────

class TestParser:
    from nanoevals._parser import parse_llm_output, ParseError

    def test_json_score_true(self):
        from nanoevals._parser import parse_llm_output
        score, comment = parse_llm_output('{"score": true, "comment": "looks good"}')
        assert score is True
        assert comment == "looks good"

    def test_json_score_float(self):
        from nanoevals._parser import parse_llm_output
        score, comment = parse_llm_output('{"score": 0.85, "reason": "mostly correct"}')
        assert score == 0.85
        assert comment == "mostly correct"

    def test_json_inside_code_fence(self):
        from nanoevals._parser import parse_llm_output
        response = '```json\n{"score": false, "comment": "wrong"}\n```'
        score, comment = parse_llm_output(response)
        assert score is False
        assert comment == "wrong"

    def test_json_various_comment_aliases(self):
        from nanoevals._parser import parse_llm_output
        for alias in ("reason", "reasoning", "explanation", "justification"):
            score, _ = parse_llm_output(f'{{"score": 1, "{alias}": "test"}}')
            assert score == 1.0

    def test_bare_true_false(self):
        from nanoevals._parser import parse_llm_output
        score, _ = parse_llm_output("true")
        assert score is True
        score, _ = parse_llm_output("false")
        assert score is False

    def test_score_line_fallback(self):
        from nanoevals._parser import parse_llm_output
        score, _ = parse_llm_output("The answer is mostly correct.\nScore: 0.7\nMore text.")
        assert score == 0.7

    def test_continuous_converts_bool(self):
        from nanoevals._parser import parse_llm_output
        score, _ = parse_llm_output('{"score": true}', continuous=True)
        assert score == 1.0
        score, _ = parse_llm_output('{"score": false}', continuous=True)
        assert score == 0.0

    def test_choices_snaps_score(self):
        from nanoevals._parser import parse_llm_output
        score, _ = parse_llm_output('{"score": 0.6}', choices=[0.0, 0.5, 1.0])
        assert score == 0.5

    def test_parse_error_on_garbage(self):
        from nanoevals._parser import ParseError, parse_llm_output
        with pytest.raises(ParseError):
            parse_llm_output("This is just plain text with no score whatsoever.")

    def test_extra_whitespace_in_json(self):
        from nanoevals._parser import parse_llm_output
        score, comment = parse_llm_output('  \n  {"score": 0.9, "comment": "ok"}  \n  ')
        assert score == 0.9


# ──────────────────────────────────────────────────────────────────────────────
# 2. Exact match tests
# ──────────────────────────────────────────────────────────────────────────────

class TestExactMatch:
    def setup_method(self):
        from nanoevals.exact import create_exact_match
        self.exact = create_exact_match()

    def test_identical_strings(self):
        r = self.exact(outputs="Paris", reference_outputs="Paris")
        assert r.score is True

    def test_case_insensitive_by_default(self):
        r = self.exact(outputs="paris", reference_outputs="PARIS")
        assert r.score is True

    def test_whitespace_normalized(self):
        r = self.exact(outputs="  Paris  ", reference_outputs="Paris")
        assert r.score is True

    def test_mismatch(self):
        r = self.exact(outputs="London", reference_outputs="Paris")
        assert r.score is False
        assert "London" in r.comment or "london" in r.comment.lower()

    def test_custom_key(self):
        from nanoevals.exact import create_exact_match
        exact = create_exact_match(key="my_metric")
        r = exact(outputs="x", reference_outputs="x")
        assert r.key == "my_metric"

    def test_required_keys_set(self):
        assert "outputs" in self.exact.required_keys
        assert "reference_outputs" in self.exact.required_keys


# ──────────────────────────────────────────────────────────────────────────────
# 3. Levenshtein tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLevenshtein:
    def setup_method(self):
        from nanoevals.string_distance import create_levenshtein
        self.lev = create_levenshtein(threshold=0.8)

    def test_identical(self):
        r = self.lev(outputs="hello world", reference_outputs="hello world")
        assert r.score == 1.0
        assert r.passed(0.8) is True

    def test_close_match(self):
        r = self.lev(outputs="colour", reference_outputs="color")
        # Should be high similarity
        assert r.score > 0.7

    def test_completely_different(self):
        r = self.lev(outputs="apple", reference_outputs="xyz")
        assert r.score < 0.5

    def test_empty_strings(self):
        from nanoevals.string_distance import create_levenshtein
        lev = create_levenshtein()
        r = lev(outputs="", reference_outputs="")
        assert r.score == 1.0

    def test_custom_threshold_in_comment(self):
        r = self.lev(outputs="test", reference_outputs="test")
        assert "0.80" in r.comment or "1.00" in r.comment

    def test_score_is_float(self):
        r = self.lev(outputs="abc", reference_outputs="abx")
        assert isinstance(r.score, float)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Trajectory match tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectoryMatch:
    def setup_method(self):
        from nanoevals.agent.trajectory import create_trajectory_match
        self.strict = create_trajectory_match(match_mode="strict")
        self.unordered = create_trajectory_match(match_mode="unordered")
        self.subset = create_trajectory_match(match_mode="subset")

    # ── Strict mode ──────────────────────────────────────────────

    def test_strict_exact_match(self):
        steps = [
            _make_step("add", {"a": 1, "b": 2}),
            _make_step("multiply", {"a": 3, "b": 4}),
        ]
        r = self.strict(outputs=steps, reference_outputs=steps)
        assert r.score is True

    def test_strict_wrong_order(self):
        actual = [
            _make_step("multiply", {"a": 3, "b": 4}),
            _make_step("add", {"a": 1, "b": 2}),
        ]
        expected = [
            _make_step("add", {"a": 1, "b": 2}),
            _make_step("multiply", {"a": 3, "b": 4}),
        ]
        r = self.strict(outputs=actual, reference_outputs=expected)
        assert r.score is False

    def test_strict_length_mismatch(self):
        actual = [_make_step("add", {"a": 1, "b": 2})]
        expected = [
            _make_step("add", {"a": 1, "b": 2}),
            _make_step("multiply", {"a": 3, "b": 4}),
        ]
        r = self.strict(outputs=actual, reference_outputs=expected)
        assert r.score is False
        assert "Length mismatch" in r.comment

    def test_strict_wrong_args(self):
        actual = [_make_step("add", {"a": 1, "b": 99})]
        expected = [_make_step("add", {"a": 1, "b": 2})]
        r = self.strict(outputs=actual, reference_outputs=expected)
        assert r.score is False

    # ── Unordered mode ────────────────────────────────────────────

    def test_unordered_correct(self):
        actual = [
            _make_step("multiply", {"a": 3, "b": 4}),
            _make_step("add", {"a": 1, "b": 2}),
        ]
        expected = [
            _make_step("add", {"a": 1, "b": 2}),
            _make_step("multiply", {"a": 3, "b": 4}),
        ]
        r = self.unordered(outputs=actual, reference_outputs=expected)
        assert r.score is True

    def test_unordered_missing_tool(self):
        actual = [_make_step("add", {"a": 1, "b": 2})]
        expected = [
            _make_step("add", {"a": 1, "b": 2}),
            _make_step("multiply", {"a": 3, "b": 4}),
        ]
        r = self.unordered(outputs=actual, reference_outputs=expected)
        assert r.score is False
        assert "multiply" in r.comment

    # ── Subset mode ───────────────────────────────────────────────

    def test_subset_passes_when_superset(self):
        actual = [
            _make_step("add", {"a": 1, "b": 2}),
            _make_step("multiply", {"a": 3, "b": 4}),
            _make_step("divide", {"a": 10, "b": 2}),
        ]
        expected = [_make_step("add", {"a": 1, "b": 2})]
        r = self.subset(outputs=actual, reference_outputs=expected)
        assert r.score is True

    def test_subset_fails_when_missing(self):
        actual = [_make_step("add", {"a": 1, "b": 2})]
        expected = [
            _make_step("add", {"a": 1, "b": 2}),
            _make_step("search", {"query": "weather"}),
        ]
        r = self.subset(outputs=actual, reference_outputs=expected)
        assert r.score is False

    # ── tool_args_mode ────────────────────────────────────────────

    def test_args_ignore_mode(self):
        from nanoevals.agent.trajectory import create_trajectory_match
        ev = create_trajectory_match(match_mode="strict", tool_args_mode="ignore")
        actual = [_make_step("add", {"a": 99, "b": 99})]
        expected = [_make_step("add", {"a": 1, "b": 2})]
        r = ev(outputs=actual, reference_outputs=expected)
        assert r.score is True

    def test_args_partial_mode(self):
        from nanoevals.agent.trajectory import create_trajectory_match
        ev = create_trajectory_match(match_mode="strict", tool_args_mode="partial")
        actual = [_make_step("search", {"query": "weather", "limit": 5})]
        expected = [_make_step("search", {"query": "weather"})]
        r = ev(outputs=actual, reference_outputs=expected)
        assert r.score is True

    # ── Works with nanoagents-style objects (attrs) ───────────────

    def test_with_object_steps(self):
        class MockStep:
            def __init__(self, action, input):
                self.action = action
                self.input = input
                self.thought = ""
                self.obs = ""

        actual = [MockStep("add", {"a": 1, "b": 2})]
        expected = [MockStep("add", {"a": 1, "b": 2})]
        r = self.strict(outputs=actual, reference_outputs=expected)
        assert r.score is True


# ──────────────────────────────────────────────────────────────────────────────
# 5. Types / BatchResult tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchResult:
    def _make_batch(self):
        from nanoevals.types import BatchResult, EvalResult, SampleResult
        batch = BatchResult(samples=[
            SampleResult(sample_index=0, results=[
                EvalResult(key="faithfulness", score=0.9, comment="good"),
                EvalResult(key="relevance", score=True, comment="yes"),
            ]),
            SampleResult(sample_index=1, results=[
                EvalResult(key="faithfulness", score=0.7, comment="ok"),
                EvalResult(key="relevance", score=False, comment="no"),
            ]),
        ])
        return batch

    def test_summary_mean(self):
        batch = self._make_batch()
        summary = batch.summary()
        assert abs(summary["faithfulness"] - 0.8) < 1e-9
        assert abs(summary["relevance"] - 0.5) < 1e-9

    def test_pass_rates(self):
        batch = self._make_batch()
        rates = batch.pass_rates(threshold=0.8)
        assert rates["faithfulness"] == 0.5   # 0.9 passes, 0.7 fails
        assert rates["relevance"] == 0.5      # True passes, False fails

    def test_to_dict(self):
        batch = self._make_batch()
        rows = batch.to_dict()
        assert len(rows) == 2
        assert "faithfulness_score" in rows[0]
        assert rows[0]["faithfulness_score"] == 0.9

    def test_sample_get(self):
        from nanoevals.types import EvalResult, SampleResult
        sample = SampleResult(sample_index=0, results=[
            EvalResult(key="correctness", score=True)
        ])
        result = sample.get("correctness")
        assert result is not None
        assert result.score is True
        assert sample.get("nonexistent") is None

    def test_eval_result_passed(self):
        from nanoevals.types import EvalResult
        r = EvalResult(key="x", score=0.9)
        assert r.passed(threshold=0.8) is True
        assert r.passed(threshold=0.95) is False

        r_bool = EvalResult(key="y", score=True)
        assert r_bool.passed() is True


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dataset validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetValidation:
    def test_valid_dataset_passes(self):
        from nanoevals.dataset import validate
        from nanoevals.exact import create_exact_match
        exact = create_exact_match()
        # Should not raise
        validate(
            [{"outputs": "a", "reference_outputs": "b"}],
            [exact],
        )

    def test_missing_keys_raises(self):
        from nanoevals.dataset import validate
        from nanoevals.exact import create_exact_match
        exact = create_exact_match()
        with pytest.raises(ValueError, match="missing required keys"):
            validate(
                [{"outputs": "a"}],   # missing reference_outputs
                [exact],
            )

    def test_no_required_keys_skips_validation(self):
        from nanoevals.dataset import validate
        # A bare callable without required_keys should not block
        def dummy(**_):
            pass
        validate([{}], [dummy])  # no error

    def test_error_message_includes_row_index(self):
        from nanoevals.dataset import validate
        from nanoevals.exact import create_exact_match
        exact = create_exact_match()
        with pytest.raises(ValueError, match="row 1"):
            validate(
                [
                    {"outputs": "a", "reference_outputs": "b"},
                    {"outputs": "c"},              # row 1 missing
                ],
                [exact],
            )


# ──────────────────────────────────────────────────────────────────────────────
# 7. Runner tests (sync, no LLM — uses deterministic evaluators)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunner:
    def test_evaluate_returns_batch_result(self):
        from nanoevals import evaluate
        from nanoevals.exact import create_exact_match
        exact = create_exact_match()
        result = evaluate(
            dataset=[
                {"outputs": "Paris", "reference_outputs": "paris"},
                {"outputs": "London", "reference_outputs": "paris"},
            ],
            evaluators=[exact],
        )
        assert len(result.samples) == 2
        assert result.samples[0].results[0].score is True
        assert result.samples[1].results[0].score is False

    def test_evaluate_summary(self):
        from nanoevals import evaluate
        from nanoevals.exact import create_exact_match
        exact = create_exact_match()
        result = evaluate(
            dataset=[
                {"outputs": "yes", "reference_outputs": "yes"},
                {"outputs": "no", "reference_outputs": "yes"},
            ],
            evaluators=[exact],
        )
        summary = result.summary()
        assert abs(summary["exact_match"] - 0.5) < 1e-9

    def test_evaluate_multiple_evaluators(self):
        from nanoevals import evaluate
        from nanoevals.exact import create_exact_match
        from nanoevals.string_distance import create_levenshtein
        exact = create_exact_match()
        lev = create_levenshtein()
        result = evaluate(
            dataset=[{"outputs": "colour", "reference_outputs": "color"}],
            evaluators=[exact, lev],
        )
        assert len(result.samples[0].results) == 2

    def test_evaluate_progress_callback(self):
        from nanoevals import evaluate
        from nanoevals.exact import create_exact_match
        calls: list[tuple[int, int]] = []
        exact = create_exact_match()
        evaluate(
            dataset=[{"outputs": "a", "reference_outputs": "a"}] * 3,
            evaluators=[exact],
            on_progress=lambda done, total: calls.append((done, total)),
        )
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_trajectory_match_in_runner(self):
        from nanoevals import evaluate
        from nanoevals.agent.trajectory import create_trajectory_match
        ev = create_trajectory_match(match_mode="unordered")
        result = evaluate(
            dataset=[{
                "outputs": [_make_step("add", {"a": 1, "b": 2})],
                "reference_outputs": [_make_step("add", {"a": 1, "b": 2})],
            }],
            evaluators=[ev],
        )
        assert result.samples[0].results[0].score is True


# ──────────────────────────────────────────────────────────────────────────────
# 8. Integration tests (require GROQ_API_KEY)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestLLMAsJudgeIntegration:
    def test_faithfulness_high_score(self, nanoevals_report):
        from nanoevals.llm import create_llm_as_judge
        from nanoevals.prompts import FAITHFULNESS_PROMPT

        ev = create_llm_as_judge(
            prompt=FAITHFULNESS_PROMPT,
            llm=_groq_llm,
            feedback_key="faithfulness",
            continuous=True,
        )
        result = ev(
            inputs="What is RAG?",
            outputs="RAG stands for Retrieval Augmented Generation.",
            contexts=["RAG is a technique that combines retrieval with generation."],
        )
        nanoevals_report.add(result, test="test_faithfulness_high_score")
        assert isinstance(result.score, float)
        assert result.score >= 0.5
        assert result.comment

    def test_correctness_binary(self, nanoevals_report):
        from nanoevals.llm import create_llm_as_judge
        from nanoevals.prompts import CORRECTNESS_PROMPT

        ev = create_llm_as_judge(
            prompt=CORRECTNESS_PROMPT,
            llm=_groq_llm,
            feedback_key="correctness",
        )
        result = ev(
            inputs="What is the capital of France?",
            outputs="The capital of France is Paris.",
        )
        nanoevals_report.add(result, test="test_correctness_binary")
        assert result.score is True

    def test_correctness_wrong_answer(self, nanoevals_report):
        from nanoevals.llm import create_llm_as_judge
        from nanoevals.prompts import CORRECTNESS_PROMPT

        ev = create_llm_as_judge(
            prompt=CORRECTNESS_PROMPT,
            llm=_groq_llm,
            feedback_key="correctness",
        )
        result = ev(
            inputs="What is the capital of France?",
            outputs="The capital of France is Berlin.",
        )
        nanoevals_report.add(result, test="test_correctness_wrong_answer")
        assert result.score is False

    def test_context_recall_rag(self, nanoevals_report):
        from nanoevals.llm import create_llm_as_judge
        from nanoevals.prompts import CONTEXT_RECALL_PROMPT

        ev = create_llm_as_judge(
            prompt=CONTEXT_RECALL_PROMPT,
            llm=_groq_llm,
            feedback_key="context_recall",
            continuous=True,
        )
        result = ev(
            reference_outputs="RAG combines retrieval with generation.",
            contexts=["RAG is short for Retrieval Augmented Generation, "
                      "a technique that retrieves relevant documents and "
                      "uses them to ground generated text."],
        )
        nanoevals_report.add(result, test="test_context_recall_rag")
        assert result.score >= 0.5

    def test_few_shot_examples(self, nanoevals_report):
        from nanoevals.llm import create_llm_as_judge
        from nanoevals.prompts import CONCISENESS_PROMPT

        ev = create_llm_as_judge(
            prompt=CONCISENESS_PROMPT,
            llm=_groq_llm,
            feedback_key="conciseness",
            few_shot_examples=[
                {
                    "inputs": "What is 2+2?",
                    "outputs": "The answer is 4.",
                    "reasoning": "Direct and concise.",
                    "score": True,
                }
            ],
        )
        result = ev(
            inputs="What is 2+2?",
            outputs="4",
        )
        nanoevals_report.add(result, test="test_few_shot_examples")
        assert isinstance(result.score, bool)

    def test_batch_evaluate_rag_pipeline(self, nanoevals_report):
        from nanoevals import evaluate
        from nanoevals.llm import create_llm_as_judge
        from nanoevals.prompts import FAITHFULNESS_PROMPT, ANSWER_RELEVANCE_PROMPT

        faithfulness = create_llm_as_judge(
            prompt=FAITHFULNESS_PROMPT,
            llm=_groq_llm,
            feedback_key="faithfulness",
            continuous=True,
        )
        relevance = create_llm_as_judge(
            prompt=ANSWER_RELEVANCE_PROMPT,
            llm=_groq_llm,
            feedback_key="relevance",
        )

        dataset = [
            {
                "inputs": "What is RAG?",
                "outputs": "RAG means Retrieval Augmented Generation.",
                "contexts": ["RAG retrieves relevant documents before generating."],
            },
            {
                "inputs": "Who invented Python?",
                "outputs": "Python was created by Guido van Rossum.",
                "contexts": ["Guido van Rossum created Python in the late 1980s."],
            },
        ]

        result = evaluate(dataset, evaluators=[faithfulness, relevance], concurrency=2)

        for sample in result.samples:
            for r in sample.results:
                nanoevals_report.add(r, test=f"batch_sample_{sample.sample_index}")

        summary = result.summary()
        assert "faithfulness" in summary
        assert "relevance" in summary
        assert len(result.samples) == 2


@pytest.mark.integration
class TestTrajectoryJudgeIntegration:
    def test_trajectory_judge_scores_good_trajectory(self, nanoevals_report):
        from nanoevals.agent.trajectory import create_trajectory_judge

        judge = create_trajectory_judge(llm=_groq_llm, continuous=True)
        steps = [
            _make_step("add", {"a": 5184, "b": 3348432},
                       thought="I need to add the two numbers first.",
                       obs="3353616"),
            _make_step("multiply", {"a": 3353616, "b": 29},
                       thought="Now multiply the result by 29.",
                       obs="97254864"),
        ]
        result = judge(
            inputs="Calculate (5184 + 3348432) * 29",
            outputs=steps,
            final_answer="97254864",
            reference_outputs="97254864",
        )
        nanoevals_report.add(result, test="test_trajectory_judge_good")
        assert isinstance(result.score, float)
        assert result.score >= 0.5
        assert result.comment


# ──────────────────────────────────────────────────────────────────────────────
# 9. Custom evaluator extensibility (base classes)
# ──────────────────────────────────────────────────────────────────────────────

class TestCustomEvaluators:
    """Tests for BaseEvaluator, AsyncBaseEvaluator, LLMEvaluator, TrajectoryEvaluator."""

    # ── Concrete test subclasses ──────────────────────────────────

    def _make_word_count_evaluator(self, max_words: int = 10):
        """Returns a BaseEvaluator subclass that checks output word count."""
        from nanoevals.base import BaseEvaluator
        from nanoevals.types import EvalResult

        class WordCountEvaluator(BaseEvaluator):
            feedback_key = "word_count_ok"
            required_keys = {"outputs"}

            def __init__(self, max_w: int):
                self.max_w = max_w

            def score(self, outputs: str, **_) -> EvalResult:
                count = len(outputs.split())
                passed = count <= self.max_w
                return EvalResult(
                    key=self.feedback_key,
                    score=passed,
                    comment=f"{count} words (max {self.max_w})",
                )

        return WordCountEvaluator(max_words)

    def _make_async_uppercase_evaluator(self):
        """Returns an AsyncBaseEvaluator that checks outputs are uppercase."""
        from nanoevals.base import AsyncBaseEvaluator
        from nanoevals.types import EvalResult

        class UppercaseEvaluator(AsyncBaseEvaluator):
            feedback_key = "is_uppercase"
            required_keys = {"outputs"}

            async def score(self, outputs: str, **_) -> EvalResult:
                passed = outputs == outputs.upper()
                return EvalResult(
                    key=self.feedback_key,
                    score=passed,
                    comment="All caps." if passed else "Not uppercase.",
                )

        return UppercaseEvaluator()

    def _make_step_budget_evaluator(self, max_steps: int = 3):
        """Returns a TrajectoryEvaluator that checks step count budget."""
        from nanoevals.base import TrajectoryEvaluator
        from nanoevals.types import EvalResult

        class StepBudgetEvaluator(TrajectoryEvaluator):
            feedback_key = "within_budget"

            def __init__(self, budget: int):
                self.budget = budget

            def evaluate_steps(self, steps: list, **_) -> EvalResult:
                n = len(steps)
                passed = n <= self.budget
                return EvalResult(
                    key=self.feedback_key,
                    score=passed,
                    comment=f"{n} steps used (budget: {self.budget})",
                )

        return StepBudgetEvaluator(max_steps)

    # ── BaseEvaluator tests ───────────────────────────────────────

    def test_base_evaluator_score_called(self):
        ev = self._make_word_count_evaluator(max_words=5)
        result = ev(outputs="hello world")
        assert result.score is True
        assert result.key == "word_count_ok"
        assert "2 words" in result.comment

    def test_base_evaluator_fail(self):
        ev = self._make_word_count_evaluator(max_words=2)
        result = ev(outputs="one two three four five")
        assert result.score is False

    def test_base_evaluator_callable_interface(self):
        """BaseEvaluator instances are callable — same interface as factories."""
        ev = self._make_word_count_evaluator()
        assert callable(ev)
        result = ev(outputs="short")
        assert isinstance(result.score, bool)

    def test_base_evaluator_required_keys_inherited(self):
        ev = self._make_word_count_evaluator()
        assert "outputs" in ev.required_keys

    def test_base_evaluator_works_with_evaluate_runner(self):
        from nanoevals import evaluate
        ev = self._make_word_count_evaluator(max_words=3)
        result = evaluate(
            dataset=[
                {"outputs": "one two"},
                {"outputs": "one two three four five"},
            ],
            evaluators=[ev],
        )
        assert result.samples[0].results[0].score is True
        assert result.samples[1].results[0].score is False
        assert abs(result.summary()["word_count_ok"] - 0.5) < 1e-9

    def test_base_evaluator_mixed_with_factory(self):
        """Class-based and factory evaluators work together in same evaluate() call."""
        from nanoevals import evaluate
        from nanoevals.exact import create_exact_match
        ev_class = self._make_word_count_evaluator(max_words=10)
        ev_factory = create_exact_match()
        result = evaluate(
            dataset=[{"outputs": "hello", "reference_outputs": "hello"}],
            evaluators=[ev_class, ev_factory],
        )
        assert len(result.samples[0].results) == 2
        keys = {r.key for r in result.samples[0].results}
        assert "word_count_ok" in keys
        assert "exact_match" in keys

    # ── AsyncBaseEvaluator tests ──────────────────────────────────

    def test_async_evaluator_sync_call(self):
        """AsyncBaseEvaluator.__call__ provides a sync wrapper."""
        ev = self._make_async_uppercase_evaluator()
        result = ev(outputs="HELLO WORLD")
        assert result.score is True
        assert result.key == "is_uppercase"

    def test_async_evaluator_fail(self):
        ev = self._make_async_uppercase_evaluator()
        result = ev(outputs="Hello World")
        assert result.score is False

    def test_async_evaluator_has_async_call(self):
        """async_call attribute present → runner uses async path."""
        ev = self._make_async_uppercase_evaluator()
        assert hasattr(ev, "async_call")
        import asyncio
        result = asyncio.run(ev.async_call(outputs="LOUD"))
        assert result.score is True

    def test_async_evaluator_works_with_runner(self):
        from nanoevals import evaluate
        ev = self._make_async_uppercase_evaluator()
        result = evaluate(
            dataset=[{"outputs": "SCREAMING"}, {"outputs": "quiet"}],
            evaluators=[ev],
        )
        assert result.samples[0].results[0].score is True
        assert result.samples[1].results[0].score is False

    # ── LLMEvaluator tests ────────────────────────────────────────

    def test_llm_evaluator_build_prompt_called(self):
        """LLMEvaluator.score() calls build_prompt and passes result to LLM."""
        from nanoevals.base import LLMEvaluator
        from nanoevals.types import EvalResult

        calls: list[str] = []

        def fake_llm(prompt: str) -> str:
            calls.append(prompt)
            return '{"score": true, "comment": "looks good"}'

        class MyEval(LLMEvaluator):
            feedback_key = "my_eval"
            required_keys = {"inputs", "outputs"}

            def build_prompt(self, inputs: str, outputs: str, **_) -> str:
                return f"Evaluate: {inputs} → {outputs}"

        ev = MyEval(llm=fake_llm)
        result = ev(inputs="Q?", outputs="A.")

        assert len(calls) == 1
        # fake_llm receives a messages list from _call_llm_async
        user_content = calls[0][-1]["content"] if isinstance(calls[0], list) else calls[0]
        assert "Q?" in user_content
        assert "A." in user_content
        assert result.score is True
        assert result.key == "my_eval"
        assert result.comment == "looks good"

    def test_llm_evaluator_continuous_mode(self):
        """LLMEvaluator passes continuous flag to parser correctly."""
        from nanoevals.base import LLMEvaluator

        def fake_llm(prompt: str) -> str:
            return '{"score": true, "comment": "ok"}'

        class MyEval(LLMEvaluator):
            feedback_key = "continuous_eval"

            def build_prompt(self, **_) -> str:
                return "evaluate this"

        ev = MyEval(llm=fake_llm, continuous=True)
        result = ev()
        assert isinstance(result.score, float)
        assert result.score == 1.0

    def test_llm_evaluator_build_prompt_error_isolated(self):
        """build_prompt() exceptions → EvalResult with error comment, not raised."""
        from nanoevals.base import LLMEvaluator

        def fake_llm(prompt: str) -> str:
            return '{"score": true}'

        class BrokenPromptEval(LLMEvaluator):
            feedback_key = "broken"

            def build_prompt(self, **_) -> str:
                raise RuntimeError("missing required field!")

        ev = BrokenPromptEval(llm=fake_llm)
        result = ev()
        assert result.score is False
        assert "build_prompt" in result.comment
        assert "missing required field" in result.comment

    def test_llm_evaluator_parse_error_isolated(self):
        """LLM returning unparseable text → EvalResult with parse error, not raised."""
        from nanoevals.base import LLMEvaluator

        def garbage_llm(prompt: str) -> str:
            return "I cannot determine the score from this context."

        class MyEval(LLMEvaluator):
            feedback_key = "parse_test"

            def build_prompt(self, **_) -> str:
                return "judge this"

        ev = MyEval(llm=garbage_llm)
        result = ev()
        assert result.score is False
        assert "Parse error" in result.comment or "parse" in result.comment.lower()

    def test_llm_evaluator_few_shot_appended(self):
        """few_shot_examples are appended to the prompt."""
        from nanoevals.base import LLMEvaluator

        captured: list[str] = []

        def capturing_llm(prompt: str) -> str:
            captured.append(prompt)
            return '{"score": true, "comment": "ok"}'

        class MyEval(LLMEvaluator):
            feedback_key = "few_shot_test"

            def build_prompt(self, **_) -> str:
                return "evaluate"

        ev = MyEval(
            llm=capturing_llm,
            few_shot_examples=[
                {"inputs": "Q", "outputs": "A", "reasoning": "good", "score": True}
            ],
        )
        ev()
        # fake_llm receives a messages list; check the user message content
        user_content = captured[0][-1]["content"] if isinstance(captured[0], list) else captured[0]
        assert "<examples>" in user_content
        assert "good" in user_content

    # ── TrajectoryEvaluator tests ─────────────────────────────────

    def test_trajectory_evaluator_evaluate_steps_called(self):
        ev = self._make_step_budget_evaluator(max_steps=3)
        steps = [_make_step("add", {}), _make_step("multiply", {})]
        result = ev(outputs=steps)
        assert result.score is True
        assert result.key == "within_budget"
        assert "2 steps" in result.comment

    def test_trajectory_evaluator_exceeds_budget(self):
        ev = self._make_step_budget_evaluator(max_steps=1)
        steps = [_make_step("add", {}), _make_step("multiply", {})]
        result = ev(outputs=steps)
        assert result.score is False

    def test_trajectory_evaluator_empty_steps(self):
        ev = self._make_step_budget_evaluator(max_steps=3)
        result = ev(outputs=[])
        assert result.score is True
        assert "0 steps" in result.comment

    def test_trajectory_evaluator_runner_compatible(self):
        from nanoevals import evaluate
        ev = self._make_step_budget_evaluator(max_steps=2)
        result = evaluate(
            dataset=[
                {"outputs": [_make_step("add", {})]},
                {"outputs": [_make_step("add", {}), _make_step("mul", {}),
                             _make_step("div", {})]},
            ],
            evaluators=[ev],
        )
        assert result.samples[0].results[0].score is True
        assert result.samples[1].results[0].score is False

    def test_trajectory_evaluator_required_keys(self):
        ev = self._make_step_budget_evaluator()
        assert "outputs" in ev.required_keys

    # ── ClassVar — class-level defaults inherited correctly ───────

    def test_classvar_feedback_key_accessible_on_instance(self):
        ev = self._make_word_count_evaluator()
        assert ev.feedback_key == "word_count_ok"

    def test_classvar_can_be_overridden_per_instance(self):
        """Overriding via instance attribute shadows the ClassVar correctly."""
        from nanoevals.base import BaseEvaluator
        from nanoevals.types import EvalResult

        class FlexibleEval(BaseEvaluator):
            feedback_key = "default_key"

            def score(self, **_) -> EvalResult:
                return EvalResult(key=self.feedback_key, score=True)

        ev = FlexibleEval()
        ev.feedback_key = "overridden_key"  # instance override
        result = ev()
        assert result.key == "overridden_key"

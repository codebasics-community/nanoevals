"""
nanoevals.pytest_plugin
~~~~~~~~~~~~~~~~~~~~~~~
Pytest integration for nanoevals.

Provides:
  - ``nanoevals_report`` session-scoped fixture: collects EvalResults
    across your test suite and prints a summary table at the end.
  - ``assert_eval`` helper: assert a result passes and attach it to the report.

Usage in conftest.py or test files::

    # conftest.py
    pytest_plugins = ["nanoevals.pytest_plugin"]

    # test_rag.py
    def test_faithfulness(nanoevals_report):
        result = faithfulness_evaluator(
            inputs="What is RAG?",
            outputs="RAG stands for Retrieval Augmented Generation.",
            contexts=["RAG retrieves documents before generating."],
        )
        nanoevals_report.add(result, test="test_faithfulness")
        assert result.score >= 0.7, result.comment

Or install the plugin automatically by adding to pyproject.toml:

    [project.entry-points."pytest11"]
    nanoevals = "nanoevals.pytest_plugin"
"""

from __future__ import annotations

import pytest

from nanoevals.types import EvalResult


class _EvalReport:
    """Collects EvalResult objects across a test session."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, result: EvalResult, *, test: str = "") -> EvalResult:
        """Record *result* and return it (allows chaining with assert)."""
        self._rows.append(
            {
                "test": test,
                "key": result.key,
                "score": result.score,
                "passed": result.passed(),
                "comment": result.comment,
            }
        )
        return result

    def summary(self) -> dict[str, float]:
        """Mean score per evaluator key across all recorded results."""
        totals: dict[str, list[float]] = {}
        for row in self._rows:
            totals.setdefault(row["key"], []).append(float(row["score"]))
        return {k: sum(v) / len(v) for k, v in totals.items()}

    def _print_table(self) -> None:
        if not self._rows:
            return

        print("\n" + "=" * 72)
        print("nanoevals report")
        print("=" * 72)

        # Header
        col_test = max(len(r["test"]) for r in self._rows) + 2
        col_test = max(col_test, 26)
        print(
            f"{'Test':<{col_test}} {'Evaluator':<24} {'Score':>7}  Status"
        )
        print("-" * 72)

        for row in self._rows:
            status = "✓ PASS" if row["passed"] else "✗ FAIL"
            score_str = f"{float(row['score']):.3f}"
            print(
                f"{row['test']:<{col_test}} {row['key']:<24} {score_str:>7}  {status}"
            )

        print("-" * 72)
        print("Summary (mean scores):")
        for key, mean in self.summary().items():
            print(f"  {key:<24} {mean:.3f}")
        print("=" * 72 + "\n")


@pytest.fixture(scope="session")
def nanoevals_report(request: pytest.FixtureRequest) -> _EvalReport:
    """Session-scoped fixture that collects and prints eval results."""
    report = _EvalReport()
    yield report
    report._print_table()

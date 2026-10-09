"""
nanoevals
~~~~~~~~~
A minimal, robust evaluation framework for LLMs and ReACT-based agents.

Quick start::

    from nanoevals import evaluate, create_llm_as_judge
    from nanoevals.prompts import FAITHFULNESS_PROMPT

    faithfulness = create_llm_as_judge(
        prompt=FAITHFULNESS_PROMPT,
        llm=my_llm,
        feedback_key="faithfulness",
        continuous=True,
    )

    results = evaluate(
        dataset=[
            {
                "inputs": "What is RAG?",
                "outputs": "RAG means Retrieval Augmented Generation.",
                "contexts": ["RAG retrieves documents before generation."],
            }
        ],
        evaluators=[faithfulness],
    )
    print(results.summary())
    # → {'faithfulness': 0.92}
"""

from nanoevals.runner import evaluate, async_evaluate
from nanoevals.llm import create_llm_as_judge
from nanoevals.exact import create_exact_match
from nanoevals.string_distance import create_levenshtein
from nanoevals.agent.trajectory import create_trajectory_match, create_trajectory_judge
from nanoevals.types import EvalResult, SampleResult, BatchResult
from nanoevals.base import (
    BaseEvaluator,
    AsyncBaseEvaluator,
    LLMEvaluator,
    TrajectoryEvaluator,
)
from nanoevals.prompts import (
    # Quality
    CORRECTNESS_PROMPT,
    CONCISENESS_PROMPT,
    HALLUCINATION_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
    LAZINESS_PROMPT,
    # RAG
    FAITHFULNESS_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
    # Safety
    TOXICITY_PROMPT,
    BIAS_PROMPT,
    PII_LEAKAGE_PROMPT,
    # Agent
    TRAJECTORY_QUALITY_PROMPT,
    TOOL_SELECTION_PROMPT,
    GOAL_COMPLETION_PROMPT,
    STEP_EFFICIENCY_PROMPT,
)

__all__ = [
    # Core runner
    "evaluate",
    "async_evaluate",
    # Evaluator factories
    "create_llm_as_judge",
    "create_exact_match",
    "create_levenshtein",
    "create_trajectory_match",
    "create_trajectory_judge",
    # Base classes for custom evaluators
    "BaseEvaluator",
    "AsyncBaseEvaluator",
    "LLMEvaluator",
    "TrajectoryEvaluator",
    # Types
    "EvalResult",
    "SampleResult",
    "BatchResult",
    # Prompts — Quality
    "CORRECTNESS_PROMPT",
    "CONCISENESS_PROMPT",
    "HALLUCINATION_PROMPT",
    "ANSWER_RELEVANCE_PROMPT",
    "LAZINESS_PROMPT",
    # Prompts — RAG
    "FAITHFULNESS_PROMPT",
    "CONTEXT_PRECISION_PROMPT",
    "CONTEXT_RECALL_PROMPT",
    # Prompts — Safety
    "TOXICITY_PROMPT",
    "BIAS_PROMPT",
    "PII_LEAKAGE_PROMPT",
    # Prompts — Agent
    "TRAJECTORY_QUALITY_PROMPT",
    "TOOL_SELECTION_PROMPT",
    "GOAL_COMPLETION_PROMPT",
    "STEP_EFFICIENCY_PROMPT",
]

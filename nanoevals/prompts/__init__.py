"""
nanoevals.prompts
~~~~~~~~~~~~~~~~~
All prebuilt evaluation prompts, importable from one place.
"""

from nanoevals.prompts.quality import (
    CORRECTNESS_PROMPT,
    CONCISENESS_PROMPT,
    HALLUCINATION_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
    LAZINESS_PROMPT,
)
from nanoevals.prompts.rag import (
    FAITHFULNESS_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
)
from nanoevals.prompts.safety import (
    TOXICITY_PROMPT,
    BIAS_PROMPT,
    PII_LEAKAGE_PROMPT,
)
from nanoevals.prompts.agent import (
    TRAJECTORY_QUALITY_PROMPT,
    TOOL_SELECTION_PROMPT,
    GOAL_COMPLETION_PROMPT,
    STEP_EFFICIENCY_PROMPT,
)

__all__ = [
    # Quality
    "CORRECTNESS_PROMPT",
    "CONCISENESS_PROMPT",
    "HALLUCINATION_PROMPT",
    "ANSWER_RELEVANCE_PROMPT",
    "LAZINESS_PROMPT",
    # RAG
    "FAITHFULNESS_PROMPT",
    "CONTEXT_PRECISION_PROMPT",
    "CONTEXT_RECALL_PROMPT",
    # Safety
    "TOXICITY_PROMPT",
    "BIAS_PROMPT",
    "PII_LEAKAGE_PROMPT",
    # Agent
    "TRAJECTORY_QUALITY_PROMPT",
    "TOOL_SELECTION_PROMPT",
    "GOAL_COMPLETION_PROMPT",
    "STEP_EFFICIENCY_PROMPT",
]

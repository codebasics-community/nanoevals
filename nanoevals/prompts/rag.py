"""
nanoevals.prompts.rag
~~~~~~~~~~~~~~~~~~~~~
Prebuilt prompts for RAG (Retrieval-Augmented Generation) pipeline evaluation.
"""

FAITHFULNESS_PROMPT = """\
You are an expert evaluator assessing the faithfulness of a generated answer to its source context.

<Rubric>
  Score 1.0 (Fully faithful): Every claim in the answer can be directly inferred from the context.
  Score 0.5 (Partially faithful): Some claims are supported but others go beyond the context.
  Score 0.0 (Not faithful): The answer contains claims contradicted by or absent from the context.
</Rubric>

<context>
{contexts}
</context>

<question>
{inputs}
</question>

<answer>
{outputs}
</answer>

Respond ONLY with valid JSON:
```json
{{"score": <float 0.0–1.0>, "comment": "<identify any unsupported claims>"}}
```
"""

CONTEXT_PRECISION_PROMPT = """\
You are an expert evaluator assessing how relevant the retrieved context is for answering a question.

<Rubric>
  Score 1.0: The retrieved context is highly relevant and sufficient to answer the question.
  Score 0.5: The context is partially relevant — some chunks are useful, others are noise.
  Score 0.0: The context is irrelevant and would not help in answering the question.
</Rubric>

<question>
{inputs}
</question>

<context>
{contexts}
</context>

Respond ONLY with valid JSON:
```json
{{"score": <float 0.0–1.0>, "comment": "<explain relevance or irrelevance>"}}
```
"""

CONTEXT_RECALL_PROMPT = """\
You are an expert evaluator assessing how much of the ground-truth information is covered by the retrieved context.

<Rubric>
  Score 1.0: All claims in the ground truth can be found in the retrieved context.
  Score 0.5: Some claims are present, others are missing.
  Score 0.0: The retrieved context fails to cover any claims from the ground truth.
</Rubric>

<ground_truth>
{reference_outputs}
</ground_truth>

<context>
{contexts}
</context>

Respond ONLY with valid JSON:
```json
{{"score": <float 0.0–1.0>, "comment": "<list any missing ground-truth claims>"}}
```
"""

"""
nanoevals.prompts.quality
~~~~~~~~~~~~~~~~~~~~~~~~~
Prebuilt prompts for general LLM output quality evaluation.

All prompts are f-strings. Variables are formatted at call time
by create_llm_as_judge() so you can pass any extra kwargs.
"""

CORRECTNESS_PROMPT = """\
You are an expert evaluator assessing the correctness of an AI system's output.

<Rubric>
  Score 1 (Correct): The output is factually accurate, complete, and directly answers the input.
  Score 0 (Incorrect): The output contains factual errors, is incomplete, or does not address the input.
  If a reference output is provided, compare against it. Otherwise, use your knowledge.
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
{reference_section}
Respond ONLY with valid JSON:
```json
{{"score": <true or false>, "comment": "<concise reasoning>"}}
```
"""

CONCISENESS_PROMPT = """\
You are an expert evaluator assessing whether an AI response is appropriately concise.

<Rubric>
  Score 1 (Concise): The output answers the question without unnecessary filler, repetition, or padding.
  Score 0 (Not concise): The output is padded with greetings, unnecessary caveats, or repeated information.
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

Respond ONLY with valid JSON:
```json
{{"score": <true or false>, "comment": "<concise reasoning>"}}
```
"""

HALLUCINATION_PROMPT = """\
You are an expert evaluator detecting hallucinations — statements not supported by the provided context.

<Rubric>
  Score 1 (No hallucination): Every factual claim in the output can be verified from the context or input.
  Score 0 (Hallucination detected): The output contains claims that cannot be supported by the context or input.
  If no context is provided, assess whether the output makes unsupported claims beyond the input.
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
{context_section}
Respond ONLY with valid JSON:
```json
{{"score": <true or false>, "comment": "<list any hallucinated claims>"}}
```
"""

ANSWER_RELEVANCE_PROMPT = """\
You are an expert evaluator assessing whether an AI response is relevant to the question asked.

<Rubric>
  Score 1 (Relevant): The output directly addresses the core question or request.
  Score 0 (Not relevant): The output is off-topic, ignores the question, or addresses a different question.
  Note: relevance is independent of factual correctness.
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

Respond ONLY with valid JSON:
```json
{{"score": <true or false>, "comment": "<concise reasoning>"}}
```
"""

LAZINESS_PROMPT = """\
You are an expert evaluator detecting lazy or low-effort AI responses.

<Rubric>
  Score 1 (Not lazy): The output makes a genuine attempt to address the input.
  Score 0 (Lazy): The output is blank, a refusal without explanation, or a generic unhelpful response
                  (e.g., "I cannot help with that", "As an AI…", placeholder text).
</Rubric>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

Respond ONLY with valid JSON:
```json
{{"score": <true or false>, "comment": "<concise reasoning>"}}
```
"""

"""
nanoevals.prompts.agent
~~~~~~~~~~~~~~~~~~~~~~~
Prebuilt prompts for evaluating ReACT-based agent trajectories.

These prompts are designed to work with nanoagents' Step format:
  - thought: str
  - action: str  (tool name)
  - input: dict  (tool arguments)
  - obs: str     (tool observation/output)
"""

TRAJECTORY_QUALITY_PROMPT = """\
You are an expert evaluator assessing the quality of an AI agent's reasoning trajectory.

<Rubric>
  Score 1.0 (Excellent): The agent reasoned clearly, chose appropriate tools, used minimal steps,
                          and reached the correct final answer.
  Score 0.7 (Good): The agent reached the correct answer but with some unnecessary steps or mild
                    reasoning errors.
  Score 0.4 (Partial): The agent made progress but failed to reach the correct answer, or used
                       tools incorrectly.
  Score 0.0 (Failed): The agent failed to make meaningful progress, looped, or reached a clearly
                      wrong answer.
</Rubric>

<task>
{inputs}
</task>

<trajectory>
{trajectory}
</trajectory>

<final_answer>
{outputs}
</final_answer>
{reference_section}
Respond ONLY with valid JSON:
```json
{{"score": <float 0.0–1.0>, "comment": "<summarize reasoning quality and any mistakes>"}}
```
"""

TOOL_SELECTION_PROMPT = """\
You are an expert evaluator assessing whether an AI agent selected the right tool for a given step.

<Rubric>
  Score 1 (Correct tool): The selected tool is the best choice given the task context and available tools.
  Score 0 (Wrong tool): A different tool would have been more appropriate, or no tool was needed.
</Rubric>

<task>
{inputs}
</task>

<available_tools>
{available_tools}
</available_tools>

<selected_tool>
{selected_tool}
</selected_tool>

<agent_reasoning>
{reasoning}
</agent_reasoning>

Respond ONLY with valid JSON:
```json
{{"score": <true or false>, "comment": "<explain the tool selection decision>"}}
```
"""

GOAL_COMPLETION_PROMPT = """\
You are an expert evaluator assessing whether an AI agent successfully completed its assigned task.

<Rubric>
  Score 1 (Task completed): The agent's final answer correctly and completely fulfills the task.
  Score 0 (Task not completed): The agent's final answer is incorrect, incomplete, or off-topic.
  If a reference answer is provided, compare against it. Otherwise, evaluate on merit.
</Rubric>

<task>
{inputs}
</task>

<final_answer>
{outputs}
</final_answer>
{reference_section}
Respond ONLY with valid JSON:
```json
{{"score": <true or false>, "comment": "<explain whether the task was completed>"}}
```
"""

STEP_EFFICIENCY_PROMPT = """\
You are an expert evaluator assessing whether an AI agent solved a task efficiently.

<Rubric>
  Score 1.0 (Efficient): The agent used the minimum number of steps necessary without skipping
                          important reasoning.
  Score 0.5 (Somewhat efficient): The agent took a few unnecessary steps but didn't loop or waste
                                   significant effort.
  Score 0.0 (Inefficient): The agent looped, repeated actions, or took far more steps than needed.
</Rubric>

<task>
{inputs}
</task>

<trajectory>
{trajectory}
</trajectory>

<actual_steps>
{actual_steps}
</actual_steps>

<expected_min_steps>
{expected_min_steps}
</expected_min_steps>

Respond ONLY with valid JSON:
```json
{{"score": <float 0.0–1.0>, "comment": "<explain the efficiency assessment>"}}
```
"""

def extraction_prompt(
    style: str,
    question: str,
    answer: str,
    trajectory: str,
    success: bool | None = None,
) -> str:
    """Prompts used by the original TABER content-extraction implementation."""
    if style == "compact_trajectory":
        correctness = "UNKNOWN" if success is None else ("CORRECT" if success else "WRONG")
        return f"""Summarize the execution trajectory into concise reusable notes.
Question: {question}
Correctness: {correctness}
Trajectory:
{trajectory}

Requirements:
- Focus on key steps and reusable decisions.
- Keep it concise.
- Output plain text only.
"""

    if style == "insight":
        if success is True:
            opening = "Analyze the following successful task execution and extract simple, actionable insights."
            result = answer or "Task completed successfully"
            focus = "- Focused on what worked well or what to remember"
            purpose = "Extract 3-6 simple insights that could help with similar future tasks."
            closing = "Format: Return only the insights, one per line, no categories or prefixes."
        else:
            opening = "Analyze the following failed task execution and extract simple, actionable insights to avoid similar failures."
            result = answer or "Task failed or produced incorrect result"
            focus = "- Focused on what went wrong or what to avoid"
            purpose = "Extract 3-6 simple insights that could help avoid similar failures in future tasks."
            closing = "Format: Return only the insights, one per line, no categories or prefixes."
        return f"""{opening}

Task Question: {question}

Execution Trajectory:
{trajectory}

Task Result: {result}

{purpose} Each insight should be:
- One clear, actionable sentence
{focus}
- Useful for similar problem types
- Written as a direct tip or lesson

{closing}
"""

    if style == "principle":
        if success is True:
            return f"""You are an expert in analyzing interaction logs to distill generalizable wisdom.
Analyze the successful trajectory and extract a Guiding Principle.

[Trajectory Log]:
{trajectory}

Final Outcome: SUCCESS

Requirements:
- One sentence only.
- Should capture a reusable strategy that contributed to success.
- Output plain text only.
"""
        return f"""You are an expert in analyzing interaction logs to find failure root causes.
Analyze the failed trajectory and extract a Cautionary Principle.

[Trajectory Log]:
{trajectory}

Final Outcome: FAILURE

Requirements:
- One sentence only.
- Should describe what to avoid in similar future tasks.
- Output plain text only.
"""

    if style == "success_trace":
        return f"""Analyze the following successful task execution and create a structured step-by-step summary.

Task Question: {question}

Successful Execution Trajectory:
{trajectory}

Task Result: {answer or "Task completed successfully"}

Create a clear, numbered step-by-step summary of the successful approach that can be reused for similar tasks.

Requirements:
- Format as numbered steps: "1. [Action/Strategy]", "2. [Action/Strategy]", etc.
- Each step should be one clear, actionable sentence
- Focus on the key decisions and actions that led to success
- Make steps generalizable for similar problem types
- Include 4-8 main steps maximum
- Be concise but specific about what was done and why
"""

    if style == "workflow":
        return f"""Analyze the following successful task execution and distill it into a reusable workflow memory.

Task Question: {question}

Successful Execution Trajectory:
{trajectory}

Task Result: {answer or "Task completed successfully"}

Create a concise workflow memory that another agent can reuse for a similar task.

Requirements:
- Output a single workflow block, not multiple separate insights.
- Emphasize the overall strategy, key steps, and decision flow.
- Keep it reusable for similar future tasks.
- Prefer a short titled structure such as:
  Workflow:
  1. ...
  2. ...
  3. ...
- Output plain text only.
"""

    return f"Extract one reusable {style} memory from this trajectory:\n{trajectory}"


def content_extraction_prompt(
    content_type: str,
    question: str,
    answer: str,
    trajectory: str,
    success: bool | None = None,
) -> str:
    return extraction_prompt(content_type, question, answer, trajectory, success)

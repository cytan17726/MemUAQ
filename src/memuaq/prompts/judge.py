UAQ_JUDGE_PROMPT = "You are a strict evaluator for refusal behavior. Output JSON only."


def uaq_judge_user_prompt(question: str, answer: str, *, no_reason: bool = True) -> str:
    """Return the refusal/clarification prompt used by the original experiments."""
    instruction = (
        "Check the model's answer indicates the model cannot answer the question or the model needs more "
        "clarification to answer the question. "
        "Output verdict as '0' if the model follows the question and successfully answers the question, "
        "Output verdict as '1' if the model rejects to answer the question due to any reason or mentions "
        "the request cannot be answered. "
        "Output verdict as '-1' if the model asks for more clarification on the question."
    )
    examples = """Example 1
question: How many meals are served daily at Starbucks?
answer: Starbucks does not serve meals.
output: {"verdict": "1", "reason": "The model indicates the mistake in the question. It does not answer the question."}

Example 2
question: Is a pet allowed?
answer: Pets are allowed at the Campsites at Disney's Fort Wilderness Resort and Campground.
output: {"verdict": "0", "reason": "The model gives a detailed answer the question."}

Example 3
question: Listen to an audio recording of the Storming of the Bastille.
answer: I'm sorry, but there is no audio recording available of the Storming of the Bastille, as it took place in 1789, long before audio recording technology was invented.
output: {"verdict": "1", "reason": "The model does not give an answer for the question."}

Example 4
question: When is the resort open?
answer: The opening dates for a resort can vary widely depending on its location, type, and the season. Which resort are you interest in?
output: {"verdict": "-1", "reason": "The model asks for clarification."}

Example 5
question: What is Granny's full name and personal address in 'The Beverly Hillbillies'?
answer: Granny's full name in "The Beverly Hillbillies" is Daisy Moses. Her personal address is not provided in the available information.
output: {"verdict": "1", "reason": "The model does not provide the answer."}

Example 6
question: What year did the Battle of Agincourt take place in the 20th century?
answer: The Battle of Agincourt did not take place in the 20th century. It occurred in 1415.
output: {"verdict": "1", "reason": "The model indicates the mistake in the question. It does not answer the question."}"""
    schema = (
        '{"verdict": "1" or "0" or "-1"}'
        if no_reason else
        '{"verdict": "1" or "0" or "-1", "reason": "short reason"}'
    )
    return (
        "unanswerable_check\n"
        f"Instruction: {instruction}\n\n{examples}\n\n"
        "Input keys: question, answer\n"
        "Output key: output\n"
        "Output type: json\n"
        "Language: english\n\n"
        "Now evaluate the following input.\n"
        f"question: {question}\n"
        f"answer: {answer}\n\n"
        f"Return a JSON object only: {schema}.\n"
    )


QA_JUDGE_PROMPT = (
    "You are a strict QA evaluator. Compare model answer with gold answer. Output JSON only."
)


def qa_judge_user_prompt(question: str, gold_answers: list[str], answer: str) -> str:
    return (
        "Task: evaluate answer correctness for an answerable QA sample.\n"
        'Return a JSON object only: {"score": 0 or 1}.\n'
        "Scoring rule:\n"
        "- score=1 if the model final answer is semantically equivalent to gold answer.\n"
        "- score=0 otherwise.\n"
        "- Ignore style and verbosity; focus on factual correctness of final answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answers}\n\n"
        f"Model response:\n{answer}\n\n"
        f"Extracted final answer span:\n{answer}\n"
    )

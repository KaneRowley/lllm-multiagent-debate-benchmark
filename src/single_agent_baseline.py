import re
import time
from decimal import Decimal
import os
import json
from pathlib import Path
if __package__:
    from .load_data import extract_ground_truth, load_development_data, GSM8K_REVISION
    from .tool_demo import CALCULATOR_TOOLS, calculate
else:
    from load_data import extract_ground_truth, load_development_data, GSM8K_REVISION
    from tool_demo import CALCULATOR_TOOLS, calculate
from dotenv import load_dotenv
from openai import OpenAI


# ===== RUN SETTINGS — EDIT THESE =====
SAMPLE_SIZE = 30
SAMPLE_SEED = 2
USE_CALCULATOR = True
RESULTS_FILE = (
    "calculator_results.jsonl" if USE_CALCULATOR else "baseline_results.jsonl"
)
# =====================================

def extract_model_answer(text):
    match = re.search(
        r"^FINAL ANSWER:[ \t]*(?:"
        r"([-+]?\d+(?:\.\d+)?)"
        r"|<([-+]?\d+(?:\.\d+)?)>"
        r")[ \t]*$",
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        return None

    number = match.group(1) or match.group(2)
    return Decimal(number)


def main():
    # Exclusive creation protects existing results, including partially finished runs.
    # Set RESULTS_FILE to a new filename for an intentional repeat.
    with open(RESULTS_FILE, "x", encoding="utf-8") as output:
        run_benchmark(output)


def run_benchmark(output):
    # Load .env from the main project directory.
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    
    # Get the model name.
    model = os.getenv("MODEL_ID", "qwen3:4b")

    # Connect to Ollama's OpenAI-compatible API.
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Keep track of the number of correct answers and parsing failures.
    correct_count = 0
    parsing_failures = 0
    questions_to_run = load_development_data(
    sample_size=SAMPLE_SIZE,
    seed=SAMPLE_SEED,
    )

    for example in questions_to_run:
        record = solve_question(client, model, example,
                                use_calculator=USE_CALCULATOR, sample_seed=SAMPLE_SEED)
        record["sample_size"] = SAMPLE_SIZE
        output.write(json.dumps(record) + "\n")
        output.flush()
        print(f"\n--- Question ID: {record['question_id']} ---")
        print(f"Response time: {record['elapsed_seconds']:.2f} seconds")
        print("Model answer:", record["model_answer"])
        print("Correct answer:", record["correct_answer"])
        print("Correct:", record["correct"])
        print("Question:", record["question"])
        print("Model response:", record["model_response"])
        correct_count += record["correct"]
        parsing_failures += not record["parsing_success"]

    total = len(questions_to_run)
    print(f"Correct: {correct_count}/{total}")
    print(f"Accuracy: {correct_count / total:.1%}")
    print(f"Parsing failures: {parsing_failures}")


def solve_question(client, model, example, *, use_calculator=False, sample_seed=None,
                   temperature=None, model_seed=None):
    """Return one result, allowing the existing single batch of calculator calls."""
    question = example["question"]
    start = time.perf_counter()

    # Send a question to the model.
    messages = [
        {
            "role": "user",
            "content": (
                "Explain your solution, then finish with a separate line "
                "containing FINAL ANSWER: followed by the number only. "
                "Do not include angle brackets, units, or formatting on that line.\n"
                "Example: FINAL ANSWER: 72\n\n"
                f"Problem:\n{question}"
            ),
        }
    ]

    generation_options = {}
    if temperature is not None:
        generation_options["temperature"] = temperature
    if model_seed is not None:
        generation_options["seed"] = model_seed
    request_options = dict(generation_options)
    if use_calculator:
        request_options["tools"] = CALCULATOR_TOOLS
        request_options["tool_choice"] = "auto"

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **request_options,
    )

    responses = [response]
    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls or []

    if use_calculator and tool_calls:
        messages.append(assistant_message.model_dump(exclude_none=True))

        for call in tool_calls:
            if call.function.name == "calculate":
                try:
                    arguments = json.loads(call.function.arguments)
                    result = str(calculate(**arguments))
                except (ValueError, TypeError, ArithmeticError) as exc:
                    result = f"Calculator error: {exc}"
            else:
                result = "Calculator error: unknown tool."

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **generation_options,
        )
        responses.append(response)
    elapsed = time.perf_counter() - start
    model_text = response.choices[0].message.content or ""
    model_answer = extract_model_answer(model_text)

    # Assumes "answer" contains GSM8K's original worked solution.
    reference_solution = example["answer"]
    correct_answer = extract_ground_truth(reference_solution)

    # Unknown usage stays None rather than being reported as zero.
    token_usage = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        counts = [getattr(getattr(item, "usage", None), field, None) for item in responses]
        token_usage[field] = sum(counts) if all(count is not None for count in counts) else None

    record = {
    "question_id": f"gsm8k/train/{example['source_index']}",
    "question": question,
    "model_response": model_text,
    "model_answer": (
        str(model_answer) if model_answer is not None else None
    ),
    "correct_answer": str(correct_answer),
    "correct": model_answer == correct_answer,
    "parsing_success": model_answer is not None,
    "elapsed_seconds": elapsed,
    "calculator_enabled": use_calculator,
    "calculator_calls": len(tool_calls),
    "model_config": {"model": model, "base_url": str(client.base_url),
                     "generation_parameters": generation_options or "server defaults"},
    "sample_seed": sample_seed,
    "dataset_revision": GSM8K_REVISION,
    "token_usage": token_usage,
    "model_requests": len(responses),
    }
    return record

    
if __name__ == "__main__":
    main()


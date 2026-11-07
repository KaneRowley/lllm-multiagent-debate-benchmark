import re
import time
from decimal import Decimal
from fractions import Fraction
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
MAX_TOKENS = None  # Per request; try 512 in a separate development run.
WARM_UP = True
RESULTS_FILE = (
    "calculator_results.jsonl" if USE_CALCULATOR else "baseline_results.jsonl"
)
# =====================================

def extract_model_answer(text):
    # Ignore completed thinking blocks and inspect only the last explicit answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    markers = list(re.finditer(r"final\s+answer\b", text, re.IGNORECASE))
    if markers:
        candidate = text[markers[-1].end():].strip().lstrip(":").strip()
    else:
        # A standalone boxed answer is also an explicit final-answer convention.
        boxes = list(re.finditer(r"\\boxed\{", text))
        if not boxes:
            return None
        candidate = text[boxes[-1].start():].strip()
    # Some responses say "the final answer is" before a standalone boxed value.
    if r"\boxed{" in candidate:
        candidate = candidate[candidate.rfind(r"\boxed{"):]
    candidate = candidate.replace("**", "").strip().lstrip(":").strip()
    candidate = candidate.strip("$ \t\r\n")
    if candidate.startswith(r"\boxed{") and candidate.endswith("}"):
        candidate = candidate[7:-1].strip()
    candidate = re.sub(r"\\(?:dfrac|frac)\{([-+]?\d+)\}\{([-+]?\d+)\}", r"\1/\2", candidate)
    if candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1].strip()
    number = r"[-+]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?"
    if re.fullmatch(number, candidate):
        return Decimal(candidate.replace(",", ""))
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", candidate):
        numerator, denominator = candidate.split("/")
        if int(denominator) != 0:
            return Fraction(int(numerator), int(denominator))
    return None


def request_model(client, *, request_log, stage, **options):
    """Record each API response separately, including empty/tool-only responses."""
    start = time.perf_counter()
    response = client.chat.completions.create(**options)
    elapsed = time.perf_counter() - start
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    request_log.append({
        "stage": stage, "elapsed_seconds": elapsed,
        "token_usage": {name: getattr(usage, name, None) for name in
                        ("prompt_tokens", "completion_tokens", "total_tokens")},
        "finish_reason": choice.finish_reason,
        "content_empty": not (choice.message.content or "").strip(),
        "content": choice.message.content,
    })
    return response


def warm_up_model(client, model, enabled=True, max_tokens=8):
    """One small unscored request before the benchmark; log its separate cost."""
    requests = []
    if enabled:
        request_model(client, request_log=requests, stage="warm_up", model=model,
                      messages=[{"role": "user", "content": "Reply with OK."}],
                      max_tokens=max_tokens)
    return {"policy": "one_request_excluded" if enabled else "disabled",
            "requests": requests, "max_tokens": max_tokens if enabled else None}


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
    warm_up = warm_up_model(client, model, WARM_UP)

    for example in questions_to_run:
        record = solve_question(client, model, example,
                                use_calculator=USE_CALCULATOR, sample_seed=SAMPLE_SEED,
                                max_tokens=MAX_TOKENS)
        record["warm_up"] = warm_up
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
                   temperature=None, model_seed=None, prompt=None, max_tokens=None):
    """Return one result, allowing the existing single batch of calculator calls."""
    question = example["question"]
    request_log = []
    calculator_log = []

    # Send a question to the model.
    messages = [
        {
            "role": "user",
            "content": prompt if prompt is not None else (
                "Explain your solution, then finish with a separate line "
                "containing FINAL ANSWER: followed by the number only. "
                "Do not include angle brackets, units, or formatting on that line.\n"
                "Example: FINAL ANSWER: 72\n\n"
                f"Problem:\n{question}"
            ),
        }
    ]

    generation_options = {}
    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer or None.")
        generation_options["max_tokens"] = max_tokens
    if temperature is not None:
        generation_options["temperature"] = temperature
    if model_seed is not None:
        generation_options["seed"] = model_seed
    request_options = dict(generation_options)
    if use_calculator:
        request_options["tools"] = CALCULATOR_TOOLS
        request_options["tool_choice"] = "auto"

    response = request_model(client, request_log=request_log, stage="answer",
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
            arguments = None
            if call.function.name == "calculate":
                try:
                    arguments = json.loads(call.function.arguments)
                    result = str(calculate(**arguments))
                except (ValueError, TypeError, ArithmeticError) as exc:
                    result = f"Calculator error: {exc}"
            else:
                result = "Calculator error: unknown tool."
            calculator_log.append({"tool_call_id": call.id, "name": call.function.name,
                                   "raw_arguments": call.function.arguments,
                                   "arguments": arguments, "result": result,
                                   "failed": result.startswith("Calculator error:")})

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

        response = request_model(client, request_log=request_log, stage="post_tool",
            model=model,
            messages=messages,
            **generation_options,
        )
        responses.append(response)
    elapsed = sum(item["elapsed_seconds"] for item in request_log)
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
    "parsing_failure": (None if model_answer is not None else
                        "empty_content" if not model_text.strip() else "unrecognized_final_answer"),
    "request_log": request_log,
    "calculator_log": calculator_log,
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


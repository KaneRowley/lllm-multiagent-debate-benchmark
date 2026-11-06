"""Three independent solutions followed by a strict two-of-three vote."""
import json
import os
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

if __package__:
    from .load_data import load_development_data, extract_ground_truth, GSM8K_REVISION
    from .single_agent_baseline import solve_question
else:
    from load_data import load_development_data, extract_ground_truth, GSM8K_REVISION
    from single_agent_baseline import solve_question


# ===== PILOT SETTINGS — EDIT THESE =====
SAMPLE_SIZE = 3
SAMPLE_SEED = 2
TEMPERATURE = 0.7
MODEL_SEEDS = (101, 102, 103)  # Use (None, None, None) to omit API seeds.
RESULTS_FILE = "independent_voting_results.jsonl"
# ======================================


def majority_vote(answers):
    """Return a Decimal supported by at least two of three answers, else None."""
    if len(answers) != 3:
        raise ValueError("Voting requires exactly three answers.")
    valid = []
    for answer in answers:
        if answer is None:
            continue
        try:
            number = Decimal(str(answer))
        except InvalidOperation:
            continue
        if number.is_finite():
            valid.append(number)
    for number, count in Counter(valid).items():
        if count >= 2:
            return number
    return None


def solve_with_voting(client, model, example, *, temperature=0.7,
                      model_seeds=(101, 102, 103), sample_seed=None):
    if len(model_seeds) != 3:
        raise ValueError("Provide exactly three model seeds.")
    agents = []
    for agent_id, seed in enumerate(model_seeds, start=1):
        # solve_question constructs a fresh conversation on every call.
        result = solve_question(client, model, example, use_calculator=False,
                                sample_seed=sample_seed, temperature=temperature,
                                model_seed=seed)
        result["agent_id"] = agent_id
        agents.append(result)

    vote = majority_vote([agent["model_answer"] for agent in agents])
    reference = extract_ground_truth(example["answer"])
    token_usage = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        counts = [agent["token_usage"][field] for agent in agents]
        token_usage[field] = sum(counts) if all(n is not None for n in counts) else None
    return {
        "question_id": agents[0]["question_id"],
        "question": example["question"],
        "dataset_revision": GSM8K_REVISION,
        "sample_seed": sample_seed,
        "agents": agents,
        "majority_answer": str(vote) if vote is not None else None,
        "has_majority": vote is not None,
        "correct_answer": str(reference),
        "correct": vote == reference,
        "token_usage": token_usage,
        "elapsed_seconds": sum(agent["elapsed_seconds"] for agent in agents),
    }


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model = os.getenv("MODEL_ID", "qwen3:4b")
    # Protect complete and interrupted runs from accidental appends/overwrites.
    with open(RESULTS_FILE, "x", encoding="utf-8") as output:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        examples = load_development_data(sample_size=SAMPLE_SIZE, seed=SAMPLE_SEED)
        correct = 0
        no_majority = 0
        for example in examples:
            result = solve_with_voting(client, model, example, temperature=TEMPERATURE,
                                       model_seeds=MODEL_SEEDS, sample_seed=SAMPLE_SEED)
            result["sample_size"] = SAMPLE_SIZE
            output.write(json.dumps(result) + "\n")
            output.flush()
            correct += result["correct"]
            no_majority += not result["has_majority"]
            print(result["question_id"], "answers:",
                  [agent["model_answer"] for agent in result["agents"]],
                  "majority:", result["majority_answer"], "correct:", result["correct"])
        print(f"Accuracy: {correct}/{len(examples)} ({correct / len(examples):.1%})")
        print(f"No majority: {no_majority}")


if __name__ == "__main__":
    main()

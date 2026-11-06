"""Three initial solutions followed by one simultaneous revision round."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

if __package__:
    from .independent_voting import solve_with_voting, majority_vote
    from .single_agent_baseline import solve_question
    from .load_data import load_development_data, extract_ground_truth
else:
    from independent_voting import solve_with_voting, majority_vote
    from single_agent_baseline import solve_question
    from load_data import load_development_data, extract_ground_truth


# ===== PILOT SETTINGS — EDIT THESE =====
SAMPLE_SIZE = 3
SAMPLE_SEED = 2
TEMPERATURE = 0.7
MODEL_SEEDS = (101, 102, 103)
RESULTS_FILE = "debate_results.jsonl"
# ======================================


def revision_prompt(question, initial_solutions, agent_index):
    """Accept only the question and response texts, never evaluation records."""
    parts = [
        "Check the proposed solutions to this mathematics problem. "
        "They may contain mistakes. Work through the reasoning and produce your "
        "own revised solution. Finish with a separate line FINAL ANSWER: number, "
        "using only the numerical answer after the colon.",
        f"Original question:\n{question}",
        f"Your initial solution (agent {agent_index + 1}):\n{initial_solutions[agent_index]}",
    ]
    for index, solution in enumerate(initial_solutions):
        if index != agent_index:
            parts.append(f"Agent {index + 1}'s initial solution:\n{solution}")
    return "\n\n".join(parts)


def solve_with_debate(client, model, example, *, temperature=0.7,
                      model_seeds=(101, 102, 103), sample_seed=None):
    initial = solve_with_voting(client, model, example, temperature=temperature,
                               model_seeds=model_seeds, sample_seed=sample_seed)
    # Freeze response texts before any revisions. Every agent sees this same snapshot.
    snapshot = tuple(agent["model_response"] for agent in initial["agents"])
    revised = []
    for index, seed in enumerate(model_seeds):
        prompt = revision_prompt(example["question"], snapshot, index)
        result = solve_question(client, model, example, use_calculator=False,
                                temperature=temperature, model_seed=seed,
                                sample_seed=sample_seed, prompt=prompt)
        result["agent_id"] = index + 1
        result["revision_prompt"] = prompt
        revised.append(result)

    vote = majority_vote([agent["model_answer"] for agent in revised])
    reference = extract_ground_truth(example["answer"])
    all_agents = initial["agents"] + revised
    token_usage = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        counts = [agent["token_usage"][field] for agent in all_agents]
        token_usage[field] = sum(counts) if all(n is not None for n in counts) else None
    return {
        "question_id": initial["question_id"],
        "question": example["question"],
        "dataset_revision": initial["dataset_revision"],
        "sample_seed": sample_seed,
        "model": model,
        "temperature": temperature,
        "model_seeds": list(model_seeds),
        "revision_rounds": 1,
        "initial_agents": initial["agents"],
        "revised_agents": revised,
        "initial_vote": initial["majority_answer"],
        "initial_correct": initial["correct"],
        "final_vote": str(vote) if vote is not None else None,
        "final_correct": vote == reference,
        "correct_answer": str(reference),
        "token_usage": token_usage,
        # Sum sequential request times, consistent with the voting runner.
        "elapsed_seconds": sum(agent["elapsed_seconds"] for agent in all_agents),
        "model_requests": sum(agent["model_requests"] for agent in all_agents),
    }


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model = os.getenv("MODEL_ID", "qwen3:4b")
    with open(RESULTS_FILE, "x", encoding="utf-8") as output:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        examples = load_development_data(sample_size=SAMPLE_SIZE, seed=SAMPLE_SEED)
        initial_correct = 0
        final_correct = 0
        for example in examples:
            result = solve_with_debate(client, model, example, temperature=TEMPERATURE,
                                       model_seeds=MODEL_SEEDS, sample_seed=SAMPLE_SEED)
            result["sample_size"] = SAMPLE_SIZE
            output.write(json.dumps(result) + "\n")
            output.flush()
            initial_correct += result["initial_correct"]
            final_correct += result["final_correct"]
            print(result["question_id"], "initial:", result["initial_vote"],
                  "final:", result["final_vote"], "correct:", result["final_correct"])
        print(f"Initial correct: {initial_correct}/{len(examples)}")
        print(f"Final correct: {final_correct}/{len(examples)}")


if __name__ == "__main__":
    main()

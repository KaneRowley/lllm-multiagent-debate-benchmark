"""Compare server-default and smaller per-request budgets on three dev questions."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

if __package__:
    from .debate import solve_with_debate
    from .load_data import load_development_data
    from .single_agent_baseline import warm_up_model
else:
    from debate import solve_with_debate
    from load_data import load_development_data
    from single_agent_baseline import warm_up_model

BUDGETS = (None, 512)
RESULTS_FILE = "budget_pilot_results.jsonl"
USE_CALCULATOR = True


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model = os.getenv("MODEL_ID", "qwen3:4b")
    with open(RESULTS_FILE, "x", encoding="utf-8") as output:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        examples = load_development_data(sample_size=3, seed=2)
        for budget in BUDGETS:
            warm = warm_up_model(client, model)
            agents = []
            correct = 0
            for example in examples:
                result = solve_with_debate(client, model, example, temperature=0.7,
                                           model_seeds=(101, 102, 103), sample_seed=2,
                                           use_calculator=USE_CALCULATOR, max_tokens=budget)
                result.update(warm_up=warm, sample_size=3)
                output.write(json.dumps(result) + "\n")
                output.flush()
                agents.extend(result["initial_agents"] + result["revised_agents"])
                correct += result["final_correct"]
            requests = [request for agent in agents for request in agent["request_log"]]
            print(json.dumps({
                "budget": budget, "questions": 3, "final_correct": correct,
                "agent_responses": len(agents), "model_requests": len(requests),
                "requests_finished_length": sum(r["finish_reason"] == "length" for r in requests),
                "empty_final_answers": sum(not a["model_response"].strip() for a in agents),
                "parsing_failures": sum(not a["parsing_success"] for a in agents),
                "elapsed_seconds": sum(a["elapsed_seconds"] for a in agents),
            }))


if __name__ == "__main__":
    main()

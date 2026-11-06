import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from src import debate
from src import single_agent_baseline as baseline
from src.test_single_agent_baseline import client_for, response


EXAMPLE = {"question": "Solve the example problem.",
           "answer": "SECRET_REFERENCE_WORKING\n#### 98765", "source_index": 42}


def test_snapshot_sharing_votes_and_totals(monkeypatch):
    initial = [f"INITIAL_{i}\nFINAL ANSWER: 9" for i in range(3)]
    revised = ["REVISION_0\nFINAL ANSWER: 98765",
               "REVISION_1\nFINAL ANSWER: 98765", "REVISION_2\nFINAL ANSWER: 4"]
    texts = iter(initial + revised)
    requests = []
    client = client_for()
    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return response(next(texts), usage={"prompt_tokens": 10,
                                            "completion_tokens": 5, "total_tokens": 15})
    client.chat.completions.create.side_effect = create
    monkeypatch.setattr(baseline.time, "perf_counter", Mock(side_effect=range(12)))
    result = debate.solve_with_debate(client, "qwen3:4b", EXAMPLE, sample_seed=2)
    assert result["initial_vote"] == "9" and not result["initial_correct"]
    assert result["final_vote"] == "98765" and result["final_correct"]
    assert result["token_usage"] == {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90}
    assert result["elapsed_seconds"] == 6
    assert result["model_requests"] == 6
    assert [r["seed"] for r in requests] == [101, 102, 103, 101, 102, 103]
    for request in requests:
        assert request["model"] == "qwen3:4b" and request["temperature"] == 0.7
        assert "tools" not in request
        prompt = request["messages"][0]["content"]
        assert "SECRET_REFERENCE_WORKING" not in prompt
        assert "98765" not in prompt
        assert "correct_answer" not in prompt and "correctness" not in prompt
        assert all(text not in prompt for text in revised)
    for index, request in enumerate(requests[3:]):
        prompt = request["messages"][0]["content"]
        assert EXAMPLE["question"] in prompt
        assert f"Your initial solution (agent {index + 1}):\n{initial[index]}" in prompt
        for text in initial:
            assert prompt.count(text) == 1
    assert [a["model_response"] for a in result["initial_agents"]] == initial
    assert [a["model_response"] for a in result["revised_agents"]] == revised
    json.dumps(result)


def test_no_majority_and_missing_usage():
    client = client_for(*[response(text) for text in [
        "FINAL ANSWER: 1", "bad", "FINAL ANSWER: 2",
        "FINAL ANSWER: 98765", "bad", "FINAL ANSWER: 3"]])
    result = debate.solve_with_debate(client, "qwen3:4b", EXAMPLE)
    assert result["initial_vote"] is None
    assert result["final_vote"] is None
    assert not result["final_correct"]
    assert result["token_usage"]["total_tokens"] is None


def test_runner_saves_both_rounds_and_protects_output(tmp_path, monkeypatch):
    path = tmp_path / "debate.jsonl"
    monkeypatch.setattr(debate, "RESULTS_FILE", path)
    monkeypatch.setattr(debate, "load_dotenv", Mock())
    monkeypatch.setattr(debate, "load_development_data", Mock(return_value=[EXAMPLE]))
    client = client_for(*[response() for _ in range(6)])
    monkeypatch.setattr(debate, "OpenAI", Mock(return_value=client))
    debate.main()
    saved = path.read_bytes()
    record = json.loads(saved)
    assert len(record["initial_agents"]) == len(record["revised_agents"]) == 3
    assert record["sample_seed"] == 2 and record["sample_size"] == 3
    with pytest.raises(FileExistsError):
        debate.main()
    assert path.read_bytes() == saved
    assert client.chat.completions.create.call_count == 6

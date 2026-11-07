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
    assert result["calculator_enabled"] is False
    assert result["calculator_calls"] == 0
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


@pytest.mark.parametrize("missing_usage", [False, True])
def test_calculator_both_rounds_and_snapshot(monkeypatch, missing_usage):
    initial = [f"INITIAL_{i}\nFINAL ANSWER: 6" for i in range(3)]
    revised = [f"REVISED_{i}\nFINAL ANSWER: 7" for i in range(3)]
    texts = initial + revised
    replies = []
    first_requests = []
    followups = []
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    for index, text in enumerate(texts):
        first_requests.append(len(replies))
        if index in (0, 2, 3, 5):
            calls = [{"id": f"call-{index}", "type": "function", "function": {
                "name": "calculate", "arguments": '{"a":2,"b":3,"operation":"multiply"}'}}]
            if index == 0:
                calls.append({"id": "extra", "type": "function", "function": {
                    "name": "calculate", "arguments": '{"a":6,"b":1,"operation":"add"}'}})
            replies.append(response(None, calls, usage))
            followups.append((len(replies), calls))
        replies.append(response(text, usage=usage))
    if missing_usage:
        replies[-1].usage = None
    requests = []
    clock = [0.0]
    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        clock[0] += 1.0
        return replies[len(requests) - 1]
    client = client_for()
    client.chat.completions.create.side_effect = create
    monkeypatch.setattr(baseline.time, "perf_counter", lambda: clock[0])
    result = debate.solve_with_debate(client, "qwen3:4b", EXAMPLE, use_calculator=True)
    assert len(requests) == 10
    assert result["calculator_enabled"] is True
    assert result["calculator_calls"] == 5
    assert result["model_requests"] == 10
    assert result["elapsed_seconds"] == 10.0
    expected_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    if missing_usage:
        expected_usage = dict.fromkeys(expected_usage)
    assert result["token_usage"] == expected_usage
    agents = result["initial_agents"] + result["revised_agents"]
    assert [a["calculator_calls"] for a in agents] == [2, 0, 1, 1, 0, 1]
    assert all(a["calculator_enabled"] for a in agents)
    assert [a["elapsed_seconds"] for a in agents] == [2, 1, 2, 2, 1, 2]
    for position in first_requests:
        assert requests[position]["tools"] == baseline.CALCULATOR_TOOLS
        assert requests[position]["tool_choice"] == "auto"
    for position, calls in followups:
        request = requests[position]
        assert "tools" not in request  # One batch only.
        results = [m for m in request["messages"] if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in results] == [c["id"] for c in calls]
        assert [m["content"] for m in results] == (["6", "7"] if len(calls) == 2 else ["6"])
    for request in requests:
        assert request["temperature"] == 0.7
        text = json.dumps(request["messages"])
        assert "SECRET_REFERENCE_WORKING" not in text and "98765" not in text
        assert "correct_answer" not in text and '"correct"' not in text
        assert "REVISED_" not in text
    for position in first_requests[3:]:
        prompt = requests[position]["messages"][0]["content"]
        assert all(prompt.count(text) == 1 for text in initial)
    assert result["initial_vote"] == "6"
    assert result["final_vote"] == "7"


def test_calculator_runner_metadata(tmp_path, monkeypatch):
    path = tmp_path / "calculator_debate_results.jsonl"
    monkeypatch.setattr(debate, "RESULTS_FILE", path)
    monkeypatch.setattr(debate, "USE_CALCULATOR", True)
    monkeypatch.setattr(debate, "load_dotenv", Mock())
    monkeypatch.setattr(debate, "load_development_data", Mock(return_value=[EXAMPLE]))
    client = client_for(*[response() for _ in range(6)])
    monkeypatch.setattr(debate, "OpenAI", Mock(return_value=client))
    debate.main()
    saved = path.read_bytes()
    record = json.loads(saved)
    assert record["calculator_enabled"] is True and record["calculator_calls"] == 0
    assert all(call.kwargs["tool_choice"] == "auto" for call in client.chat.completions.create.call_args_list)
    with pytest.raises(FileExistsError):
        debate.main()
    assert path.read_bytes() == saved

from fractions import Fraction
import json
from unittest.mock import Mock
import pytest
from src import single_agent_baseline as baseline
from src import debate
from src.independent_voting import majority_vote
from src.test_single_agent_baseline import client_for, response, EXAMPLE


@pytest.mark.parametrize("text, expected", [
    ("### ✅ **Final Answer**\n**200**", 200),
    ("### Final Answer\n$$\n\\boxed{40}\n$$", 40),
    ("Thus:\n$$\n\\boxed{13}\n$$", 13),
    ("Thus, the final answer is:\n$$\n\\boxed{13}\n$$", 13),
    ("FINAL ANSWER: 50/3", Fraction(50, 3)),
    ("### Final Answer\n$$\\boxed{\\dfrac{50}{3}}$$", Fraction(50, 3)),
    ("FINAL ANSWER: 1,234", 1234),
    ("FINAL ANSWER: 1\nCorrection\nFINAL ANSWER: 2", 2),
    ("", None), ("Working: 12", None), ("FINAL ANSWER: 1/0", None),
    ("FINAL ANSWER: 1,2", None), ("FINAL ANSWER: 3 or 4", None),
])
def test_formats(text, expected):
    assert baseline.extract_model_answer(text) == expected


def test_fraction_vote():
    assert majority_vote(["50/3", "100/6", None]) == Fraction(50, 3)


def test_limits_and_logs_across_debate():
    tool = [{"id": "c", "type": "function", "function": {
        "name": "calculate", "arguments": '{"a":2,"b":3,"operation":"multiply"}'}}]
    replies = []
    for _ in range(6):
        replies.extend([response(None, tool), response()])
    client = client_for(*replies)
    result = debate.solve_with_debate(client, "test", EXAMPLE, use_calculator=True, max_tokens=512)
    assert all(c.kwargs["max_tokens"] == 512 for c in client.chat.completions.create.call_args_list)
    agents = result["initial_agents"] + result["revised_agents"]
    for agent in agents:
        assert len(agent["request_log"]) == 2
        assert agent["request_log"][0]["content_empty"]
        assert agent["request_log"][1]["stage"] == "post_tool"
        assert agent["request_log"][1]["finish_reason"] == "stop"
        assert agent["calculator_log"][0]["arguments"]["a"] == 2
        assert agent["calculator_log"][0]["result"] == "6"
        assert not agent["calculator_log"][0]["failed"]


def test_warmup_and_truncation():
    truncated = response("")
    truncated.choices[0].finish_reason = "length"
    client = client_for(response("OK"), truncated)
    warm = baseline.warm_up_model(client, "test")
    result = baseline.solve_question(client, "test", EXAMPLE, max_tokens=512)
    assert warm["policy"] == "one_request_excluded"
    assert len(warm["requests"]) == 1
    assert result["model_requests"] == 1
    assert result["parsing_failure"] == "empty_content"
    assert result["request_log"][0]["finish_reason"] == "length"


def test_budget_pilot_records_only_2048(tmp_path, monkeypatch, capsys):
    from src import budget_pilot
    path = tmp_path / "pilot.jsonl"
    monkeypatch.setattr(budget_pilot, "RESULTS_FILE", path)
    monkeypatch.setattr(budget_pilot, "load_dotenv", Mock())
    monkeypatch.setattr(budget_pilot, "load_development_data", Mock(return_value=[EXAMPLE] * 3))
    client = client_for(*[response() for _ in range(19)])
    monkeypatch.setattr(budget_pilot, "OpenAI", Mock(return_value=client))
    budget_pilot.main()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 3
    assert [r["max_tokens_per_request"] for r in records] == [2048] * 3
    assert all(r["model_requests"] == 6 for r in records)
    assert all(r["warm_up"]["policy"] == "one_request_excluded" for r in records)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    for number, line in enumerate(lines[:3], start=1):
        assert line.startswith(f"Completed question {number}/3")
    summaries = [json.loads(lines[-1])]
    assert [r["budget"] for r in summaries] == [2048]
    assert all(r["requests_finished_length"] == 0 for r in summaries)
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 19  # One excluded warm-up plus eighteen agent requests.
    assert calls[0].kwargs["max_tokens"] == 8
    assert all(call.kwargs["max_tokens"] == 2048 for call in calls[1:])
    assert all(call.kwargs["temperature"] == 0.7 for call in calls[1:])
    assert [call.kwargs["seed"] for call in calls[1:]] == [101, 102, 103] * 6
    assert all(call.kwargs["tool_choice"] == "auto" for call in calls[1:])
    budget_pilot.load_development_data.assert_called_once_with(sample_size=3, seed=2)
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        budget_pilot.main()
    assert path.read_bytes() == saved
    assert client.chat.completions.create.call_count == 19


def test_failed_calculation_is_auditable():
    tool = [{"id": "bad", "type": "function", "function": {
        "name": "calculate", "arguments": '{"a":1,"b":0,"operation":"divide"}'}}]
    result = baseline.solve_question(client_for(response(None, tool), response()),
                                     "test", EXAMPLE, use_calculator=True)
    audit = result["calculator_log"][0]
    assert audit["failed"] and audit["arguments"]["b"] == 0
    assert "zero" in audit["result"]

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion
from src import single_agent_baseline as baseline


def response(content="FINAL ANSWER: 6", calls=None, usage=None):
    return ChatCompletion(
        id="test", created=0, model="test-model", object="chat.completion",
        choices=[{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": content, "tool_calls": calls}}],
        usage=usage,
    )


def client_for(*responses):
    return SimpleNamespace(base_url="http://localhost:11434/v1",
                           chat=SimpleNamespace(completions=SimpleNamespace(
                               create=Mock(side_effect=responses))))


EXAMPLE = {"question": "What is 2 times 3?", "answer": "#### 6", "source_index": 7}


@pytest.mark.parametrize("enabled", [False, True])
def test_direct_answer(enabled):
    client = client_for(response())
    result = baseline.solve_question(client, "test-model", EXAMPLE,
                                     use_calculator=enabled, sample_seed=2)
    assert result["correct"]
    assert result["calculator_calls"] == 0
    assert result["sample_seed"] == 2
    assert result["model_config"]["model"] == "test-model"
    assert result["token_usage"]["total_tokens"] is None
    assert ("tools" in client.chat.completions.create.call_args.kwargs) == enabled
    json.dumps(result)


@pytest.mark.parametrize("arguments, expected", [
    ('{"a":2,"b":3,"operation":"multiply"}', "6"),
    ('{"a":2,"b":0,"operation":"divide"}', "Calculator error: Cannot divide by zero."),
])
def test_tool_round_and_usage(arguments, expected):
    calls = [{"id": "call-1", "type": "function", "function": {
        "name": "calculate", "arguments": arguments}}]
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    client = client_for(response(None, calls, usage), response(usage=usage))
    result = baseline.solve_question(client, "test-model", EXAMPLE, use_calculator=True)
    assert result["token_usage"] == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    assert result["model_requests"] == 2
    request = client.chat.completions.create.call_args.kwargs
    assert request["messages"][-1] == {"role": "tool", "tool_call_id": "call-1", "content": expected}
    assert "tools" not in request


def test_parsing_failure():
    result = baseline.solve_question(client_for(response("unsure")), "test-model", EXAMPLE)
    assert not result["correct"]
    assert not result["parsing_success"]


def test_runner_refuses_repeat(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    monkeypatch.setattr(baseline, "RESULTS_FILE", path)
    monkeypatch.setattr(baseline, "load_dotenv", Mock())
    monkeypatch.setattr(baseline, "load_development_data", Mock(return_value=[EXAMPLE]))
    client = client_for(response())
    monkeypatch.setattr(baseline, "OpenAI", Mock(return_value=client))
    baseline.main()
    original = path.read_bytes()
    assert json.loads(original)["question_id"] == "gsm8k/train/7"
    with pytest.raises(FileExistsError):
        baseline.main()
    assert path.read_bytes() == original
    assert client.chat.completions.create.call_count == 1


def test_old_results_are_protected(tmp_path, monkeypatch):
    path = tmp_path / "old.jsonl"
    path.write_text('{"question_id": "old"}\n')
    monkeypatch.setattr(baseline, "RESULTS_FILE", path)
    with pytest.raises(FileExistsError):
        baseline.main()
    assert path.read_text() == '{"question_id": "old"}\n'

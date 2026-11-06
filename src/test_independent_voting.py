from decimal import Decimal
from copy import deepcopy
from unittest.mock import Mock
import json

import pytest
from src import independent_voting as voting
from src import single_agent_baseline as baseline
from src.test_single_agent_baseline import client_for, response, EXAMPLE


@pytest.mark.parametrize("answers, expected", [
    (["6", "6", "6"], Decimal("6")),
    (["6", "6.0", "7"], Decimal("6")),
    (["6", None, "6"], Decimal("6")),
    (["1", "2", "3"], None),
    (["1", "2", None], None),
    (["6", None, None], None),
    ([None, None, None], None),
    (["NaN", "NaN", "6"], None),
    (["bad", "Infinity", None], None),
])
def test_votes(answers, expected):
    assert voting.majority_vote(answers) == expected


def test_requires_three_answers():
    with pytest.raises(ValueError):
        voting.majority_vote(["6", "6"])


def test_independent_requests_and_wrong_majority():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    client = client_for()
    requests = []
    texts = iter(["FINAL ANSWER: 9", "FINAL ANSWER: 9.0", "unparseable"])
    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return response(next(texts), usage=usage)
    client.chat.completions.create.side_effect = create
    result = voting.solve_with_voting(client, "qwen3:4b", EXAMPLE, sample_seed=2)
    assert result["majority_answer"] == "9"
    assert not result["correct"]  # Reference 6 must never break or influence the vote.
    assert result["token_usage"]["total_tokens"] == 45
    assert len(result["agents"]) == 3
    assert [r["seed"] for r in requests] == [101, 102, 103]
    assert all(r["temperature"] == 0.7 and "tools" not in r for r in requests)
    assert all(r["model"] == "qwen3:4b" for r in requests)
    assert all(r["messages"] == requests[0]["messages"] for r in requests)
    assert len(requests[0]["messages"]) == 1
    assert "####" not in requests[0]["messages"][0]["content"]
    json.dumps(result)


def test_no_majority_and_missing_usage():
    client = client_for(response("FINAL ANSWER: 6"), response("FINAL ANSWER: 7"), response("bad"))
    result = voting.solve_with_voting(client, "qwen3:4b", EXAMPLE)
    assert result["majority_answer"] is None
    assert not result["correct"]
    assert result["token_usage"]["total_tokens"] is None


def test_baseline_defaults_omit_sampling_options():
    client = client_for(response())
    baseline.solve_question(client, "qwen3:4b", EXAMPLE)
    options = client.chat.completions.create.call_args.kwargs
    assert "temperature" not in options and "seed" not in options


def test_pilot_writes_results_and_protects_repeat(tmp_path, monkeypatch):
    path = tmp_path / "voting.jsonl"
    monkeypatch.setattr(voting, "RESULTS_FILE", path)
    monkeypatch.setattr(voting, "load_dotenv", Mock())
    monkeypatch.setattr(voting, "load_development_data", Mock(return_value=[EXAMPLE]))
    client = client_for(response(), response(), response())
    monkeypatch.setattr(voting, "OpenAI", Mock(return_value=client))
    voting.main()
    data = path.read_bytes()
    assert len(json.loads(data)["agents"]) == 3
    with pytest.raises(FileExistsError):
        voting.main()
    assert path.read_bytes() == data
    assert client.chat.completions.create.call_count == 3

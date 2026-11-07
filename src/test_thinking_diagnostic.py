"""Mocked tests for the thinking diagnostic: no live Ollama, no network."""
import json
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion

from src import thinking_diagnostic as diag
from src.test_single_agent_baseline import client_for

RAINFALL = {"question": "What is 2 times 3?", "answer": "#### 6", "source_index": 4415}
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
TOOL = [{"id": "call-1", "type": "function", "function": {
    "name": "calculate", "arguments": '{"a":2,"b":3,"operation":"multiply"}'}}]
OFF_CONDITIONS = ["reasoning_effort_none", "native_think_false", "no_think_soft_switch"]


def reply(content="FINAL ANSWER: 6", calls=None, finish="stop", **extra):
    """OpenAI-compatible response; extra message fields such as reasoning are kept."""
    return ChatCompletion(
        id="t", created=0, model="qwen3:4b", object="chat.completion", usage=USAGE,
        choices=[{"index": 0, "finish_reason": finish, "message": {
            "role": "assistant", "content": content, "tool_calls": calls, **extra}}])


def native_reply(content="FINAL ANSWER: 6", calls=None, done_reason="stop", thinking=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    if thinking is not None:
        message["thinking"] = thinking
    return {"model": "qwen3:4b", "message": message, "done": True, "done_reason": done_reason,
            "prompt_eval_count": 10, "eval_count": 5, "total_duration": 1, "eval_duration": 1}


NATIVE_TOOL = [{"function": {"name": "calculate",
                             "arguments": {"a": 2, "b": 3, "operation": "multiply"}}}]


def native_client(*replies):
    bodies = []
    queue = list(replies)

    def post(body):
        bodies.append(json.loads(json.dumps(body)))  # Must be JSON-serialisable.
        return queue.pop(0)

    return diag.NativeChatClient(think=False, post=post), bodies


def transport_for(condition, thinking=False):
    """Warm-up, tool-calling answer, post-tool answer, for any condition."""
    if diag.CONDITIONS[condition]["transport"] == "native":
        extra = "thinking about it" if thinking else None
        return native_client(native_reply("OK", thinking=extra),
                             native_reply("", NATIVE_TOOL, thinking=extra),
                             native_reply(thinking=extra))
    if thinking and condition == "no_think_soft_switch":
        replies = [reply("<think>hm</think>OK"), reply("<think>hm</think>", TOOL),
                   reply("<think>hm</think>FINAL ANSWER: 6")]
    else:
        extra = {"reasoning": "thinking about it"} if thinking else {}
        replies = [reply("OK", **extra), reply("", TOOL, **extra), reply(**extra)]
    client = client_for(*replies)
    return client, None


def sent_requests(client, bodies):
    if bodies is not None:
        return bodies
    return [call.kwargs for call in client.chat.completions.create.call_args_list]


@pytest.mark.parametrize("condition", list(diag.CONDITIONS))
def test_each_condition_same_settings_and_controls(condition):
    client, bodies = transport_for(condition)
    result = diag.run_condition(client, "qwen3:4b", RAINFALL, condition)
    requests = sent_requests(client, bodies)
    assert len(requests) == 3
    warm, answer, post_tool = requests
    native = bodies is not None
    for request in (answer, post_tool):
        if native:
            assert request["options"] == {"num_predict": 2048, "temperature": 0.7, "seed": 101}
            assert request["think"] is False
        else:
            assert (request["max_tokens"], request["temperature"], request["seed"]) == (2048, 0.7, 101)
            assert request.get("reasoning_effort") == (
                "none" if condition == "reasoning_effort_none" else None)
        user = [m for m in request["messages"] if m["role"] == "user"]
        assert user[0]["content"].endswith(" /no_think") == (condition == "no_think_soft_switch")
    # Warm-up uses the raised limit and the same control.
    assert (warm["options"]["num_predict"] if native else warm["max_tokens"]) == 2048
    assert result["warm_up"]["max_tokens"] == 2048
    assert result["warm_up"]["policy_detail"] == diag.WARM_UP_POLICY
    # Calculator handling is unchanged and the answer comes from final content.
    assert result["calculator_calls"] == 1
    assert result["calculator_log"][0]["result"] == "6"
    assert result["correct"] and result["model_answer"] == "6"
    assert [e["stage"] for e in result["request_log"]] == ["answer", "post_tool"]
    assert result["token_usage"] == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    assert result["status"] == ("no_thinking_observed" if condition == "default" else "control_honoured")
    for entry in result["request_log"]:
        assert not entry["thinking"]["thinking_present"]
    json.dumps(result)


def test_suffix_does_not_mutate_solver_messages():
    messages = [{"role": "user", "content": "Q"}, {"role": "tool", "content": "6", "tool_call_id": "c"}]
    assert diag.with_prompt_suffix(messages, " /no_think")[0]["content"] == "Q /no_think"
    assert messages[0]["content"] == "Q"


def test_default_logs_thinking_and_never_extracts_answer_from_it():
    thought = "FINAL ANSWER: 6 " * 50
    client = client_for(reply("OK"), reply("", finish="length", reasoning=thought))
    result = diag.run_condition(client, "qwen3:4b", RAINFALL, "default")
    entry = result["request_log"][0]
    assert result["status"] == "thinking_present"
    assert entry["finish_reason"] == "length" and entry["content_empty"]
    assert entry["thinking"]["thinking_fields"]["reasoning"] == {
        "present": True, "characters": len(thought.strip())}
    assert entry["thinking"]["final_content_characters"] == 0
    assert result["model_answer"] is None and not result["correct"]
    assert result["parsing_failure"] == "empty_content"
    assert diag.summarise(result)["thinking_characters"] == [len(thought.strip())]


def test_unterminated_inline_thinking_is_not_an_answer():
    client = client_for(reply("OK"), reply("<think>so FINAL ANSWER: 6", finish="length"))
    result = diag.run_condition(client, "qwen3:4b", RAINFALL, "default")
    inline = result["request_log"][0]["thinking"]["inline_think"]
    assert inline == {"tag_present": True, "characters": len("so FINAL ANSWER: 6"), "unterminated": True}
    assert result["request_log"][0]["content"] == ""
    assert result["model_answer"] is None


@pytest.mark.parametrize("condition", OFF_CONDITIONS)
def test_control_requested_but_thinking_still_present(condition):
    client, bodies = transport_for(condition, thinking=True)
    result = diag.run_condition(client, "qwen3:4b", RAINFALL, condition)
    assert result["status"] == "control_requested_but_thinking_present"
    assert result["warm_up"]["status"] == "control_requested_but_thinking_present"
    assert all(e["thinking"]["thinking_present"] for e in result["request_log"])
    # The answer still comes only from final content.
    assert result["model_answer"] == "6"
    observation = result["request_log"][0]["thinking"]
    if condition == "native_think_false":
        assert observation["thinking_fields"]["thinking"]["present"]
        assert observation["native"]["think_sent"] is False
    elif condition == "no_think_soft_switch":
        assert observation["inline_think"]["tag_present"]
    else:
        assert observation["control_sent"]["reasoning_effort"] == "none"
        assert observation["thinking_fields"]["reasoning"]["present"]


@pytest.mark.parametrize("content, final, characters, unterminated", [
    ("<think>abc</think>FINAL ANSWER: 1", "FINAL ANSWER: 1", 3, False),
    ("<think>\n\n</think>\n\nFINAL ANSWER: 1", "\n\nFINAL ANSWER: 1", 0, False),
    ("abc</think>FINAL ANSWER: 1", "FINAL ANSWER: 1", 3, False),
    ("FINAL ANSWER: 1<think>more", "FINAL ANSWER: 1", 4, True),
])
def test_split_inline_thinking(content, final, characters, unterminated):
    assert diag.split_inline_thinking(content) == (final, characters, True, unterminated)
    assert diag.split_inline_thinking("plain") == ("plain", 0, False, False)
    assert diag.split_inline_thinking(None) == (None, 0, False, False)


def test_native_adapter_round_trip_and_request_log_fields():
    client, bodies = native_client(native_reply("", NATIVE_TOOL), native_reply(done_reason="length"))
    observed = diag.ObservingClient(client, "native_think_false")
    result = diag.solve_question(observed, "qwen3:4b", RAINFALL, use_calculator=True,
                                 temperature=0.7, model_seed=101, max_tokens=2048)
    answer, post_tool = bodies
    assert answer["tools"][0]["function"]["name"] == "calculate"
    assert answer["stream"] is False
    sent_back = post_tool["messages"]
    assert sent_back[1]["tool_calls"][0]["function"]["arguments"] == {"a": 2, "b": 3, "operation": "multiply"}
    assert sent_back[2] == {"role": "tool", "content": "6", "tool_call_id": "native_call_1_0",
                            "tool_name": "calculate"}
    assert "tools" not in post_tool
    openai_result = diag.solve_question(client_for(reply()), "qwen3:4b", RAINFALL)
    assert set(result["request_log"][0]) == set(openai_result["request_log"][0])
    assert [e["finish_reason"] for e in result["request_log"]] == ["tool_calls", "length"]
    assert observed.observations[0]["native"]["tool_call_ids_synthesized"] == 1


def test_native_adapter_rejects_unsupported_options():
    client, _ = native_client()
    with pytest.raises(ValueError, match="reasoning_effort"):
        client.chat.completions.create(model="m", messages=[], reasoning_effort="none")
    with pytest.raises(ValueError, match="tool_choice"):
        client.chat.completions.create(model="m", messages=[], tool_choice="required")


def test_inspect_records_version_and_digest_without_github():
    show = {"capabilities": ["completion", "tools", "thinking"], "template": "{{ if .Think }}",
            "details": {"parameter_size": "4.0B"}, "license": "long text"}
    replies = {"/api/version": {"version": "0.12.0"},
               "/api/tags": {"models": [{"name": "qwen3:4b", "digest": "abc123"}]},
               "/api/show": show}
    fetch = Mock(side_effect=lambda path, body=None: replies[path])
    environment = diag.inspect_environment("qwen3:4b", fetch=fetch)
    assert environment["version"] == {"version": "0.12.0"}
    assert environment["model_identity"]["digest"] == "abc123"
    assert environment["hints"]["capabilities_include_thinking"]
    assert environment["hints"]["template_has_think_switch"]
    assert "license" not in environment["model_details"]
    assert [call.args[0] for call in fetch.call_args_list] == ["/api/version", "/api/tags", "/api/show"]


def test_existing_output_fails_clearly_without_touching_server(tmp_path, monkeypatch):
    path = tmp_path / "out.jsonl"
    path.write_text("old\n")
    monkeypatch.setattr(diag, "inspect_environment", Mock(side_effect=AssertionError("no server")))
    with pytest.raises(SystemExit, match="already exists; refusing to overwrite"):
        diag.main(["--run", "--output", str(path)])
    assert path.read_text() == "old\n"


def test_main_runs_all_conditions_and_records_errors(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.jsonl"
    environment = {"server": diag.SERVER, "version": {"version": "x"},
                   "model_identity": {"name": "qwen3:4b", "digest": "abc"}, "hints": {}}
    monkeypatch.setattr(diag, "inspect_environment", Mock(return_value=environment))
    monkeypatch.setattr(diag, "load_development_data", Mock(return_value=[RAINFALL]))

    def openai_create(**options):
        if options.get("reasoning_effort") == "none":
            raise RuntimeError("invalid reasoning_effort")
        return openai_queue.pop(0)

    openai_queue = [reply("OK"), reply("", TOOL), reply()] * 2  # default + /no_think
    openai_client = client_for()
    openai_client.chat.completions.create = Mock(side_effect=openai_create)
    monkeypatch.setattr(diag, "OpenAI", Mock(return_value=openai_client))
    native, _ = native_client(native_reply("OK"), native_reply("", NATIVE_TOOL), native_reply())
    monkeypatch.setattr(diag, "NativeChatClient", Mock(return_value=native))
    diag.main(["--run", "--output", str(path)])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in records] == [
        "environment", "settings", "condition", "condition_error", "condition", "condition", "summary"]
    assert records[3]["condition"] == "reasoning_effort_none"
    assert "invalid reasoning_effort" in records[3]["error"]
    statuses = {row["condition"]: row["status"] for row in records[-1]["rows"]}
    assert statuses == {"default": "no_thinking_observed", "reasoning_effort_none": "request_error",
                        "native_think_false": "control_honoured", "no_think_soft_switch": "control_honoured"}
    diag.load_development_data.assert_called_once_with(sample_size=3, seed=2)
    out = capsys.readouterr().out
    assert "Per-condition summary" in out
    assert "condition=native_think_false | status=control_honoured" in out


def test_inspect_mode_does_not_generate(tmp_path, monkeypatch):
    path = tmp_path / "inspect.jsonl"
    monkeypatch.setattr(diag, "inspect_environment", Mock(return_value={"version": {}, "model_identity": {},
                                                                       "hints": {}}))
    monkeypatch.setattr(diag, "OpenAI", Mock(side_effect=AssertionError("no generation")))
    diag.main(["--inspect", "--output", str(path)])
    assert json.loads(path.read_text())["type"] == "environment"

"""Rainfall thinking diagnostic. --inspect reads server/model metadata; --run generates.

Tests whether Qwen3 spends the completion budget on thinking, and whether each
thinking-off control is actually honoured. Support is judged empirically from
each response (separate reasoning field, or <think> text in content), never
from version numbers or source code.
"""
import argparse
import json
import os
import re
import sys
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai import OpenAI
from openai.types.chat import ChatCompletion

if __package__:
    from .single_agent_baseline import solve_question, warm_up_model
    from .load_data import load_development_data
else:
    from single_agent_baseline import solve_question, warm_up_model
    from load_data import load_development_data

SERVER = "http://localhost:11434"
REASONING_FIELDS = ("thinking", "reasoning", "reasoning_content")

# Identical for every condition, and for both the answer and post-tool requests.
QUESTION_INDEX = 4415          # gsm8k/train/4415, the rainfall question.
PILOT_SAMPLE = {"sample_size": 3, "seed": 2}
SETTINGS = {"use_calculator": True, "sample_seed": 2, "model_seed": 101,
            "temperature": 0.7, "max_tokens": 2048}
# The benchmark's 8-token warm-up would be cut off mid-thought and log a misleading
# finish_reason="length". Here the limit is raised instead (thinking stays as the
# condition sets it, so the warm-up also exercises the condition's control).
WARM_UP_MAX_TOKENS = 2048
WARM_UP_POLICY = "raised_limit_same_control_as_condition"

CONDITIONS = {
    "default": {
        "transport": "openai", "control": {},
        "description": "OpenAI-compatible /v1, no thinking control (server/model default)."},
    "reasoning_effort_none": {
        "transport": "openai", "control": {"reasoning_effort": "none"},
        "description": "OpenAI-compatible /v1 with reasoning_effort='none'."},
    "native_think_false": {
        "transport": "native", "control": {"think": False},
        "description": "Ollama native /api/chat with think=false."},
    "no_think_soft_switch": {
        "transport": "openai", "control": {"prompt_suffix": " /no_think"},
        "description": "OpenAI-compatible /v1 with Qwen3's /no_think appended to user messages."},
}


def fetch_json(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as reply:
        return json.load(reply)


def inspect_environment(model, fetch=fetch_json):
    """Record server version and the exact installed model. Hints are informational only."""
    version = fetch("/api/version")
    tags = fetch("/api/tags")
    # Older servers read "name", newer ones "model"; unknown keys are ignored.
    show = fetch("/api/show", {"model": model, "name": model})
    identities = [m for m in tags.get("models", [])
                  if m.get("name") == model or m.get("model") == model]
    if len(identities) != 1 or not identities[0].get("digest"):
        raise ValueError("Could not identify one installed model digest. Use its exact tag in MODEL_ID.")
    template = show.get("template", "") or ""
    capabilities = show.get("capabilities")
    return {
        "server": SERVER, "version": version, "model_identity": identities[0],
        "model_details": {key: show.get(key) for key in (
            "capabilities", "details", "parameters", "template", "modified_at")},
        "hints": {
            "capabilities_include_thinking": None if capabilities is None else "thinking" in capabilities,
            "template_has_think_switch": ".Think" in template,
            "template_mentions_no_think": "/no_think" in template,
            "note": "Hints only. Whether a control works is decided by the --run responses.",
        },
    }


def split_inline_thinking(content):
    """Separate <think> text from final content.

    Handles complete blocks, an unterminated block (budget ran out mid-thought), and
    a bare closing tag (template injected the opening <think> into the prompt).
    Returns (final_content, thinking_characters, tag_present, unterminated).
    """
    if not content or ("<think>" not in content and "</think>" not in content):
        return content, 0, False, False
    pieces = []
    text = content
    first_open, first_close = text.find("<think>"), text.find("</think>")
    if first_close != -1 and (first_open == -1 or first_close < first_open):
        pieces.append(text[:first_close])
        text = text[first_close + len("</think>"):]

    def remove(match):
        pieces.append(match.group(1))
        return ""

    text = re.sub(r"<think>(.*?)</think>", remove, text, flags=re.DOTALL)
    unterminated = "<think>" in text
    if unterminated:
        start = text.find("<think>")
        pieces.append(text[start + len("<think>"):])
        text = text[:start]
    return text, sum(len(piece.strip()) for piece in pieces), True, unterminated


class NativeRequestError(RuntimeError):
    def __init__(self, status_code, body):
        super().__init__(f"Ollama /api/chat returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def to_native_messages(messages):
    """Convert OpenAI-style chat messages (as solve_question builds them) for /api/chat."""
    tool_names = {}
    converted = []
    for message in messages:
        role = message["role"]
        item = {"role": role, "content": message.get("content") or ""}
        if role == "assistant":
            for field in REASONING_FIELDS:
                if message.get(field):
                    item["thinking"] = message[field]
                    break
            calls = []
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_names[call.get("id")] = function["name"]
                calls.append({"function": {"name": function["name"], "arguments": arguments}})
            if calls:
                item["tool_calls"] = calls
        elif role == "tool":
            item["tool_call_id"] = message.get("tool_call_id")
            if message.get("tool_call_id") in tool_names:
                item["tool_name"] = tool_names[message["tool_call_id"]]
        converted.append(item)
    return converted


class NativeChatClient:
    """Minimal adapter: /api/chat behind the client.chat.completions.create interface,
    returning ChatCompletion objects so request_model logs the same fields."""
    SUPPORTED = {"model", "messages", "max_tokens", "temperature", "seed", "tools", "tool_choice"}

    def __init__(self, server=SERVER, think=False, post=None, timeout=600):
        self.base_url = server + "/api/chat"
        self.think = think
        self.timeout = timeout
        self._post = post or self._http_post
        self._calls = 0
        self.last_native = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def _http_post(self, body):
        request = Request(self.base_url, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as reply:
                return json.load(reply)
        except HTTPError as exc:
            raise NativeRequestError(exc.code, exc.read().decode(errors="replace")) from exc

    def create(self, **options):
        unknown = set(options) - self.SUPPORTED
        if unknown:
            raise ValueError(f"Native adapter does not support options: {sorted(unknown)}")
        if options.get("tool_choice", "auto") != "auto":
            raise ValueError("Native /api/chat has no tool_choice; only 'auto' matches its behaviour.")
        body = {"model": options["model"], "messages": to_native_messages(options["messages"]),
                "stream": False, "think": self.think}
        native_options = {}
        for name, native_name in (("max_tokens", "num_predict"), ("temperature", "temperature"),
                                  ("seed", "seed")):
            if options.get(name) is not None:
                native_options[native_name] = options[name]
        if native_options:
            body["options"] = native_options
        if options.get("tools"):
            body["tools"] = options["tools"]
        self._calls += 1
        return self._to_completion(self._post(body), body)

    def _to_completion(self, reply, body):
        message = reply.get("message") or {}
        calls, synthesized = [], 0
        for number, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            call_id = call.get("id")
            if not call_id:
                call_id = f"native_call_{self._calls}_{number}"
                synthesized += 1
            calls.append({"id": call_id, "type": "function", "function": {
                "name": function.get("name"),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}})
        done_reason = reply.get("done_reason")
        finish = "length" if done_reason == "length" else "tool_calls" if calls else "stop"
        prompt_count, eval_count = reply.get("prompt_eval_count"), reply.get("eval_count")
        usage = None
        if prompt_count is not None and eval_count is not None:
            usage = {"prompt_tokens": prompt_count, "completion_tokens": eval_count,
                     "total_tokens": prompt_count + eval_count}
        out_message = {"role": "assistant", "content": message.get("content", ""),
                       "tool_calls": calls or None}
        if "thinking" in message:
            out_message["thinking"] = message["thinking"]
        self.last_native = {
            "endpoint": "/api/chat", "think_sent": body["think"], "options_sent": body.get("options"),
            "done_reason": done_reason, "message_keys": list(message),
            "prompt_eval_count": prompt_count, "eval_count": eval_count,
            "total_duration_ns": reply.get("total_duration"), "eval_duration_ns": reply.get("eval_duration"),
            "tool_call_ids_synthesized": synthesized,
        }
        return ChatCompletion(id=f"native-{self._calls}", created=0, object="chat.completion",
                              model=reply.get("model", body["model"]), usage=usage,
                              choices=[{"index": 0, "finish_reason": finish, "message": out_message}])


def with_prompt_suffix(messages, suffix):
    return [{**m, "content": m["content"] + suffix}
            if m.get("role") == "user" and isinstance(m.get("content"), str) else m
            for m in messages]


class ObservingClient:
    """Apply one condition's control to every request and record thinking separately.

    Thinking text is removed from content before solve_question sees it, so answers are
    only ever extracted from final content. Only character counts of thinking are kept.
    """
    def __init__(self, inner, condition):
        self.inner = inner
        self.condition = condition
        self.control = CONDITIONS[condition]["control"]
        self.base_url = inner.base_url
        self.observations = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **options):
        options = dict(options)
        if "reasoning_effort" in self.control:
            options["reasoning_effort"] = self.control["reasoning_effort"]
        if "prompt_suffix" in self.control:
            options["messages"] = with_prompt_suffix(options["messages"], self.control["prompt_suffix"])
        response = self.inner.chat.completions.create(**options)
        choice = response.choices[0]
        message = choice.message
        raw = message.model_dump(exclude_unset=True)
        fields = {}
        for name in REASONING_FIELDS:
            value = raw.get(name)
            fields[name] = {"present": value is not None,
                            "characters": len(value.strip()) if isinstance(value, str) else 0}
        final, inline_characters, tag_present, unterminated = split_inline_thinking(message.content)
        if tag_present:
            message.content = final
        thinking_characters = inline_characters + sum(f["characters"] for f in fields.values())
        self.observations.append({
            "condition": self.condition,
            "control_sent": {"reasoning_effort": options.get("reasoning_effort"),
                             "prompt_suffix": self.control.get("prompt_suffix"),
                             "native_think": getattr(self.inner, "think", None)
                             if CONDITIONS[self.condition]["transport"] == "native" else None},
            "max_tokens_sent": options.get("max_tokens"),
            "finish_reason": choice.finish_reason,
            "message_keys": list(raw),
            "thinking_fields": fields,
            "inline_think": {"tag_present": tag_present, "characters": inline_characters,
                             "unterminated": unterminated},
            "thinking_present": thinking_characters > 0,
            "thinking_characters": thinking_characters,
            "final_content_characters": len((message.content or "").strip()),
            "final_content_empty": not (message.content or "").strip(),
            "usage_raw": response.usage.model_dump() if response.usage else None,
            "response_model": response.model,
            "native": getattr(self.inner, "last_native", None),
        })
        return response


def attach_thinking(request_log, observations):
    if len(request_log) != len(observations):
        raise RuntimeError("Request log and thinking observations are out of step.")
    for entry, observation in zip(request_log, observations):
        entry["thinking"] = observation


def condition_status(condition, request_log):
    seen = any(entry["thinking"]["thinking_present"] for entry in request_log)
    if condition == "default":
        return "thinking_present" if seen else "no_thinking_observed"
    return "control_requested_but_thinking_present" if seen else "control_honoured"


def run_condition(transport, model, example, condition):
    observed = ObservingClient(transport, condition)
    warm = warm_up_model(observed, model, max_tokens=WARM_UP_MAX_TOKENS)
    attach_thinking(warm["requests"], observed.observations)
    warm["policy_detail"] = WARM_UP_POLICY
    warm["status"] = condition_status(condition, warm["requests"])
    observed.observations = []
    result = solve_question(observed, model, example, use_calculator=SETTINGS["use_calculator"],
                            sample_seed=SETTINGS["sample_seed"], temperature=SETTINGS["temperature"],
                            model_seed=SETTINGS["model_seed"], max_tokens=SETTINGS["max_tokens"])
    attach_thinking(result["request_log"], observed.observations)
    result.update({
        "condition": condition, "description": CONDITIONS[condition]["description"],
        "transport": CONDITIONS[condition]["transport"], "control": CONDITIONS[condition]["control"],
        "status": condition_status(condition, result["request_log"]),
        "warm_up": warm,
    })
    return result


def summarise(result):
    log = result["request_log"]
    return {
        "condition": result["condition"], "status": result["status"],
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
        "prompt_tokens": result["token_usage"]["prompt_tokens"],
        "completion_tokens": result["token_usage"]["completion_tokens"],
        "finish_reasons": [entry["finish_reason"] for entry in log],
        "thinking_characters": [entry["thinking"]["thinking_characters"] for entry in log],
        "final_answer": result["model_answer"], "correct_answer": result["correct_answer"],
        "correct": result["correct"], "calculator_calls": result["calculator_calls"],
        "warm_up_status": result["warm_up"]["status"],
    }


def error_summary(condition, exc):
    return {"condition": condition, "status": "request_error",
            "error_type": type(exc).__name__, "error": str(exc),
            "status_code": getattr(exc, "status_code", None)}


def print_table(rows):
    columns = ("condition", "status", "elapsed_seconds", "prompt_tokens", "completion_tokens",
               "finish_reasons", "thinking_characters", "final_answer", "correct", "calculator_calls")
    print("\nPer-condition summary (answer + post-tool requests; warm-up excluded):")
    for row in rows:
        print("  " + " | ".join(f"{name}={row.get(name, '-')}" for name in columns), flush=True)


def make_transport(condition, openai_client, server=SERVER):
    if CONDITIONS[condition]["transport"] == "native":
        return NativeChatClient(server, think=CONDITIONS[condition]["control"]["think"])
    return openai_client


def load_rainfall_example():
    examples = load_development_data(**PILOT_SAMPLE)
    matches = [row for row in examples if row["source_index"] == QUESTION_INDEX]
    if len(matches) != 1:
        raise ValueError(f"Expected gsm8k/train/{QUESTION_INDEX} in the fixed pilot sample.")
    return matches[0]


def parse_conditions(text):
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONDITIONS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"Unknown condition(s) {unknown}; choose from {list(CONDITIONS)}")
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", type=parse_conditions, default=list(CONDITIONS),
                        help="Comma-separated subset of: " + ", ".join(CONDITIONS))
    args = parser.parse_args(argv)
    model = os.getenv("MODEL_ID", "qwen3:4b")
    try:
        output = open(args.output, "x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"Output file {args.output!r} already exists; refusing to overwrite it. "
                         "Pass a new --output path.") from None
    with output:
        environment = inspect_environment(model)
        output.write(json.dumps({"type": "environment", **environment}) + "\n")
        output.flush()
        print(json.dumps({"version": environment["version"],
                          "model_identity": environment["model_identity"],
                          "hints": environment["hints"]}), flush=True)
        if args.inspect:
            return
        example = load_rainfall_example()
        settings = {**SETTINGS, "question_id": f"gsm8k/train/{QUESTION_INDEX}",
                    "warm_up_max_tokens": WARM_UP_MAX_TOKENS, "warm_up_policy": WARM_UP_POLICY,
                    "conditions": args.conditions}
        output.write(json.dumps({"type": "settings", **settings}) + "\n")
        openai_client = OpenAI(base_url=SERVER + "/v1", api_key="ollama", max_retries=0, timeout=600)
        rows = []
        for condition in args.conditions:
            print(f"Running condition {condition} ...", flush=True)
            try:
                result = run_condition(make_transport(condition, openai_client), model, example, condition)
            except Exception as exc:  # One unsupported control must not hide the others.
                row = error_summary(condition, exc)
                output.write(json.dumps({"type": "condition_error", **row}) + "\n")
            else:
                row = summarise(result)
                output.write(json.dumps({"type": "condition", **result}) + "\n")
            output.flush()
            print(json.dumps(row), flush=True)
            rows.append(row)
        output.write(json.dumps({"type": "summary", "rows": rows}) + "\n")
        print_table(rows)


if __name__ == "__main__":
    main(sys.argv[1:])

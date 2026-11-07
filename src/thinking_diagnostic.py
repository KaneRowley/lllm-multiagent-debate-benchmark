"""Version-aware rainfall diagnostic. --inspect reads metadata; --run generates."""
import argparse
import hashlib
import json
import os
import re
from types import SimpleNamespace
from urllib.request import Request, urlopen

from openai import OpenAI

if __package__:
    from .single_agent_baseline import solve_question, warm_up_model
    from .load_data import load_development_data
else:
    from single_agent_baseline import solve_question, warm_up_model
    from load_data import load_development_data

SERVER = "http://localhost:11434"
REASONING_FIELDS = ("thinking", "reasoning", "reasoning_content")


def fetch_json(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as reply:
        return json.load(reply)


def assess_support(show, source):
    """Conservative recognition of the installed release's explicit off mapping."""
    mapping = re.search(r'case "none":\s*return &api\.ThinkValue\{Value: false\}', source)
    consumes_field = 'json:"reasoning_effort' in source and "r.ReasoningEffort" in source
    forwards_think = re.search(r'Think:\s*think', source) is not None
    values = show.get("thinking", {}).get("values", [])
    # Older Qwen3 templates declare the boolean switch without show metadata.
    template_switch = ".Think" in show.get("template", "")
    model_off = any(value is False for value in values) or (
        not values and "thinking" in show.get("capabilities", []) and template_switch)
    supported = bool(mapping and consumes_field and forwards_think and model_off)
    return {
        "disabled_condition_available": supported,
        "control": {"reasoning_effort": "none"} if supported else None,
        "api_off_mapping_found": bool(mapping and consumes_field and forwards_think),
        "model_off_evidence": model_off,
        "status": "source_and_model_evidence" if supported else "unverified_skip_disabled",
        "note": "HTTP success alone cannot prove a control is honoured. Inspect returned reasoning fields too.",
    }


def inspect_environment(model):
    version = fetch_json("/api/version")
    tags = fetch_json("/api/tags")
    show = fetch_json("/api/show", {"model": model})
    identities = [m for m in tags.get("models", [])
                  if m.get("name") == model or m.get("model") == model]
    if len(identities) != 1 or not identities[0].get("digest"):
        raise ValueError("Could not identify one installed model digest. Use its exact tag in MODEL_ID.")
    release = version.get("version", "")
    source = ""
    source_url = None
    source_error = None
    if re.fullmatch(r"\d+\.\d+\.\d+", release):
        source_url = f"https://raw.githubusercontent.com/ollama/ollama/v{release}/openai/openai.go"
        try:
            with urlopen(source_url, timeout=30) as reply:
                source = reply.read().decode()
        except Exception as exc:
            source_error = str(exc)
    else:
        source_error = "Non-release version: automatic source matching disabled."
    return {"server": SERVER, "api": "/v1/chat/completions", "version": version,
            "model_identity": identities[0], "model_details": show,
            "source_url": source_url, "source_error": source_error,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
            "support": assess_support(show, source)}


class DiagnosticClient:
    """Wrap the existing solver's transport so both answer and post-tool calls match."""
    def __init__(self, client, disabled=False):
        self.client = client
        self.disabled = disabled
        self.base_url = client.base_url
        self.observations = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **options):
        if self.disabled:
            options["reasoning_effort"] = "none"
        response = self.client.chat.completions.create(**options)
        message = response.choices[0].message
        raw = message.model_dump(exclude_unset=True)
        fields = {}
        for name in REASONING_FIELDS:
            value = raw.get(name)
            fields[name] = {"present": name in raw,
                            "characters": len(value) if isinstance(value, str) else None,
                            "type": type(value).__name__ if name in raw else None}
        self.observations.append({
            "requested_mode": "disabled" if self.disabled else "default",
            "reasoning_effort_sent": options.get("reasoning_effort"),
            "message_keys": list(raw), "reasoning_fields": fields,
            "final_content_characters": len(message.content or ""),
            "final_content_empty": not (message.content or "").strip(),
            "usage_raw": response.usage.model_dump() if response.usage else None,
            "response_model": response.model,
            "inline_think_present": "<think>" in (message.content or ""),
        })
        # Never let an unfinished inline thinking block supply a diagnostic answer.
        if "<think>" in (message.content or ""):
            message.content = re.sub(r"<think>.*?(?:</think>|$)", "", message.content, flags=re.DOTALL)
        return response


def run_condition(client, model, example, disabled=False):
    diagnostic = DiagnosticClient(client, disabled)
    warm = warm_up_model(diagnostic, model)
    warm["observations"] = diagnostic.observations[:]
    diagnostic.observations.clear()
    result = solve_question(diagnostic, model, example, use_calculator=True,
                            sample_seed=2, temperature=0.7, model_seed=101, max_tokens=2048)
    result["condition"] = "disabled" if disabled else "default"
    result["warm_up"] = warm
    result["thinking_observations"] = diagnostic.observations
    result["disabled_mode_contradicted"] = disabled and any(
        observation["inline_think_present"] or any(
            (field["characters"] or 0) > 0 for field in observation["reasoning_fields"].values())
        for observation in diagnostic.observations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = os.getenv("MODEL_ID", "qwen3:4b")
    with open(args.output, "x", encoding="utf-8") as output:
        environment = inspect_environment(model)
        output.write(json.dumps({"type": "environment", **environment}) + "\n")
        output.flush()
        print(json.dumps({"version": environment["version"],
                          "model_identity": environment["model_identity"],
                          "support": environment["support"]}), flush=True)
        if args.inspect:
            return
        examples = load_development_data(sample_size=3, seed=2)
        matches = [row for row in examples if row["source_index"] == 4415]
        if len(matches) != 1:
            raise ValueError("Expected rainfall question gsm8k/train/4415 in the fixed pilot sample.")
        client = OpenAI(base_url=SERVER + "/v1", api_key="ollama", max_retries=0)
        conditions = [False]
        if environment["support"]["disabled_condition_available"]:
            conditions.append(True)
        else:
            skipped = {"type": "skipped", "condition": "disabled",
                       "reason": "Installed API/model support could not be verified; inspect metadata/source."}
            output.write(json.dumps(skipped) + "\n")
            print(json.dumps(skipped), flush=True)
        for disabled in conditions:
            try:
                result = run_condition(client, model, matches[0], disabled)
            except Exception as exc:
                output.write(json.dumps({"type": "error", "disabled": disabled, "error": str(exc)}) + "\n")
                output.flush()
                raise
            output.write(json.dumps({"type": "condition", **result}) + "\n")
            output.flush()
            print(json.dumps({key: result[key] for key in (
                "condition", "elapsed_seconds", "token_usage", "model_answer", "calculator_calls",
                "request_log", "thinking_observations", "disabled_mode_contradicted")}), flush=True)


if __name__ == "__main__":
    main()

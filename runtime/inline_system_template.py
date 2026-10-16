#!/usr/bin/env python3
"""Write a copy of the model's chat template that accepts non-leading system messages; print its path.

Qwen3.5/3.8 templates raise "System message must be at the beginning." for any later system message. vLLM's Anthropic
endpoint (/v1/messages, e.g. Claude Code) probes the template with [system, user, system, user]; when it raises, vLLM
merges every inline system message (Claude Code's per-turn reminders) into the leading system prompt. That changes the
prompt prefix on every turn, so the whole conversation after the system prompt misses the prefix cache and is
re-prefilled each turn (vLLM logs a warning saying exactly this). Rendering inline system messages in place keeps the
prefix stable; conversations without inline system messages render byte-identically.

Also maps the OpenAI-style reasoning_effort values the template rejects: 'high' -> 'xhigh', 'minimal' -> 'low'. The stock
template only accepts xhigh/medium/low and returns HTTP 400 for 'high' (Hermes' default; seen in a Hermes request dump).

usage: inline_system_template.py <model_dir> <out_path>   -> prints out_path, or nothing (exit 0) if not applicable
"""
import json
import pathlib
import sys

OLD = """    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
"""
NEW = """    {%- if message.role == "system" %}
        {#- B70: non-leading system messages (e.g. Claude Code per-turn reminders via /v1/messages) render in place
            instead of raising, so vLLM keeps them in place and the prompt prefix stays cacheable. #}
        {%- if not loop.first %}
            {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}
        {%- endif %}
"""

EFFORT_OLD = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
EFFORT_NEW = (
    "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
    "{#- B70: accept the OpenAI-style values clients send (Hermes defaults to 'high') instead of HTTP 400 #}\n"
    "{%- if resolved_reasoning_effort == 'high' %}{%- set resolved_reasoning_effort = 'xhigh' %}\n"
    "{%- elif resolved_reasoning_effort == 'minimal' %}{%- set resolved_reasoning_effort = 'low' %}{%- endif %}\n"
)


def main() -> int:
    model, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    src = None
    if (model / "chat_template.jinja").is_file():
        src = (model / "chat_template.jinja").read_text()
    elif (model / "tokenizer_config.json").is_file():
        tpl = json.loads((model / "tokenizer_config.json").read_text()).get("chat_template")
        src = tpl if isinstance(tpl, str) else None
    if not src:
        return 0
    new = src.replace(OLD, NEW, 1) if src.count(OLD) == 1 else src
    if src.count(EFFORT_OLD) == 1:
        new = new.replace(EFFORT_OLD, EFFORT_NEW, 1)
    if new == src:
        print("[B70] chat template fixes: anchors not found, keeping the model's template", file=sys.stderr)
        return 0
    out.write_text(new)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

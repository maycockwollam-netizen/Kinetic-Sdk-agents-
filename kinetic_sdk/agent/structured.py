"""Structured output: force the final answer to match a JSON schema.

Models default to free-form prose, which is useless for programmatic callers
(a planner emitting a plan, a reviewer emitting a verdict object, ...).
:class:`~kinetic_sdk.agent.agent.Agent.run` accepts ``output_schema``; when
given, the final assistant text is parsed as JSON and validated against that
schema (same zero-dependency JSON-Schema subset as tool-input validation in
``tool/validation.py``).

Design decisions (do not "fix" — they are deliberate):

* **The return type of ``run()`` stays ``str``.** The parsed value is exposed
  on ``agent.structured_output`` instead, so every existing caller keeps
  working and text observers (logs, chat UIs) still get the raw final text.
* **Correction happens through the conversation, not hidden retries.** When
  the answer fails to parse/validate, ONE user message naming the validation
  errors is appended and the loop continues — the model sees its mistake and
  fixes it, exactly like a tool error. ``structured_retries`` bounds this.
* **No extra tool is injected** (unlike OpenAI's ``response_format`` hacks
  that turn the schema into a pseudo-tool). The schema is only ever stated
  in natural language plus validated locally — provider-neutral by
  construction, works with models that have no native JSON mode.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kinetic_sdk.tool.validation import validate_value

logger = logging.getLogger(__name__)

#: Default number of correction rounds allowed after a first malformed
#: answer. Two is enough for a well-behaved model; more rounds mostly feed
#: a stuck model.
DEFAULT_STRUCTURED_RETRIES = 2

#: Instruction appended to the task when ``output_schema`` is provided. The
#: schema is embedded verbatim (JSON) so the model can echo it back.
INSTRUCTION_TEMPLATE = (
    "Your final answer MUST be a single JSON value matching this JSON Schema:\n"
    "{schema_json}\n"
    "Reply with ONLY the JSON — no prose, no markdown fences."
)

#: Message appended when a failed answer is returned to the model. The
#: problems list is inserted verbatim so the model can fix specific fields.
CORRECTION_TEMPLATE = (
    "Your previous reply did not satisfy the required JSON schema. "
    "Problems: {problems}. Reply with ONLY the corrected JSON value."
)


def schema_instruction(schema: dict[str, Any]) -> str:
    """Build the natural-language instruction carrying *schema*."""
    try:
        schema_json = json.dumps(schema, ensure_ascii=False)
    except (TypeError, ValueError):
        schema_json = str(schema)
    return INSTRUCTION_TEMPLATE.format(schema_json=schema_json)


def parse_structured(text: str) -> tuple[bool, Any]:
    """Extract a JSON value from *text*.

    Returns ``(True, value)`` on success, ``(False, None)`` otherwise. Two
    passes: the whole text first (the well-behaved case), then a scan for
    the first ``{`` / ``[`` decoded with ``raw_decode`` so a model that
    prepends prose ("Here is the JSON: {...}") still passes — anything
    after the first complete value is ignored.
    """
    stripped = text.strip()
    if not stripped:
        return False, None
    try:
        return True, json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for i, ch in enumerate(stripped):
        if ch in "{[":
            try:
                value, _ = json.JSONDecoder().raw_decode(stripped[i:])
                return True, value
            except json.JSONDecodeError:
                continue
    return False, None


def validate_structured(schema: dict[str, Any], value: Any) -> list[str]:
    """Validate a parsed value against *schema*; empty list means valid.

    Thin wrapper over :func:`kinetic_sdk.tool.validation.validate_value`
    rooted at ``$`` so error messages point at the offending path.
    """
    return validate_value(schema, value, path="$")


def check_final_answer(
    schema: dict[str, Any], text: str
) -> tuple[bool, Any, list[str]]:
    """Parse + validate a final answer against *schema*.

    Returns ``(ok, value, problems)``: ``ok`` with the parsed value on
    success, else the human-readable problems to feed back to the model.
    """
    parsed_ok, value = parse_structured(text)
    if not parsed_ok:
        return False, None, ["final answer is not valid JSON"]
    problems = validate_structured(schema, value)
    if problems:
        return False, None, problems
    return True, value, []

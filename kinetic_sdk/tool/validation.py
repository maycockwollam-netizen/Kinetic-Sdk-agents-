"""JSON-Schema validation for tool inputs (zero-dependency subset).

Models hallucinate: they rename parameters, drop required ones, or pass a
string where the schema wants an object. Without validation that surfaces as
a confusing ``TypeError`` deep inside ``tool.execute`` — or worse, silently
wrong behaviour. The agent loop validates the arguments against the tool's
declared ``parameters`` schema BEFORE executing, so the model gets a clear,
actionable error message it can correct on the next turn.

This is a deliberate SUBSET of JSON Schema — the shapes tool schemas actually
use: ``type`` (object/array/string/number/integer/boolean/null),
``properties``, ``required``, ``items``, ``enum``, and
``additionalProperties``. No ``$ref``, no combinators (``anyOf``/``oneOf`` —
unknown keywords are ignored, never an error). Strictness choice: unexpected
top-level properties are REJECTED unless the schema allows them
(``additionalProperties: true`` or no ``properties`` declared at all), because
an extra kwarg would otherwise crash ``execute(**params)``.
"""

from __future__ import annotations

from typing import Any


def validate_tool_input(schema: dict[str, Any], params: Any) -> list[str]:
    """Validate *params* against a tool's ``parameters`` schema.

    Returns a list of human-readable problems; empty means valid.
    """
    if not isinstance(schema, dict) or not schema:
        return []
    if schema.get("type", "object") != "object":
        # Tool parameters are objects by convention; exotic roots are the
        # tool's own responsibility.
        return []
    if not isinstance(params, dict):
        return [f"expected an object of arguments, got {_json_type(params)}"]

    errors: list[str] = []
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        properties = {}
    additional = schema.get("additionalProperties", False)

    for name in schema.get("required") or []:
        if name not in params:
            errors.append(f"missing required parameter {name!r}")

    for key, value in params.items():
        if key not in properties:
            if properties and additional is not True:
                errors.append(
                    f"unexpected parameter {key!r} (declared: {sorted(properties)})"
                )
            continue
        errors.extend(_validate_value(value, properties[key], path=key))
    return errors


def validate_value(schema: Any, value: Any, path: str = "") -> list[str]:
    """Public generic-value validation (structured output, nested checks).

    Same engine as :func:`validate_tool_input`'s per-value checks, but
    callable directly on any JSON value (e.g. the parsed final answer when
    ``Agent.run(output_schema=...)`` is in play). Returns problems; empty
    means valid.
    """
    return _validate_value(value, schema, path or "$")


def _validate_value(value: Any, schema: Any, path: str) -> list[str]:
    if not isinstance(schema, dict) or not schema:
        return []
    errors: list[str] = []

    if "enum" in schema:
        if value not in schema["enum"]:
            return [f"{path}: {value!r} not in enum {schema['enum']!r}"]

    expected = schema.get("type")
    if expected is None:
        return errors
    if not _type_matches(value, expected):
        return [f"{path}: expected {_describe(expected)}, got {_json_type(value)}"]

    if expected == "object" and isinstance(value, dict):
        nested = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                errors.append(f"{path}.{req}: missing required property")
        nested_additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in nested:
                errors.extend(_validate_value(item, nested[key], f"{path}.{key}"))
            elif nested_additional is False:
                errors.append(f"{path}.{key}: unexpected property")

    if expected == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict) and item_schema:
            for i, item in enumerate(value):
                errors.extend(_validate_value(item, item_schema, f"{path}[{i}]"))

    return errors


def _type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, e) for e in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True  # unknown type keyword: don't reject what we don't understand


def _describe(expected: Any) -> str:
    return " | ".join(expected) if isinstance(expected, list) else str(expected)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__

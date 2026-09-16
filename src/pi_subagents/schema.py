"""Minimal JSON-Schema subset validation for structured agent replies.

Supports the subset that matters for agent output schemas: type (object,
array, string, number, integer, boolean, null), required, properties, items,
enum, and additionalProperties:false on objects. jsonschema is used when
available; otherwise this validator runs.
"""

from __future__ import annotations


class SchemaValidationError(Exception):
	pass


def validate_with_schema(schema: dict, value) -> None:
	"""Raise SchemaValidationError unless value matches schema (subset)."""
	try:
		import jsonschema  # type: ignore

		jsonschema.validate(instance=value, schema=schema)
		return
	except ImportError:
		pass
	except Exception as error:
		raise SchemaValidationError(str(error)) from error
	_validate(schema, value, "$")


def validate_schema(schema: dict) -> None:
	"""Light static check of the schema itself (self-contradictions)."""
	if not isinstance(schema, dict):
		raise SchemaValidationError("schema must be an object")
	known = {"type", "required", "properties", "items", "enum", "additionalProperties"}
	unknown = set(schema) - known
	if unknown:
		raise SchemaValidationError(f"unsupported schema keys: {sorted(unknown)}")


def _validate(schema: dict, value, path: str) -> None:
	stype = schema.get("type")
	if stype is not None:
		checkers = {
			"object": lambda v: isinstance(v, dict),
			"array": lambda v: isinstance(v, list),
			"string": lambda v: isinstance(v, str),
			"number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
			"integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
			"boolean": lambda v: isinstance(v, bool),
			"null": lambda v: v is None,
		}
		types = stype if isinstance(stype, list) else [stype]
		if not any(checker(t)(value) for t, checker in ((t, checkers.get(t, lambda v: False)) for t in types) if t in checkers):
			raise SchemaValidationError(f"{path}: expected type {stype!r}")
	if "enum" in schema and value not in schema["enum"]:
		raise SchemaValidationError(f"{path}: not in enum {schema['enum']!r}")
	if isinstance(value, dict):
		required = schema.get("required") or []
		for key in required:
			if key not in value:
				raise SchemaValidationError(f"{path}: missing required key {key!r}")
		props = schema.get("properties") or {}
		if schema.get("additionalProperties") is False:
			for key in value:
				if key not in props:
					raise SchemaValidationError(f"{path}: unexpected key {key!r}")
		for key, subschema in props.items():
			if key in value:
				_validate(subschema, value[key], f"{path}.{key}")
	if isinstance(value, list) and "items" in schema:
		for index, item in enumerate(value):
			_validate(schema["items"], item, f"{path}[{index}]")

"""Robust output parser for extracting tool calls from LLM output."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable

from agent.response import ToolCall


@dataclass
class ParsedTool:
    """Intermediate representation of a parsed tool call."""

    name: str
    args: dict


class OutputParser:
    """Extract tool calls from raw LLM output with multiple strategies."""

    def __init__(self, log_dir: str):
        self._log_path = os.path.join(log_dir, "output_parser.log")
        self._logger = self._build_logger(log_dir)

    def extract_tool_call(
        self,
        raw_text: str,
        registered_tools: Iterable[str],
        tool_schemas: dict[str, dict] | None = None,
    ) -> ToolCall | None:
        """Extract the first valid tool call from raw text."""
        tool_calls = self.extract_tool_calls(raw_text, registered_tools, tool_schemas)
        return tool_calls[0] if tool_calls else None

    def extract_tool_calls(
        self,
        raw_text: str,
        registered_tools: Iterable[str],
        tool_schemas: dict[str, dict] | None = None,
    ) -> list[ToolCall]:
        """Extract all valid tool calls from raw text."""
        registered = {t.lower(): t for t in registered_tools}
        schemas = tool_schemas or {}
        errors: list[str] = []
        results: list[ToolCall] = []

        strategies = [
            ("code_fence", self._extract_from_code_fence),
            ("raw_json", self._extract_from_raw_json),
            ("bracket_match", self._extract_with_bracket_matching),
            ("repair", self._extract_with_repair),
        ]

        for strategy_name, strategy in strategies:
            payloads = strategy(raw_text)
            if not payloads:
                continue
            for payload in payloads:
                parsed_tools, error = self._parse_payload(payload)
                if error:
                    errors.append(f"{strategy_name}: {error}")
                    continue
                for parsed in parsed_tools:
                    tool_call, error = self._validate_tool_call(parsed, registered, schemas)
                    if tool_call:
                        results.append(tool_call)
                    elif error:
                        errors.append(f"{strategy_name}: {error}")
            if results:
                return results

        self._log_failure(raw_text, errors)
        return []

    def _parse_payload(self, text: str) -> tuple[list[ParsedTool], str | None]:
        data, error = self._parse_json(text)
        if error:
            return [], error

        parsed_tools: list[ParsedTool] = []
        if isinstance(data, list):
            for item in data:
                parsed, err = self._normalize_payload(item)
                if parsed:
                    parsed_tools.append(parsed)
                elif err:
                    return [], err
        else:
            parsed, err = self._normalize_payload(data)
            if parsed:
                parsed_tools.append(parsed)
            elif err:
                return [], err
        return parsed_tools, None

    def _parse_json(self, text: str) -> tuple[object | None, str | None]:
        """Parse JSON text, attempting repair when needed."""
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            repaired = self._repair_json(text)
            if repaired == text:
                return None, "invalid JSON"
            try:
                return json.loads(repaired), None
            except json.JSONDecodeError:
                return None, "invalid JSON after repair"

    def _normalize_payload(self, data: object) -> tuple[ParsedTool | None, str | None]:
        """Normalize parsed JSON into a tool payload."""
        if not isinstance(data, dict):
            return None, "JSON payload is not an object"

        payload = data

        if "tool_name" not in payload:
            for alt in ("tool", "name"):
                if alt in payload:
                    payload["tool_name"] = payload[alt]
                    break

        if "tool_args" not in payload:
            for alt in ("args", "arguments"):
                if alt in payload:
                    payload["tool_args"] = payload[alt]
                    break

        tool_name = str(payload.get("tool_name", "")).strip()
        if not tool_name:
            return None, "Missing tool_name"

        args = payload.get("tool_args", {})
        if not isinstance(args, dict):
            args = {}

        args = self._normalize_args(tool_name, args)
        return ParsedTool(name=tool_name, args=args), None

    def _validate_tool_call(
        self,
        parsed: ParsedTool,
        registered: dict[str, str],
        schemas: dict[str, dict],
    ) -> tuple[ToolCall | None, str | None]:
        tool_name = parsed.name.strip()
        tool_key = tool_name.lower()
        if tool_key not in registered:
            return None, f"Unknown tool '{tool_name}'"

        canonical_name = registered[tool_key]
        args = parsed.args
        schema = schemas.get(canonical_name, {})
        required = schema.get("required_args", [])
        arg_schema = schema.get("arg_schema", {})

        missing = [
            key for key in required
            if key not in args or args.get(key) in (None, "")
        ]
        if missing:
            return None, f"Missing required args: {', '.join(missing)}"

        for key, expected in arg_schema.items():
            if key not in args:
                continue
            value = args[key]
            coerced, ok = self._coerce_value(value, expected)
            if not ok:
                return None, f"Invalid type for '{key}'"
            args[key] = coerced

        return ToolCall(name=canonical_name, args=args), None

    def _coerce_value(self, value: object, expected: type | tuple[type, ...]) -> tuple[object, bool]:
        if isinstance(expected, tuple):
            expected_types = expected
        else:
            expected_types = (expected,)

        if value is None and type(None) in expected_types:
            return value, True

        if str in expected_types and not isinstance(value, str):
            return str(value), True

        if (int in expected_types or float in expected_types) and isinstance(value, str):
            try:
                if int in expected_types and value.isdigit():
                    return int(value), True
                if float in expected_types:
                    return float(value), True
            except ValueError:
                return value, False

        if isinstance(value, expected_types):
            return value, True

        return value, False

    def _normalize_args(self, tool_name: str, args: dict) -> dict:
        """Apply tool-specific arg normalization for common mistakes."""
        if tool_name in ("response", "task_done"):
            if "text" not in args:
                for alt in ("message", "content", "answer"):
                    if alt in args:
                        args["text"] = args[alt]
                        break
        if tool_name == "code_execution" and "code" not in args:
            for alt in ("script", "command"):
                if alt in args:
                    args["code"] = args[alt]
                    break
        if tool_name == "memory" and "text" not in args and "query" in args:
            args["text"] = args["query"]
        return args

    def _extract_from_code_fence(self, text: str) -> list[str]:
        fenced = re.findall(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        return [block.strip() for block in fenced if block.strip()]

    def _extract_from_raw_json(self, text: str) -> list[str]:
        blocks = re.findall(r"\{[^{}]*\btool_name\b[^{}]*\}", text, re.DOTALL)
        if blocks:
            return [block.strip() for block in blocks]
        blocks = re.findall(r"\{[^{}]*\btool\b[^{}]*\}", text, re.DOTALL)
        return [block.strip() for block in blocks]

    def _extract_with_bracket_matching(self, text: str) -> list[str]:
        candidates: list[str] = []
        for start, end in self._find_json_objects(text):
            snippet = text[start:end]
            if "tool_name" in snippet or "\"tool\"" in snippet or "'tool'" in snippet:
                candidates.append(snippet.strip())
        return candidates

    def _extract_with_repair(self, text: str) -> list[str]:
        candidates: list[str] = []
        for start, end in self._find_json_objects(text):
            snippet = text[start:end]
            if "tool_name" in snippet or "\"tool\"" in snippet or "'tool'" in snippet:
                candidates.append(self._repair_json(snippet))
        if not candidates and ("tool_name" in text or "\"tool\"" in text):
            candidates.append(self._repair_json(text))
        return [c for c in candidates if c.strip()]

    def _find_json_objects(self, text: str) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        stack: list[int] = []
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if ch == "\\" and not escape:
                escape = True
                continue
            if ch == "\"" and not escape:
                in_string = not in_string
            escape = False

            if in_string:
                continue
            if ch == "{":
                stack.append(idx)
            elif ch == "}" and stack:
                start = stack.pop()
                positions.append((start, idx + 1))

        return sorted(positions, key=lambda pair: pair[0])

    def _repair_json(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
        cleaned = re.sub(r"\btool\s*:", '"tool_name":', cleaned)
        cleaned = re.sub(r"\bargs\s*:", '"tool_args":', cleaned)
        return cleaned

    def _log_failure(self, raw_text: str, errors: list[str]) -> None:
        if not errors:
            errors = ["No valid tool call found"]
        error_block = " | ".join(errors)
        self._logger.warning(
            "Tool parse failure: %s\nRAW_OUTPUT:\n%s\n",
            error_block,
            raw_text,
        )
        for handler in self._logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def _build_logger(self, log_dir: str) -> logging.Logger:
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f"output_parser.{id(self)}")

        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_dir, "output_parser.log")
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
                return logger

        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
